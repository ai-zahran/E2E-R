import speechbrain as sb
import torch

from result_logger.result_logger import ResultLogger
from speechbrain.utils import hpopt as hp


class APRWav2vec2(sb.Brain):
    def __init__(  # noqa: C901
            self,
            modules=None,
            opt_class=None,
            hparams=None,
            run_opts=None,
            checkpointer=None,
            profiler=None,
    ):
        super().__init__(modules, opt_class, hparams, run_opts, checkpointer, profiler)
        self.result_logger = ResultLogger(self.hparams)

    def compute_forward(self, batch, stage):
        """Given an input batch it computes the phoneme probabilities."""
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig
        phns_bos, _ = batch.phn_encoded_bos

        if stage == sb.Stage.TRAIN:
            if hasattr(self.hparams, "augmentation"):
                wavs = self.hparams.augmentation(wavs, wav_lens)

        feats = self.modules.wav2vec2(wavs)
        x = self.modules.enc(feats)

        # output layer for ctc log-probabilities
        logits = self.modules.ctc_lin(x)
        p_ctc = self.hparams.log_softmax(logits)

        e_in = self.modules.emb(phns_bos)
        h, _ = self.modules.dec(e_in, x, wav_lens)

        # output layer for seq2seq log-probabilities
        logits = self.modules.seq_lin(h)
        p_seq = self.hparams.log_softmax(logits)

        if stage == sb.Stage.VALID:
            hyps, scores = self.hparams.greedy_searcher(x, wav_lens)
            return p_ctc, p_seq, wav_lens, hyps

        elif stage == sb.Stage.TEST:
            hyps, scores = self.hparams.beam_searcher(x, wav_lens)
            return p_ctc, p_seq, wav_lens, hyps

        return p_ctc, p_seq, wav_lens

    def compute_objectives(self, predictions, batch, stage):
        """Given the network predictions and targets computed the NLL loss."""
        if stage == sb.Stage.TRAIN:
            p_ctc, p_seq, wav_lens = predictions
        else:
            p_ctc, p_seq, wav_lens, hyps = predictions

        ids = batch.id
        phns_eos, phn_lens_eos = batch.phn_encoded_eos
        phns, phn_lens = batch.phn_encoded

        loss_ctc = self.hparams.ctc_cost(p_ctc, phns, wav_lens, phn_lens)
        loss_seq = self.hparams.seq_cost(p_seq, phns_eos, phn_lens_eos)
        loss = self.hparams.ctc_weight * loss_ctc
        loss += (1 - self.hparams.ctc_weight) * loss_seq

        # Record losses for posterity
        if stage != sb.Stage.TRAIN:
            self.ctc_metrics.append(ids, p_ctc, phns, wav_lens, phn_lens)
            self.seq_metrics.append(ids, p_seq, phns_eos, phn_lens_eos)
            self.per_metrics.append(
                ids, hyps, phns, None, phn_lens, self.label_encoder.decode_ndim,
            )

        return loss

    def fit_batch(self, batch):
        """Fit one batch, override to do multiple updates."""
        # Managing automatic mixed precision
        if self.auto_mix_prec:

            self.wav2vec_optimizer.zero_grad()
            self.adam_optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                outputs = self.compute_forward(batch, sb.Stage.TRAIN)
                loss = self.compute_objectives(outputs, batch, sb.Stage.TRAIN)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.wav2vec_optimizer)
            self.scaler.unscale_(self.adam_optimizer)

            if self.check_gradients(loss):
                self.scaler.step(self.wav2vec_optimizer)
                self.scaler.step(self.adam_optimizer)

            self.scaler.update()
        else:
            outputs = self.compute_forward(batch, sb.Stage.TRAIN)

            loss = self.compute_objectives(outputs, batch, sb.Stage.TRAIN)
            loss.backward()

            if self.check_gradients(loss):
                self.wav2vec_optimizer.step()
                self.adam_optimizer.step()

            self.wav2vec_optimizer.zero_grad()
            self.adam_optimizer.zero_grad()

        return loss.detach().cpu()

    def evaluate_batch(self, batch, stage):
        """Computations needed for validation/test batches"""
        predictions = self.compute_forward(batch, stage=stage)
        loss = self.compute_objectives(predictions, batch, stage=stage)
        return loss.detach()

    def on_stage_start(self, stage, epoch):
        """Gets called when a stage (either training, validation, test) starts."""
        self.ctc_metrics = self.hparams.ctc_stats()
        self.seq_metrics = self.hparams.seq_stats()

        if stage != sb.Stage.TRAIN:
            self.per_metrics = self.hparams.per_stats()

    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of an epoch."""
        if stage == sb.Stage.TRAIN:
            self.train_loss = stage_loss
        else:
            per = self.per_metrics.summarize("error_rate")
            stats = {"loss": stage_loss, "error": per}

        results_to_log = dict()

        if stage == sb.Stage.VALID:
            old_lr_adam, new_lr_adam = self.hparams.lr_annealing_adam(per)
            old_lr_wav2vec, new_lr_wav2vec = self.hparams.lr_annealing_wav2vec(per)
            sb.nnet.schedulers.update_learning_rate(self.adam_optimizer, new_lr_adam)
            sb.nnet.schedulers.update_learning_rate(self.wav2vec_optimizer, new_lr_wav2vec)

            valid_ctc_loss = self.ctc_metrics.summarize("average")
            valid_seq_loss = self.seq_metrics.summarize("average")

            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch, "lr_adam": old_lr_adam, "lr_wav2vec": old_lr_wav2vec, },
                train_stats={"loss": self.train_loss},
                valid_stats={
                    "Loss": stage_loss,
                    "CTC Loss": valid_ctc_loss,
                    "Seq Loss": valid_seq_loss,
                    "PER": per,
                },
            )
            if self.hparams.ckpt_enable:
                self.checkpointer.save_and_keep_only(
                    meta={"PER": per}, min_keys=["PER"]
                )

            if hasattr(self.hparams, "optimizing_hps") and self.hparams.optimizing_hps == True:
                hp.report_result(stats)

            results_to_log["train_loss"] = self.train_loss
            results_to_log["valid_loss"] = stage_loss
            results_to_log["valid_ctc_loss"] = valid_ctc_loss
            results_to_log["valid_seq_loss"] = valid_seq_loss
            results_to_log["valid_per"] = per

        if stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats={"Loss": stage_loss, "PER": per},
            )
            with open(self.hparams.wer_file, "w") as w:
                w.write("CTC loss stats:\n")
                self.ctc_metrics.write_stats(w)
                w.write("\nseq2seq loss stats:\n")
                self.seq_metrics.write_stats(w)
                w.write("\nPER stats:\n")
                self.per_metrics.write_stats(w)
                print(
                    "CTC, seq2seq, and PER stats written to file",
                    self.hparams.wer_file,
                )
            results_to_log["test_loss"] = stage_loss
            results_to_log["test_per"] = per

        self.result_logger.log_results(stage, results_to_log)

    def init_optimizers(self):
        """Initializes the wav2vec2 optimizer and model optimizer"""
        self.wav2vec_optimizer = self.hparams.wav2vec_opt_class(self.modules.wav2vec2.parameters())
        self.adam_optimizer = self.hparams.adam_opt_class(self.hparams.model.parameters())

        if self.hparams.ckpt_enable and self.checkpointer is not None:
            self.checkpointer.add_recoverable("wav2vec_opt", self.wav2vec_optimizer)
            self.checkpointer.add_recoverable("adam_opt", self.adam_optimizer)
