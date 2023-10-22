import speechbrain as sb
import torch

from result_logger.result_logger import ResultLogger


class APRFeatBased(sb.Brain):
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
        phns, _ = batch.phn_encoded

        if stage == sb.Stage.TRAIN:
            if hasattr(self.hparams, "env_corrupt"):
                wavs_noise = self.hparams.env_corrupt(wavs, wav_lens)
                wavs = torch.cat([wavs, wavs_noise], dim=0)
                wav_lens = torch.cat([wav_lens, wav_lens])
                phns_bos = torch.cat([phns_bos, phns_bos])
            if hasattr(self.hparams, "augmentation"):
                wavs = self.hparams.augmentation(wavs, wav_lens)

        feats = self.hparams.feat_extractor(wavs)
        feats = self.modules.normalize(feats, wav_lens)
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

        if hasattr(self.modules, "env_corrupt") and stage == sb.Stage.TRAIN:
            phns = torch.cat([phns, phns], dim=0)
            phn_lens = torch.cat([phn_lens, phn_lens], dim=0)
            phns_eos = torch.cat([phns_eos, phns_eos], dim=0)
            phn_lens_eos = torch.cat([phn_lens_eos, phn_lens_eos], dim=0)

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
        """Train the parameters given a single batch in input"""
        predictions = self.compute_forward(batch, sb.Stage.TRAIN)
        loss = self.compute_objectives(predictions, batch, sb.Stage.TRAIN)
        loss.backward()
        if self.check_gradients(loss):
            self.optimizer.step()
        self.optimizer.zero_grad()
        return loss.detach()

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

        results_to_log = dict()

        if stage == sb.Stage.VALID:
            old_lr, new_lr = self.hparams.lr_annealing(per)
            sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)

            valid_ctc_loss = self.ctc_metrics.summarize("average")
            valid_seq_loss = self.seq_metrics.summarize("average")

            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch, "lr": old_lr},
                train_stats={"loss": self.train_loss},
                valid_stats={
                    "loss": stage_loss,
                    "ctc_loss": valid_ctc_loss,
                    "seq_loss": valid_seq_loss,
                    "PER": per,
                },
            )
            self.checkpointer.save_and_keep_only(
                meta={"PER": per}, min_keys=["PER"]
            )
            results_to_log["train_loss"] = self.train_loss
            results_to_log["valid_loss"] = stage_loss
            results_to_log["valid_ctc_loss"] = valid_ctc_loss
            results_to_log["valid_seq_loss"] = valid_seq_loss
            results_to_log["valid_per"] = per

        if stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats={"loss": stage_loss, "PER": per},
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
        self.optimizer = self.hparams.adam_opt_class(
            self.hparams.model.parameters()
        )

        if self.checkpointer is not None:
            self.checkpointer.add_recoverable("adam_opt", self.optimizer)
