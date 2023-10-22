import logging
import speechbrain as sb
import torch

from result_logger.result_logger import ResultLogger
from speechbrain.utils import hpopt as hp

logger = logging.getLogger(__name__)

MAX_SCORE = 2.0


class ScorerWav2vec2LSTM(sb.Brain):
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
        phns_canonical_bos, _ = batch.phn_canonical_encoded_bos
        phns_canonical_eos, _ = batch.phn_canonical_encoded_eos
        phns_canonical, phns_canonical_lens = batch.phn_canonical_encoded
        phns_ali = batch.phn_ali_list
        phns_ali_start = batch.phn_ali_start_list
        phns_ali_duration = batch.phn_ali_duration_list

        feats = self.hparams.wav2vec2(wavs)
        x = self.modules.enc(feats)

        # Get utterance duration in sec (wav shape * wav len / sampling_rate), sr is 16Khz
        utt_durations = (wav_lens * wavs.shape[1])

        # Phone representations for canonical phones
        emb_actual = self.modules.emb_scorer(phns_canonical)

        enc_len = torch.round(x.shape[1] * wav_lens).long()

        # Loop over utterances
        x_rearranged = []
        emb_actual_rearranged = []
        for utt_index in range(phns_canonical_lens.shape[0]):

            emb_actual_trunc_len = phns_canonical_lens[utt_index] * emb_actual.shape[1]
            emb_actual_trunc = emb_actual[utt_index][:round(emb_actual_trunc_len.item())]
            emb_actual_rearranged.append(emb_actual_trunc)

            utt_phn_x = []

            # Loop over phones alignments
            for phn_index in range(len(phns_ali[utt_index])):
                # If phone is not silent, get start time and end time (start + duration)
                if phns_ali[utt_index][phn_index] != 'sil':
                    # Get sample indices ((time / utt_len) * (feat vector shape * wav_len)) and append to a list of
                    # phone feats
                    phn_start_index = torch.round(
                        phns_ali_start[utt_index][phn_index] * 16000 / utt_durations[utt_index] * enc_len[utt_index]
                    )
                    phn_end_index = torch.round(
                        (phns_ali_start[utt_index][phn_index] + phns_ali_duration[utt_index][phn_index]) * 16000
                        / utt_durations[utt_index] * enc_len[utt_index]
                    )

                    if phn_end_index == phn_start_index:
                        phn_end_index += 1
                    if phn_end_index > x.shape[1]:
                        logger.warning(f"End index for phoneme at index {phn_index} in utterance {batch.id[utt_index]} "
                                       f"exceeds the shape of encoder output.")
                        phn_end_index = x.shape[1]
                        phn_start_index = phn_start_index if phn_start_index < phn_end_index else phn_end_index - 1

                    phn_x = x[utt_index][int(phn_start_index):int(phn_end_index)]
                    utt_phn_x.append(phn_x)
            x_rearranged.append(utt_phn_x)

        # Concat phone feats and pass through LSTM
        x_phn_lens = torch.tensor([len(utt_phn_x) - 1 for utt_x in x_rearranged for utt_phn_x in utt_x]).to(self.device)
        x_phns = torch.nn.utils.rnn.pad_sequence([utt_phn_x for utt_x in x_rearranged
                                                  for utt_phn_x in utt_x], batch_first=True)
        phone_rep_seqs = self.modules.phoneme_rep_lstm(x_phns)
        phone_rep_pred = torch.squeeze(phone_rep_seqs.gather(1, x_phn_lens.view(-1, 1, 1).expand(
            phone_rep_seqs.shape[0], 1, phone_rep_seqs.shape[2])), 1)

        emb_actual = torch.concat(emb_actual_rearranged, 0)
        emb_actual = self.modules.scorer_nn(emb_actual)
        phone_rep_pred = self.modules.scorer_nn(phone_rep_pred)

        if len(phone_rep_pred) != len(emb_actual):
            print("Mismatch in length.")
            print("Phns ali:\n", batch.phn_ali_list)
            print("Phns:\n", batch.phn_list)
            print("Lengths of Phns ali:\n", [len(phn_ali) for phn_ali in batch.phn_ali_list])
            print("Lengths of Phns:\n", [len(phn) for phn in batch.phn_list])

        # Computing similarity
        if self.hparams.similarity_calc == "cosine":
            # Cosine similarity
            scores_pred = torch.nn.functional.cosine_similarity(
                phone_rep_pred, emb_actual, dim=len(phone_rep_pred.shape) - 1)
        elif self.hparams.similarity_calc == "euclidean":
            # Normalized Euclidean similarity (NES)
            scores_pred = 1.0 - 0.5 * (phone_rep_pred - emb_actual).var(dim=2) / \
                          (phone_rep_pred.var(dim=2) + emb_actual.var(dim=2))
        else:
            scores_pred = self.modules.scorer_similarity_nn(torch.concat([phone_rep_pred, emb_actual], dim=2)) \
                .view(self.hparams.batch_size, emb_actual.shape[1])

        return scores_pred, wav_lens

    def rescale_scores(self, scores):
        """Rescales scores from range [0, 1] to range [0, 2]."""
        return MAX_SCORE * scores

    def round_scores(self, scores):
        """Rescales scores to the nearest integer."""
        return torch.round(torch.minimum(torch.maximum(scores, torch.full_like(scores, 0)), torch.full_like(scores, 2)))

    def get_real_length_sequences(self, seq, lens):
        """Return the sequences with their real length."""
        seqs = []
        for i in range(len(lens)):
            seq_len = round((lens[i] * seq.shape[1]).item())
            seqs.append(seq[i, :seq_len].squeeze())
        return seqs

    def compute_objectives(self, predictions, batch, stage):
        """Given the network predictions and targets computed the NLL loss."""
        scores_pred, _ = predictions
        scores_actual, _ = batch.scores_list

        scores_actual = scores_actual.unsqueeze(2)

        _, phn_canonical_lens = batch.phn_canonical_encoded
        scores_actual_lens = scores_actual.shape[1] * phn_canonical_lens
        scores_actual_rearranged = []
        for i in range(scores_actual.shape[0]):
            scores_actual_rearranged.append(scores_actual[i, :round(scores_actual_lens[i].item())].squeeze())
        scores_actual = torch.concat(scores_actual_rearranged, dim=0)

        loss = self.hparams.score_cost(scores_actual.view(-1, 1), scores_pred.view(-1, 1))

        # Rescale and round scores for final evaluation.
        scores_pred = self.rescale_scores(scores_pred)
        scores_actual = self.rescale_scores(scores_actual)

        self.stage_preds.append(scores_pred.detach().cpu())
        self.stage_scores.append(scores_actual.detach().cpu())
        self.stage_preds_rounded.append(self.round_scores(scores_pred).detach().cpu())
        self.stage_scores_rounded.append(self.round_scores(scores_actual).detach().cpu())

        return loss

    def fit_batch(self, batch):
        """Fit one batch, override to do multiple updates.

        The default implementation depends on a few methods being defined
        with a particular behavior:

        * ``compute_forward()``
        * ``compute_objectives()``

        Also depends on having optimizers passed at initialization.

        Arguments
        ---------
        batch : list of torch.Tensors
            Batch of data to use for training. Default implementation assumes
            this batch has two elements: inputs and targets.

        Returns
        -------
        detached loss
        """
        # Managing automatic mixed precision
        if self.auto_mix_prec:

            self.wav2vec_optimizer.zero_grad()
            self.asr_optimizer.zero_grad()
            self.scorer_optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                outputs = self.compute_forward(batch, sb.Stage.TRAIN)
                loss = self.compute_objectives(outputs, batch, sb.Stage.TRAIN)

            self.scaler.scale(loss).backward()
            if self.optimizer_step > self.hparams.warmup_steps_wav2vec:
                self.scaler.unscale_(self.wav2vec_optimizer)
            if self.optimizer_step > self.hparams.warmup_steps_asr:
                self.scaler.unscale_(self.asr_optimizer)
            self.scaler.unscale_(self.scorer_optimizer)

            if self.check_gradients(loss):
                if self.optimizer_step > self.hparams.warmup_steps_wav2vec and not self.hparams.wav2vec2.freeze:
                    self.scaler.step(self.wav2vec_optimizer)
                if self.optimizer_step > self.hparams.warmup_steps_asr:
                    self.scaler.step(self.asr_optimizer)
                self.scaler.step(self.scorer_optimizer)

            self.scaler.update()
        else:
            outputs = self.compute_forward(batch, sb.Stage.TRAIN)

            loss = self.compute_objectives(outputs, batch, sb.Stage.TRAIN)
            loss.backward()

            if self.check_gradients(loss):
                if self.optimizer_step > self.hparams.warmup_steps_wav2vec:
                    self.wav2vec_optimizer.step()
                if self.optimizer_step > self.hparams.warmup_steps_asr:
                    self.asr_optimizer.step()
                self.scorer_optimizer.step()

            self.wav2vec_optimizer.zero_grad()
            self.asr_optimizer.zero_grad()
            self.scorer_optimizer.zero_grad()

        self.optimizer_step += 1
        return loss.detach().cpu()

    def evaluate_batch(self, batch, stage):
        """Computations needed for validation/test batches"""
        predictions = self.compute_forward(batch, stage=stage)
        loss = self.compute_objectives(predictions, batch, stage=stage)
        return loss.detach()

    def on_stage_start(self, stage, epoch):
        """Gets called when a stage (either training, validation, test) starts."""
        self.score_metrics_mse = self.hparams.score_stats_mse()
        self.stage_preds = []
        self.stage_preds_rounded = []
        self.stage_scores = []
        self.stage_scores_rounded = []

    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of an epoch."""
        stage_preds = torch.concat(self.stage_preds, 0)
        stage_scores = torch.concat(self.stage_scores, 0)
        stage_preds_rounded = torch.concat(self.stage_preds_rounded, 0)
        stage_scores_rounded = torch.concat(self.stage_scores_rounded, 0)

        stage_pcc = torch.corrcoef(torch.stack([stage_preds, stage_scores]))[0, 1].item()
        stage_mse = torch.nn.functional.mse_loss(stage_preds, stage_scores).item()
        stage_pcc_rounded = torch.corrcoef(torch.stack([stage_preds_rounded, stage_scores_rounded]))[0, 1].item()
        stage_mse_rounded = torch.nn.functional.mse_loss(stage_preds_rounded, stage_scores_rounded).item()

        results_to_log = dict()

        if stage == sb.Stage.TRAIN:
            self.train_loss = stage_loss
            self.train_pcc = stage_pcc
            self.train_mse = stage_mse
        else:
            stats = {"loss": stage_loss, "error": stage_mse}

        if stage == sb.Stage.VALID:
            scoring_error = stage_mse
            old_lr_asr, new_lr_asr = self.hparams.lr_annealing_asr(scoring_error)
            old_lr_scorer, new_lr_scorer = self.hparams.lr_annealing_scorer(scoring_error)
            old_lr_wav2vec, new_lr_wav2vec = self.hparams.lr_annealing_wav2vec(scoring_error)
            sb.nnet.schedulers.update_learning_rate(self.asr_optimizer, new_lr_asr)
            sb.nnet.schedulers.update_learning_rate(self.scorer_optimizer, new_lr_scorer)
            sb.nnet.schedulers.update_learning_rate(self.wav2vec_optimizer, new_lr_wav2vec)

            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch, "lr_asr": old_lr_asr, "lr_scorer": old_lr_scorer,
                            "lr_wav2vec": old_lr_wav2vec},
                train_stats={"loss": self.train_loss},
                valid_stats={
                    "loss": stage_loss,
                    "scoring error": stage_mse,
                    "PCC": stage_pcc,
                    "scoring error (rounded)": stage_mse_rounded,
                    "PCC (rounded)": stage_pcc_rounded,
                },
            )
            if self.hparams.ckpt_enable:
                self.checkpointer.save_and_keep_only(
                    meta={"scoring_error": scoring_error}, min_keys=["scoring_error"]
                )

            results_to_log["train_loss"] = self.train_loss
            results_to_log["valid_loss"] = stage_loss
            results_to_log["valid_pcc"] = stage_pcc
            results_to_log["valid_mse"] = stage_mse
            results_to_log["valid_pcc_rounded"] = stage_pcc_rounded
            results_to_log["valid_mse_rounded"] = stage_mse_rounded

            print("Reporting the following stats to hpopt", stats)
            if hasattr(self.hparams, "optimizing_hps") and self.hparams.optimizing_hps == True:
                hp.report_result(stats)

        if stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats={"loss": stage_loss, "scoring error": stage_mse, "PCC": stage_pcc,
                            "scoring error (rounded)": stage_mse_rounded, "PCC (rounded)": stage_pcc_rounded},
            )

            results_to_log["test_loss"] = stage_loss
            results_to_log["test_pcc"] = stage_pcc
            results_to_log["test_mse"] = stage_mse
            results_to_log["test_pcc_rounded"] = stage_pcc_rounded
            results_to_log["test_mse_rounded"] = stage_mse_rounded

        self.result_logger.log_results(stage, results_to_log)

    def init_optimizers(self):
        """Initializes the wav2vec2 optimizer and model optimizer"""
        self.wav2vec_optimizer = self.hparams.wav2vec_opt_class(
            self.modules.wav2vec2.parameters()
        )
        self.asr_optimizer = self.hparams.asr_opt_class(
            self.hparams.model.parameters()
        )
        self.scorer_optimizer = self.hparams.scorer_opt_class(
            self.hparams.model_scorer.parameters()
        )

        if self.checkpointer is not None:
            self.checkpointer.add_recoverable("wav2vec_opt", self.wav2vec_optimizer)
            self.checkpointer.add_recoverable("asr_opt", self.asr_optimizer)
            self.checkpointer.add_recoverable("scorer_opt", self.scorer_optimizer)
