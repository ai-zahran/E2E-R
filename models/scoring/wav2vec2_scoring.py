import logging
import speechbrain as sb
import torch

from result_logger.result_logger import ResultLogger
from speechbrain.utils import hpopt as hp

logger = logging.getLogger(__name__)

MAX_SCORE = 2.0


class ScorerWav2vec2(sb.Brain):
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

        if stage == sb.Stage.TRAIN:
            if self.hparams.use_augmentation is True and hasattr(self.hparams, "augmentation"):
                wavs = self.hparams.augmentation(wavs, wav_lens)

        feats = self.hparams.wav2vec2(wavs)
        x = self.modules.enc(feats)

        e_in_canonical = self.modules.emb(phns_canonical_bos)
        h_scoring, _ = self.modules.dec(e_in_canonical, x, wav_lens)

        # Computing phone representations for pronounced and canonical phones
        phone_rep_pred = self.modules.scorer_nn(h_scoring)
        emb_actual = self.modules.emb_scorer(phns_canonical_eos)
        emb_actual = self.modules.scorer_nn(emb_actual)

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
        ids = batch.id
        phn, phn_lens = batch.phn_canonical_encoded
        scores_actual, _ = batch.scores_list

        if hasattr(self.modules, "env_corrupt") and stage == sb.Stage.TRAIN:
            phn_lens = torch.cat([phn_lens, phn_lens], dim=0)

        scores_actual = scores_actual.unsqueeze(2)
        scores_pred = scores_pred[:, :-1].unsqueeze(2)

        loss = self.hparams.score_cost(scores_actual, scores_pred, phn_lens)

        # Rescale and round scores for final evaluation.
        scores_pred = self.rescale_scores(scores_pred)
        scores_actual = self.rescale_scores(scores_actual)

        # Record losses for posterity
        if stage != sb.Stage.TRAIN:
            self.distance_scoring_metrics.append(ids, scores_pred, scores_actual, phn, phn_lens,
                                                 self.label_encoder.decode_ndim)

        # Save predictions to compute MSE and PCC in the end of the stage.
        real_length_prediction_seq = self.get_real_length_sequences(scores_pred, phn_lens)
        real_length_scores = self.get_real_length_sequences(scores_actual, phn_lens)

        real_length_prediction_seq = torch.concat(real_length_prediction_seq, 0)
        real_length_scores = torch.concat(real_length_scores, 0)

        self.stage_preds.append(real_length_prediction_seq.detach().cpu())
        self.stage_scores.append(real_length_scores.detach().cpu())
        self.stage_preds_rounded.append(self.round_scores(real_length_prediction_seq).detach().cpu())
        self.stage_scores_rounded.append(self.round_scores(real_length_scores).detach().cpu())

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

        if stage != sb.Stage.TRAIN:
            self.distance_scoring_metrics = self.hparams.scoring_stats_dist()

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
            with open(self.hparams.scoring_dist_file, "w") as w:
                w.write("Score loss stats:\n")
                self.distance_scoring_metrics.write_stats(w)
                logger.info(f"Scoring stats written to file {self.hparams.scoring_dist_file}")

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
