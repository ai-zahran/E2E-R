import os
import shutil
import speechbrain as sb
import pandas as pd


class ResultLogger:
    """ A logger to curate training and validation results into one place."""

    def __init__(self, hparams):
        """Initializes the result logger."""
        self.results_file_path = hparams.results_file
        self.exp_metadata_file = hparams.exp_metadata_file
        self.number_of_epochs = hparams.number_of_epochs

        self.training_type = hparams.training_type
        self.model_task = hparams.model_task
        self.exp_description = hparams.exp_description

        # Hyperparameters file for the experiment.
        self.hparams_file_path = os.path.join(hparams.output_folder, "hyperparams.yaml")

        # Create an auto-incrementing ID for this experiment.
        if os.path.exists(self.exp_metadata_file):
            exp_metadata_df = pd.read_csv(self.exp_metadata_file)
            self.exp_id = exp_metadata_df["exp_id"].max() + 1
        else:
            self.exp_id = 1
        hparams.checkpointer.recoverables['exp_id'] = self.exp_id

        # Pre-trained model ID
        if self.training_type == "fine_tuning":
            pretrained_model_exp_id_file_path = os.path.join(hparams.pretrained_model_folder, "save", "best", "exp_id")
            with open(pretrained_model_exp_id_file_path, 'r') as f:
                self.pretrained_model_exp_id = int(f.readlines()[0])
        else:
            self.pretrained_model_exp_id = None

        # Files to save parameters and per epoch results.
        self.epoch_results_dir_path = hparams.epoch_results_dir  # Directory to save results per epoch.
        self.params_dir_path = hparams.params_dir  # Directory to save parameters file.
        self.epoch_results_file_path = os.path.join(self.epoch_results_dir_path, f"{self.exp_id}.csv")
        self.hparams_out_file_path = os.path.join(self.params_dir_path, f"{self.exp_id}.yaml")

        os.makedirs(self.epoch_results_dir_path, exist_ok=True)
        os.makedirs(self.params_dir_path, exist_ok=True)

        self.epoch_results = []
        self.final_results = {"exp_id": self.exp_id}

    def log_results(self, stage, results):
        """ Logs results for the given stage (training results and validation results in VALID stage, and test results
        in TEST stage)."""
        if stage == sb.Stage.VALID:
            epoch_result = {"epoch": len(self.epoch_results)}
            for k, v in results.items():
                epoch_result[k] = v
            self.epoch_results.append(epoch_result)

        elif stage == sb.Stage.TEST:
            # Log final results.
            for k, v in results.items():
                self.final_results[k] = v
            self.log_experiment()
            self.log_epoch_results()

    def log_experiment(self):
        """ Logs the experiment metadata and final results. """
        results_df = pd.DataFrame([self.final_results])
        exp_metadata_df = pd.DataFrame([{
            "exp_id": self.exp_id,
            "exp_type": '_'.join([self.model_task, self.training_type]),
            "exp_description": self.exp_description,
            "pretrained_model_exp_id": self.pretrained_model_exp_id
        }])

        if os.path.exists(self.exp_metadata_file):
            old_exp_metadata_df = pd.read_csv(self.exp_metadata_file)
            exp_metadata_df = pd.concat([old_exp_metadata_df, exp_metadata_df])
        exp_metadata_df.to_csv(self.exp_metadata_file, index=False, na_rep='NULL')

        if os.path.exists(self.results_file_path):
            old_results_df = pd.read_csv(self.results_file_path)
            results_df = pd.concat([old_results_df, results_df])
        results_df.to_csv(self.results_file_path, index=False, na_rep='NULL')

        shutil.copyfile(self.hparams_file_path, self.hparams_out_file_path)

    def log_epoch_results(self):
        """Log training results per epoch."""
        for i, epoch_result in enumerate(self.epoch_results):
            epoch_result["epoch"] = i
        epoch_results_df = pd.DataFrame(self.epoch_results)
        epoch_results_df.to_csv(self.epoch_results_file_path, index=False, na_rep='NULL')
