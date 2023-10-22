#!/usr/bin/env python3
"""Generic training script

To run this recipe, do the following:
> python train.py hparams/train.yaml --data_folder /path/to/data

Authors
 * Ahmed Zahran 2023
"""

import logging
import os
import speechbrain as sb
import sys
from data_prep_utils.dataset_preparation.dataset_preparation import prepare_dataset
from data_prep_utils.dataset_preparation.dataio_prep import dataio_prep
from hyperpyyaml import load_hyperpyyaml
from models.brain import get_brain_class
from speechbrain.pretrained.training import save_for_pretrained

logger = logging.getLogger(__name__)


def int_save(obj: int, path):
    with open(path, 'w') as f:
        f.write(str(obj))


def int_recovery_checkpointer(obj, loadpath, end_of_epoch, device):
    del end_of_epoch  # Unused
    with open(loadpath, 'r') as f:
        obj = int(f.readlines()[0].strip())


def int_recovery_pretrainer(obj, loadpath, device):
    with open(loadpath, 'r') as f:
        obj = int(f.readlines()[0].strip())


if __name__ == "__main__":
    # CLI:
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    label_encoder_path = None

    # Load hyperparameters file with command-line overrides
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)
    hparams["wer_file"] = os.path.join(hparams["output_folder"], "wer.txt")
    hparams["hyp_file"] = os.path.join(hparams["output_folder"], "hyp.txt")

    # Register hooks for saving and loading experiment IDs.
    hparams["checkpointer"].custom_save_hooks.update({
        'exp_id': int_save,
        'pretrained_model_exp_id': int_save
    })
    hparams["checkpointer"].custom_load_hooks.update({
        'exp_id': int_recovery_checkpointer,
        'pretrained_model_exp_id': int_recovery_checkpointer
    })

    # Load the pre-trained model
    if hparams["training_type"] == "fine_tuning":
        hparams["pretrained_model"].add_custom_hooks({
                'exp_id': int_recovery_pretrainer,
        })
        hparams["pretrained_model"].set_collect_in(os.path.join(hparams['pretrained_model_folder'], "save", "best"))

        if "multi_task_pretrained_model" in hparams and hparams["multi_task_pretrained_model"] is True:
            hparams["pretrained_model"].add_loadables({
                "model_scorer": hparams["model_scorer"]
            })

        hparams["pretrained_model"].load_collected()

    # Load the label encoder
    if "pretrained_model_folder" in hparams or hparams["model_task"] == "scoring":
        logger.info(f"Loading label encoder from pretrained model folder {hparams['pretrained_model_folder']}.")
        label_encoder_path = os.path.join(hparams["pretrained_model_folder"], "save", "label_encoder.txt")
    
    if "label_encoder_path" in hparams and hparams["label_encoder_path"] is not None:
        logger.info(f"Loading label encoder from path {hparams['label_encoder_path']}.")
        label_encoder_path = hparams["label_encoder_path"]

    if label_encoder_path is not None:
        label_encoder = sb.dataio.encoder.CTCTextEncoder.from_saved(label_encoder_path)

    # Initialize ddp (useful only for multi-GPU DDP training)
    sb.utils.distributed.ddp_init_group(run_opts)

    prepare_dataset(hparams)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    if label_encoder_path is not None:
            train_data, valid_data, test_data, label_encoder = dataio_prep(hparams, label_encoder)
    else:
        train_data, valid_data, test_data, label_encoder = dataio_prep(hparams)

    # Trainer initializer
    brain_class = get_brain_class(hparams)

    # Figure out training type (training or fine-tuning)
    brain = brain_class(
        modules=hparams["modules"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    brain.label_encoder = label_encoder

    # Training/validation loop
    brain.fit(
        brain.hparams.epoch_counter,
        train_data,
        valid_data,
        train_loader_kwargs=hparams["train_dataloader_opts"],
        valid_loader_kwargs=hparams["valid_dataloader_opts"],
    )

    # Test
    brain.evaluate(
        test_data,
        min_key="PER",
        test_loader_kwargs=hparams["test_dataloader_opts"],
    )
    # Save the best model to be used for fine-tuning
    save_for_pretrained(hparams)
