from data_prep_utils.dataset_preparation.speechocean762_prepare import \
    prepare_speechocean762_for_pronunciation_scoring, prepare_speechocean762_for_phoneme_recognition
from data_prep_utils.dataset_preparation.timit_prepare import prepare_timit
from speechbrain.utils.distributed import run_on_main


def prepare_dataset(hparams):
    kwargs = {"data_folder": hparams["data_folder"],
              "save_json_train": hparams["train_annotation"],
              "save_json_valid": hparams["valid_annotation"],
              "save_json_test": hparams["test_annotation"],
              "skip_prep": hparams["skip_prep"]
              }
    prep_func = prepare_timit  # Default value

    if hparams["dataset_name"] == "speechocean762":
        if "model_task" in hparams and hparams["model_task"] == "apr":
            prep_func = prepare_speechocean762_for_phoneme_recognition
            if "train_split_ratio" in hparams:
                kwargs["train_split_ratio"] = hparams["train_split_ratio"]
        else:
            prep_func = prepare_speechocean762_for_pronunciation_scoring

    elif hparams["dataset_name"] == "timit":
        prep_func = prepare_timit
        kwargs["uppercase"] = True

    run_on_main(prep_func, kwargs=kwargs)
