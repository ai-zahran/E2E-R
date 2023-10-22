import json
import logging
import os
import random
import speechbrain as sb
import torch

from collections import defaultdict
from data_prep_utils.data.utterance_metadata import UtteranceMetadata
from data_prep_utils.data.metadata_constants import MetadataConstants
from data_prep_utils.dataset_preparation.metadata_loading import load_alignments, load_kaldi_wav_scp,\
    load_kaldi_text_file, load_kaldi_utt2spk, load_utterance_metadata
from speechbrain.dataio.dataio import read_audio
from typing import Dict, List, Tuple


random.seed(10)
logger = logging.getLogger(__name__)
SAMPLERATE = 16000
PHONE_LIST = ['V', 'AE', 'Y', 'EH', 'DH', 'AW', 'P', 'JH', 'M', 'UW', 'B', 'R', 'TH', 'G', 'CH', 'UH', 'EY', 'D', 'L',
              'AO', 'Z', 'W', 'IH', 'ER', 'AA', 'AY', 'HH', 'S', 'F', 'N', 'ZH', 'K', 'OW', 'NG', 'OY', 'AH', 'SH', 'T',
              'IY']


def read_metadata(data_folder: str) -> Tuple[Dict, Dict, Dict]:
    utt_paths = dict()
    text = dict()
    utt2spk = dict()

    # Read wav.scp from data directory.
    for split_name in ["train", "test"]:
        split_dir_path = os.path.join(data_folder, split_name)
        utt_paths.update(load_kaldi_wav_scp(split_dir_path))
        utt_paths = {utt_id: os.path.join(data_folder, utt_path) for utt_id, utt_path in utt_paths.items()}
        text.update(load_kaldi_text_file(split_dir_path))
        utt2spk.update(load_kaldi_utt2spk(split_dir_path))
    return utt_paths, text, utt2spk

def get_split_utt_speaker_metadata(original_data_folder: str, split_name: str) \
        -> Tuple[Dict[str, str], Dict[str, List[str]]]:
    """Gets utterance-to-speaker and speaker-to-utterance mapping for a subset of the dataset."""
    metadata_folder = os.path.join(original_data_folder, split_name)
    utt2spk_file_path = os.path.join(metadata_folder, "utt2spk")
    utt2spk = dict()
    spk2utt = defaultdict(list)

    with open(utt2spk_file_path, 'r') as utt2spk_file:
        lines = utt2spk_file.readlines()

    for line in lines:
        utterance_id, speaker = line.strip().split()
        utt2spk[utterance_id] = speaker
        spk2utt[speaker].append(utterance_id)

    return utt2spk, spk2utt


def get_split_utterances(original_data_folder: str, train_split_ratio: float = 0.8) \
        -> Tuple[List[str], List[str], List[str]]:
    """Get utterances for training, dev and test splits."""
    utt2spk, spk2utt = get_split_utt_speaker_metadata(original_data_folder, "train")

    speakers = list(spk2utt.keys())
    random.shuffle(speakers)
    num_train_speakers = round(len(speakers) * train_split_ratio)
    train_speakers = speakers[:num_train_speakers]
    dev_speakers = speakers[num_train_speakers:]

    train_utterances = [utt for speaker in train_speakers for utt in spk2utt[speaker]]
    dev_utterances = [utt for speaker in dev_speakers for utt in spk2utt[speaker]]

    utt2spk, spk2utt = get_split_utt_speaker_metadata(original_data_folder, "test")
    speakers = list(spk2utt.keys())
    test_utterances = [utt for speaker in speakers for utt in spk2utt[speaker]]

    return train_utterances, dev_utterances, test_utterances


def prepare_speechocean762_for_phoneme_recognition(
        data_folder: str,
        save_json_train: str,
        save_json_valid: str,
        save_json_test: str,
        train_split_ratio: float = 0.8,
        skip_prep: bool = False,
):
    """Prepares the json files for the speechocean762 dataset for phoneme recognition task."""
    # Skip if needed
    if skip_prep:
        return

    if skip_prep or skip([save_json_train, save_json_valid, save_json_test]):
        logger.info("Skipping preparation, completed in previous run.")
        return

    utt_paths, text, utt2spk = read_metadata(data_folder)
    utterance_data = load_utterance_metadata(data_folder)

    phones = get_phone_set(utterance_data)
    logger.info(f"Found {len(phones)} phones in the original dataset:\n{phones}")

    # Read text from data directory.
    train_utterances, dev_utterances, test_utterances = get_split_utterances(data_folder, train_split_ratio)

    # Create JSON file with {utt_id: location, duration, text}.
    for split_utterances, save_json_file in zip([train_utterances, dev_utterances, test_utterances],
                                                [save_json_train, save_json_valid, save_json_test]):
        split_utt_paths, split_text, split_utt2spk, split_utterance_data = filter_data_by_split(
            split_utterances, utt_paths, text, utt2spk, utterance_data)

        prepare_phoneme_recognition_split_data(split_utt_paths, split_text, split_utt2spk, split_utterance_data,
                                               save_json_file)

def prepare_phoneme_recognition_split_data(
        utt_paths: Dict[str, str],
        text: Dict[str, str],
        utt2spk: Dict[str, str],
        utterance_data: Dict[str, UtteranceMetadata],
        json_file: str,
):
    """Creates the json file given a set of phone alignment pairs."""
    json_dict = dict()
    for utterance_id in utt_paths:
        signal = read_audio(utt_paths[utterance_id])
        duration = len(signal) / SAMPLERATE

        phones = utterance_data[utterance_id].phones
        scores = utterance_data[utterance_id].phone_accuracy_scores

        # If all the phones in this utterance are correctly pronounced, add it to the set
        if len([score for score in scores if round(score == 2)]) == len(scores):
            phonetic_transcript = ' '.join(phones).lower()
            json_dict[utterance_id] = {
                "wav": utt_paths[utterance_id],
                "wrd": text[utterance_id],
                "spk_id": utt2spk[utterance_id],
                "phn": phonetic_transcript,
                "phn_canonical": phonetic_transcript,
                "scores": ' '.join([str(s) for s in scores]),
                "duration": duration
            }

    # Writing the dictionary to the json file
    with open(json_file, mode="w") as json_f:
        json.dump(json_dict, json_f, indent=2)

    logger.info(f"{json_file} successfully created!")

def prepare_speechocean762_for_pronunciation_scoring(
        data_folder: str,
        save_json_train: str,
        save_json_valid: str,
        save_json_test: str,
        train_split_ratio: float = 0.8,
        skip_prep: bool = False,
):
    """Prepares the json files for the speechocean762 dataset for pronunciation scoring task."""
    # Skip if needed
    if skip_prep:
        return

    if skip_prep or skip([save_json_train, save_json_valid, save_json_test]):
        logger.info("Skipping preparation, completed in previous run.")
        return

    utt_paths, text, utt2spk = read_metadata(data_folder)
    utterance_data = load_utterance_metadata(data_folder)
    alignments = load_alignments(data_folder)

    phones = get_phone_set(utterance_data)
    logger.info(f"Found {len(phones)} phones in the original dataset:\n{phones}")

    # Read text from data directory.
    train_utterances, dev_utterances, test_utterances = get_split_utterances(data_folder, train_split_ratio)

    # Create JSON file with {utt_id: location, duration, text}.
    for split_utterances, save_json_file in zip([train_utterances, dev_utterances, test_utterances],
                                                [save_json_train, save_json_valid, save_json_test]):
        split_utt_paths, split_utt2spk, split_utterance_data, split_alignments = \
            filter_data_by_split(split_utterances, utt_paths, utt2spk, utterance_data, alignments)

        prepare_pronunciation_scoring_split_data(split_utt_paths, split_utt2spk, split_utterance_data,
                                                 split_alignments, save_json_file)


def get_phone_set(text_phone: Dict[str, UtteranceMetadata]):
    phones = set()
    for utt_id in text_phone:
        phones.update(set(text_phone[utt_id].phones))
    return phones


def filter_data_by_split(split_utterances, *args):
    filtered_args = []
    for arg in args:
        filtered_arg = dict()
        for utt_id in split_utterances:
            filtered_arg[utt_id] = arg[utt_id]
        filtered_args.append(filtered_arg)
    return filtered_args


def prepare_pronunciation_scoring_split_data(
        utterance_paths: Dict[str, str],
        utt2spk: Dict[str, str],
        utterance_metadata: Dict[str, UtteranceMetadata],
        alignments: Dict[str, List[Tuple[str, float, float]]],
        json_file: str,
):
    """Creates the json file."""
    json_dict = dict()
    for utterance_id in utterance_paths:
        signal = read_audio(utterance_paths[utterance_id])
        duration = len(signal) / SAMPLERATE

        json_dict[utterance_id] = {
            "utterance_id": utterance_id,
            "wav": utterance_paths[utterance_id],
            "text": ' '.join(utterance_metadata[utterance_id].words).lower(),
            "spk_id": utt2spk[utterance_id],
            "phn": ' '.join(utterance_metadata[utterance_id].phones).lower(),
            "phn_canonical": ' '.join(utterance_metadata[utterance_id].phones).lower(),
            "scores": ' '.join([str(s) for s in utterance_metadata[utterance_id].phone_accuracy_scores]),
            "duration": duration
        }

        if len(alignments) != 0:
            alignment = alignments[utterance_id]
            json_dict[utterance_id]["phn_ali"] = ' '.join([ali_item[0] for ali_item in alignment])
            json_dict[utterance_id]["phn_ali_start"] = ' '.join([str(ali_item[1]) for ali_item in alignment])
            json_dict[utterance_id]["phn_ali_duration"] = ' '.join([str(ali_item[2]) for ali_item in alignment])

    # Writing the dictionary to the json file
    with open(json_file, mode="w") as json_f:
        json.dump(json_dict, json_f, indent=2)

    logger.info(f"{json_file} successfully created!")


def skip(annotations):
    """Detects if the data_preparation has been already done. If the preparation has been done, we can skip it."""
    skip = True
    for annotation in annotations:
        if not os.path.isfile(annotation):
            skip = False
            break
    return skip


def create_score_tensor_from_string(scores_string, min_score_limit, max_score_limit):
    return torch.FloatTensor([(float(s) - min_score_limit)
                              / max_score_limit for s in scores_string.strip().split()])


def dataio_prep_speechocean762(hparams, label_encoder=None):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions."""

    data_folder = hparams["data_folder"]
    # 1. Declarations:
    train_data = sb.dataio.dataset.DynamicItemDataset.from_json(
        json_path=hparams["train_annotation"],
        replacements={"data_root": data_folder},
    )
    if hparams["sorting"] == "ascending":
        # we sort training data to speed up training and get better results.
        train_data = train_data.filtered_sorted(sort_key="duration")
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["train_dataloader_opts"]["shuffle"] = False

    elif hparams["sorting"] == "descending":
        train_data = train_data.filtered_sorted(
            sort_key="duration", reverse=True
        )
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["train_dataloader_opts"]["shuffle"] = False

    elif hparams["sorting"] == "random":
        pass

    else:
        raise NotImplementedError(
            "sorting must be random, ascending or descending"
        )

    valid_data = sb.dataio.dataset.DynamicItemDataset.from_json(
        json_path=hparams["valid_annotation"],
        replacements={"data_root": data_folder},
    )
    valid_data = valid_data.filtered_sorted(sort_key="duration")

    test_data = sb.dataio.dataset.DynamicItemDataset.from_json(
        json_path=hparams["test_annotation"],
        replacements={"data_root": data_folder},
    )
    test_data = test_data.filtered_sorted(sort_key="duration")

    datasets = [train_data, valid_data, test_data]

    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        sig = sb.dataio.dataio.read_audio(wav)
        return sig

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)

    @sb.utils.data_pipeline.takes("phn")
    @sb.utils.data_pipeline.provides(
        "phn_encoded",
        "phn_encoded_eos",
        "phn_encoded_bos",
    )
    def text_pipeline(phn):
        phn_list = phn.strip().split()
        phn_encoded_list = label_encoder.encode_sequence(phn_list)
        phn_encoded = torch.LongTensor(phn_encoded_list)
        yield phn_encoded
        phn_encoded_eos = torch.LongTensor(
            label_encoder.append_eos_index(phn_encoded_list)
        )
        yield phn_encoded_eos
        phn_encoded_bos = torch.LongTensor(
            label_encoder.prepend_bos_index(phn_encoded_list)
        )
        yield phn_encoded_bos

    sb.dataio.dataset.add_dynamic_item(datasets, text_pipeline)

    @sb.utils.data_pipeline.takes("phn_canonical")
    @sb.utils.data_pipeline.provides(
        "phn_canonical_list",
        "phn_canonical_encoded",
        "phn_canonical_encoded_eos",
        "phn_canonical_encoded_bos",
    )
    def text_canonical_pipeline(phn_canonical):
        phn_list = phn_canonical.strip().split()
        yield phn_list
        phn_encoded_list = label_encoder.encode_sequence(phn_list)
        phn_encoded = torch.LongTensor(phn_encoded_list)
        yield phn_encoded
        phn_encoded_eos = torch.LongTensor(
            label_encoder.append_eos_index(phn_encoded_list)
        )
        yield phn_encoded_eos
        phn_encoded_bos = torch.LongTensor(
            label_encoder.prepend_bos_index(phn_encoded_list)
        )
        yield phn_encoded_bos

    sb.dataio.dataset.add_dynamic_item(datasets, text_canonical_pipeline)

    # 3. Fit encoder (if needed) and save:
    lab_enc_file = os.path.join(hparams["save_folder"], "label_encoder.txt")
    if label_encoder is None:
        special_labels = {
            "bos_label": hparams["bos_index"],
            "eos_label": hparams["eos_index"],
            "blank_label": hparams["blank_index"],
        }
        label_encoder = sb.dataio.encoder.CTCTextEncoder()
        label_encoder.load_or_create(
            path=lab_enc_file,
            from_didatasets=[train_data],
            output_key="phn_list",
            special_labels=special_labels,
            sequence_input=True,
        )
    else:
        label_encoder.load_or_create(
            path=lab_enc_file,
            output_key="phn_list",
        )

    # 4. Define phone scores pipeline:
    @sb.utils.data_pipeline.takes(
        "scores",
    )
    @sb.utils.data_pipeline.provides(
        "scores_list",
    )
    def scores_pipeline(
            scores
    ):
        # The returned sequence has the same length as phn_canonical_encoded (i.e.: one-less element than
        # phn_canonical_encoded_bos and phn_canonical_encoded_eos.
        yield create_score_tensor_from_string(scores, MetadataConstants.MIN_PHONE_ACCURACY_SCORE.value,
                                              MetadataConstants.MAX_PHONE_ACCURACY_SCORE.value)
    sb.dataio.dataset.add_dynamic_item(datasets, scores_pipeline)

    # 6. Define alignments pipeline:
    if "network_type" in hparams and hparams["network_type"] == "lstm":
        @sb.utils.data_pipeline.takes("phn_ali", "phn_ali_start", "phn_ali_duration")
        @sb.utils.data_pipeline.provides("phn_ali_list", "phn_ali_encoded", "phn_ali_start_list",
                                         "phn_ali_duration_list")
        def alignments_pipeline(phn_ali, phn_ali_start, phn_ali_duration):
            phn_ali_list = phn_ali.strip().split()
            yield phn_ali_list
            phn_ali_encoded_list = label_encoder.encode_sequence(phn_ali_list)
            phn_ali_encoded = torch.LongTensor(phn_ali_encoded_list)
            yield phn_ali_encoded
            phn_ali_start_list = [float(i) for i in phn_ali_start.strip().split()]
            yield phn_ali_start_list
            phn_ali_duration_list = [float(i) for i in phn_ali_duration.strip().split()]
            yield phn_ali_duration_list

        sb.dataio.dataset.add_dynamic_item(datasets, alignments_pipeline)

    output_keys = ["id",
                   "sig",
                   "phn_canonical_list",
                   "phn_canonical_encoded",
                   "phn_canonical_encoded_bos",
                   "phn_canonical_encoded_eos",
                   "scores_list",
                   ]

    if "model_task" in hparams and hparams["model_task"] == "apr":
        output_keys.extend(["phn_encoded", "phn_encoded_eos", "phn_encoded_bos"])

    if "network_type" in hparams and hparams["network_type"] == "lstm":
        output_keys.extend(["phn_ali_list", "phn_ali_encoded", "phn_ali_start_list", "phn_ali_duration_list"])

    # 7. Set output:
    sb.dataio.dataset.set_output_keys(datasets, output_keys)

    return train_data, valid_data, test_data, label_encoder
