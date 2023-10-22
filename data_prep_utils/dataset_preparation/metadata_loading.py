import json
import os
import re

from collections import defaultdict
from typing import Dict, List, Tuple
from data_prep_utils.data.utterance_metadata import UtteranceMetadata


def load_utterance_metadata(dataset_folder) -> Dict[str, UtteranceMetadata]:
    """
    Loads scores for the dataset.
    """
    scores_file_path = os.path.join(dataset_folder, "resource", "scores.json")
    scores_data = dict()
    with open(scores_file_path, 'r') as scores_file:
        scores = json.load(scores_file)
    for utt_id in scores:
        phone_index = 0
        word_phone_offsets = [phone_index]
        for word in scores[utt_id]["words"]:
            phone_index += len(word["phones"])
            word_phone_offsets.append(phone_index)
        utterance_metadata = UtteranceMetadata(
            utterance_id=utt_id,
            phones=[process_phone(phone) for word in scores[utt_id]["words"] for phone in word["phones"]],
            words=[word["text"] for word in scores[utt_id]["words"]],
            word_phone_offsets=word_phone_offsets,
            phone_accuracy_scores=[float(phone_accuracy) for word in scores[utt_id]["words"]
                                   for phone_accuracy in word["phones-accuracy"]],
            word_accuracy_scores=[float(word["accuracy"]) for word in scores[utt_id]["words"]],
            word_stress_scores=[float(word["stress"]) for word in scores[utt_id]["words"]],
            word_total_scores=[float(word["total"]) for word in scores[utt_id]["words"]],
            sentence_accuracy_score=float(scores[utt_id]["accuracy"]),
            sentence_completeness_score=float(scores[utt_id]["completeness"]),
            sentence_fluency_score=float(scores[utt_id]["fluency"]),
            sentence_prosodic_score=float(scores[utt_id]["prosodic"]),
            sentence_total_score=float(scores[utt_id]["total"])
        )
        scores_data[utt_id] = utterance_metadata
    return scores_data


def load_scores_dict(dataset_folder) -> Dict[str, Dict[str, float]]:
    """
    Loads scores for the dataset.
    """
    phone_scores_lists = load_utterance_metadata(dataset_folder)
    phone_scores = defaultdict(Dict)
    for utt_id in phone_scores_lists:
        for phone_index, phone_accuracy in enumerate(phone_scores_lists[utt_id]):
            phone_scores[utt_id][f"{utt_id}_{phone_index}"] = phone_accuracy
    return phone_scores


def load_kaldi_data_file(file_path: str) -> Dict[str, str]:
    """Loads the content of a Kaldi data file."""
    file_content = {}
    with open(file_path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        utt_id, utt_data = line.strip().split(None, 1)
        file_content[utt_id] = utt_data
    return file_content


def load_kaldi_wav_scp(folder_path: str) -> Dict[str, str]:
    """Loads the content of a Kaldi wav.scp file."""
    wav_scp_file_path = os.path.join(folder_path, "wav.scp")
    return load_kaldi_data_file(wav_scp_file_path)


def load_kaldi_text_file(folder_path: str) -> Dict[str, str]:
    """Loads the text transcription of utterances in a Kaldi text file."""
    text_file_path = os.path.join(folder_path, "text")
    return load_kaldi_data_file(text_file_path)


def load_kaldi_utt2spk(folder_path: str) -> Dict[str, str]:
    """Loads Kaldi utt2spk information."""
    utt2spk_path = os.path.join(folder_path, "utt2spk")
    return load_kaldi_data_file(utt2spk_path)


def load_alignments(folder_path: str) -> Dict[str, List[Tuple[str, float, float]]]:
    """Loads alignments from an alignments file."""
    alignments_file_path = os.path.join(folder_path, "resource", "alignments.txt")
    kaldi_phones_file_path = os.path.join(folder_path, "resource", "phones.txt")
    alignments = defaultdict(list)

    if os.path.exists(alignments_file_path) and os.path.join(kaldi_phones_file_path):
        phone_id_to_phoneme = load_kaldi_phones_dict(kaldi_phones_file_path)
        with open(alignments_file_path, 'r') as alignments_file:
            for line in alignments_file:
                utt_id, channel, start_time, duration, phone_id = line.strip().split()
                alignments[utt_id].append((phone_id_to_phoneme[phone_id], float(start_time), float(duration)))
    return alignments


def load_kaldi_phones_dict(kaldi_phones_file_path) -> Dict[int, str]:
    """Loads the phone ID to phoneme dictionary from the kaldi phones file."""
    phone_id_to_phoneme = dict()
    with open(kaldi_phones_file_path, 'r') as phones_file:
        lines = phones_file.readlines()
    for line in lines:
        phoneme, phone_id = line.strip().split()
        phone_id_to_phoneme[phone_id] = process_phone(phoneme)
    return phone_id_to_phoneme


def process_phone(phone: str) -> str:
    """Removes unnecessary information from the phoneme string."""
    processed_phone = phone.split('_')[0].lower()
    processed_phone = re.sub(r"\d$", "", processed_phone)
    return processed_phone
