"""
Adapted from code by Mirco Ravanelli and Elena Rastorgueva (2020).
"""
import os
import json
import logging
import speechbrain as sb
import torch
from speechbrain.utils.data_utils import get_all_files
from speechbrain.dataio.dataio import read_audio

logger = logging.getLogger(__name__)
SAMPLERATE = 16000
PHONE_LIST = ['V', 'AE', 'Y', 'EH', 'DH', 'AW', 'P', 'JH', 'M', 'UW', 'B', 'R', 'TH', 'G', 'CH', 'UH', 'EY', 'D', 'L',
              'AO', 'Z', 'W', 'IH', 'ER', 'AA', 'AY', 'HH', 'S', 'F', 'N', 'ZH', 'K', 'OW', 'NG', 'OY', 'AH', 'SH', 'T',
              'IY']


def prepare_timit(
        data_folder,
        save_json_train,
        save_json_valid,
        save_json_test,
        phn_set=40,
        uppercase=False,
        skip_prep=False,
):
    """
    repares the json files for the TIMIT dataset.

    Arguments
    ---------
    data_folder : str
        Path to the folder where the original TIMIT dataset is stored.
    save_json_train : str
        The path where to store the training json file.
    save_json_valid : str
        The path where to store the valid json file.
    save_json_test : str
        The path where to store the test json file.
    phn_set : {60, 48, 39}, optional,
        Default: 39
        The phoneme set to use in the phn label.
        It could be composed of 60, 48, or 39 phonemes.
    uppercase : bool, optional
        Default: False
        This option must be True when the TIMIT dataset
        is in the upper-case version.
    skip_prep: bool
        Default: False
        If True, the data preparation is skipped.
    """

    # Skip if needed
    if skip_prep:
        return

    # Getting speaker dictionary
    dev_spk, test_spk = _get_speaker()
    avoid_sentences = ["sa1", "sa2"]
    extension = [".wav"]

    # Checking TIMIT_uppercase
    if uppercase:
        avoid_sentences = [item.upper() for item in avoid_sentences]
        extension = [item.upper() for item in extension]
        dev_spk = [item.upper() for item in dev_spk]
        test_spk = [item.upper() for item in test_spk]

    # Check if this phase is already done (if so, skip it)
    if skip([save_json_train, save_json_valid, save_json_test]):
        logger.info("Skipping preparation, completed in previous run.")
        return

    # Additional checks to make sure the data folder contains TIMIT
    _check_timit_folders(uppercase, data_folder)

    msg = "Creating json files for the TIMIT Dataset.."
    logger.info(msg)

    # Creating json files
    # NOTE: TIMIT has the DEV files in the test directory.
    splits = ["train", "test", "test"]
    annotations = [save_json_train, save_json_valid, save_json_test]
    match_or = [None, dev_spk, test_spk]

    for split, save_file, match in zip(splits, annotations, match_or):
        if uppercase:
            match_lst = extension + [split.upper()]
        else:
            match_lst = extension + [split]

        # List of the wav files
        wav_lst = get_all_files(
            data_folder,
            match_and=match_lst,
            match_or=match,
            exclude_or=avoid_sentences,
        )
        if split == "dev":
            print(wav_lst)

        # Json creation
        create_json(wav_lst, save_file, uppercase, phn_set)


def _get_phonemes():
    # This dictionary is used to conver the 60 phoneme set
    # into the 48 one
    # This dictionary is used to convert the 60 phoneme set to 40 one
    from_60_to_40_phn = {}
    from_60_to_40_phn["sil"] = "sil"
    from_60_to_40_phn["aa"] = "aa"
    from_60_to_40_phn["ae"] = "ae"
    from_60_to_40_phn["ah"] = "ah"
    from_60_to_40_phn["ao"] = "ao"
    from_60_to_40_phn["aw"] = "aw"
    from_60_to_40_phn["ax"] = "ah"
    from_60_to_40_phn["ax-h"] = "ah"
    from_60_to_40_phn["axr"] = "er"
    from_60_to_40_phn["ay"] = "ay"
    from_60_to_40_phn["b"] = "b"
    from_60_to_40_phn["bcl"] = "sil"
    from_60_to_40_phn["ch"] = "ch"
    from_60_to_40_phn["d"] = "d"
    from_60_to_40_phn["dcl"] = "sil"
    from_60_to_40_phn["dh"] = "dh"
    from_60_to_40_phn["dx"] = "t"
    from_60_to_40_phn["eh"] = "eh"
    from_60_to_40_phn["el"] = "l"
    from_60_to_40_phn["em"] = "m"
    from_60_to_40_phn["en"] = "n"
    from_60_to_40_phn["eng"] = "ng"
    from_60_to_40_phn["epi"] = "sil"
    from_60_to_40_phn["er"] = "er"
    from_60_to_40_phn["ey"] = "ey"
    from_60_to_40_phn["f"] = "f"
    from_60_to_40_phn["g"] = "g"
    from_60_to_40_phn["gcl"] = "sil"
    from_60_to_40_phn["h#"] = "sil"
    from_60_to_40_phn["hh"] = "hh"
    from_60_to_40_phn["hv"] = "hh"
    from_60_to_40_phn["ih"] = "ih"
    from_60_to_40_phn["ix"] = "ih"
    from_60_to_40_phn["iy"] = "iy"
    from_60_to_40_phn["jh"] = "jh"
    from_60_to_40_phn["k"] = "k"
    from_60_to_40_phn["kcl"] = "sil"
    from_60_to_40_phn["l"] = "l"
    from_60_to_40_phn["m"] = "m"
    from_60_to_40_phn["ng"] = "ng"
    from_60_to_40_phn["n"] = "n"
    from_60_to_40_phn["nx"] = "n"
    from_60_to_40_phn["ow"] = "ow"
    from_60_to_40_phn["oy"] = "oy"
    from_60_to_40_phn["p"] = "p"
    from_60_to_40_phn["pau"] = "sil"
    from_60_to_40_phn["pcl"] = "sil"
    from_60_to_40_phn["q"] = ""
    from_60_to_40_phn["r"] = "r"
    from_60_to_40_phn["s"] = "s"
    from_60_to_40_phn["sh"] = "sh"
    from_60_to_40_phn["t"] = "t"
    from_60_to_40_phn["tcl"] = "sil"
    from_60_to_40_phn["th"] = "th"
    from_60_to_40_phn["uh"] = "uh"
    from_60_to_40_phn["uw"] = "uw"
    from_60_to_40_phn["ux"] = "uw"
    from_60_to_40_phn["v"] = "v"
    from_60_to_40_phn["w"] = "w"
    from_60_to_40_phn["y"] = "y"
    from_60_to_40_phn["z"] = "z"
    from_60_to_40_phn["zh"] = "zh"

    return from_60_to_40_phn


def _get_speaker():
    # List of test speakers
    test_spk = [
        "fdhc0",
        "felc0",
        "fjlm0",
        "fmgd0",
        "fmld0",
        "fnlp0",
        "fpas0",
        "fpkt0",
        "mbpm0",
        "mcmj0",
        "mdab0",
        "mgrt0",
        "mjdh0",
        "mjln0",
        "mjmp0",
        "mklt0",
        "mlll0",
        "mlnt0",
        "mnjm0",
        "mpam0",
        "mtas1",
        "mtls0",
        "mwbt0",
        "mwew0",
    ]

    # List of dev speakers
    dev_spk = [
        "fadg0",
        "faks0",
        "fcal1",
        "fcmh0",
        "fdac1",
        "fdms0",
        "fdrw0",
        "fedw0",
        "fgjd0",
        "fjem0",
        "fjmg0",
        "fjsj0",
        "fkms0",
        "fmah0",
        "fmml0",
        "fnmr0",
        "frew0",
        "fsem0",
        "majc0",
        "mbdg0",
        "mbns0",
        "mbwm0",
        "mcsh0",
        "mdlf0",
        "mdls0",
        "mdvc0",
        "mers0",
        "mgjf0",
        "mglb0",
        "mgwt0",
        "mjar0",
        "mjfc0",
        "mjsw0",
        "mmdb1",
        "mmdm2",
        "mmjr0",
        "mmwh0",
        "mpdf0",
        "mrcs0",
        "mreb0",
        "mrjm4",
        "mrjr0",
        "mroa0",
        "mrtk0",
        "mrws1",
        "mtaa0",
        "mtdt0",
        "mteb0",
        "mthc0",
        "mwjg0",
    ]

    return dev_spk, test_spk


def skip(annotations):
    """
    Detects if the timit data_preparation has been already done.
    If the preparation has been done, we can skip it.

    Returns
    -------
    bool
        if True, the preparation phase can be skipped.
        if False, it must be done.
    """
    skip = True

    for annotation in annotations:
        if not os.path.isfile(annotation):
            skip = False
            break

    return skip


def create_json(
        wav_lst, json_file, uppercase, phn_set
):
    """
    Creates the json file given a list of wav files.

    Arguments
    ---------
    wav_lst : list
        The list of wav files of a given data split.
    json_file : str
            The path of the output json file.
    uppercase : bool
        Whether this is the uppercase version of timit.
    phn_set : {60, 48, 39}, optional,
        Default: 39
        The phoneme set to use in the phn label.
    """

    # Adding some Prints
    msg = "Creating %s..." % (json_file)
    logger.info(msg)
    json_dict = {}

    for wav_file in wav_lst:

        # Getting sentence and speaker ids
        spk_id = wav_file.split("/")[-2]
        snt_id = wav_file.split("/")[-1].replace(".wav", "")
        snt_id = spk_id + "_" + snt_id

        # Reading the signal (to retrieve duration in seconds)
        signal = read_audio(wav_file)
        duration = len(signal) / SAMPLERATE

        # Retrieving words and check for uppercase
        if uppercase:
            wrd_file = wav_file.replace(".WAV", ".WRD")
        else:
            wrd_file = wav_file.replace(".wav", ".wrd")

        if not os.path.exists(os.path.dirname(wrd_file)):
            err_msg = "the wrd file %s does not exists!" % (wrd_file)
            raise FileNotFoundError(err_msg)

        words = [line.rstrip("\n").split(" ")[2] for line in open(wrd_file)]
        words = " ".join(words)

        # Retrieving phonemes
        if uppercase:
            phn_file = wav_file.replace(".WAV", ".PHN")
        else:
            phn_file = wav_file.replace(".wav", ".phn")

        if not os.path.exists(os.path.dirname(phn_file)):
            err_msg = "the wrd file %s does not exists!" % (phn_file)
            raise FileNotFoundError(err_msg)

        # Getting the phoneme and ground truth ends lists from the phn files
        phonemes, ends = get_phoneme_lists(phn_file, phn_set)

        # Convert to e.g. "a sil b", "1 4 5"
        phonemes_str = " ".join(phonemes)
        ends_str = " ".join(ends)

        utt_dict = {
            "wav": wav_file,
            "duration": duration,
            "spk_id": spk_id,
            "phn": phonemes_str,
            "wrd": words,
            "ground_truth_phn_ends": ends_str,
        }
        json_dict[snt_id] = utt_dict

    # Writing the dictionary to the json file
    with open(json_file, mode="w") as json_f:
        json.dump(json_dict, json_f, indent=2)

    logger.info(f"{json_file} successfully created!")


def get_phoneme_lists(phn_file, phn_set):
    """
    Reads the phn file and gets the phoneme list & ground truth ends list.
    """

    phonemes = []
    ends = []

    for line in open(phn_file):
        end, phoneme = line.rstrip("\n").replace("h#", "sil").split(" ")[1:]

        # Getting dictionaries for phoneme conversion
        from_60_to_40_phn = _get_phonemes()

        # Removing end corresponding to q if phn set is not 61
        if phn_set != 60:
            if phoneme == "q":
                end = ""

        # Converting phns if necessary
        if phn_set == 40:
            phoneme = from_60_to_40_phn[phoneme]

        # Appending arrays
        if len(phoneme) > 0:
            phonemes.append(phoneme)
        if len(end) > 0:
            ends.append(end)

    if phn_set != 60:
        # Filtering out consecutive silences by applying a mask with `True` marking
        # which sils to remove
        # e.g.
        # phonemes          [  "a", "sil", "sil",  "sil",   "b"]
        # ends              [   1 ,    2 ,    3 ,     4 ,    5 ]
        # ---
        # create:
        # remove_sil_mask   [False,  True,  True,  False,  False]
        # ---
        # so end result is:
        # phonemes ["a", "sil", "b"]
        # ends     [  1,     4,   5]

        remove_sil_mask = [True if x == "sil" else False for x in phonemes]

        for i, val in enumerate(remove_sil_mask):
            if val is True:
                if i == len(remove_sil_mask) - 1:
                    remove_sil_mask[i] = False
                elif remove_sil_mask[i + 1] is False:
                    remove_sil_mask[i] = False

        phonemes = [
            phon for i, phon in enumerate(phonemes) if not remove_sil_mask[i]
        ]
        ends = [end for i, end in enumerate(ends) if not remove_sil_mask[i]]

    return phonemes, ends


def _check_timit_folders(uppercase, data_folder):
    """
    Check if the data folder actually contains the TIMIT dataset.

    If not, raises an error.

    Returns
    -------
    None

    Raises
    ------
    FileNotFoundError
        If data folder doesn't contain TIMIT dataset.
    """

    # Creating checking string wrt to lower or uppercase
    if uppercase:
        test_str = "/TEST/DR1"
        train_str = "/TRAIN/DR1"
    else:
        test_str = "/test/dr1"
        train_str = "/train/dr1"

    # Checking test/dr1
    if not os.path.exists(data_folder + test_str):
        err_msg = (
                "the folder %s does not exist (it is expected in "
                "the TIMIT dataset)" % (data_folder + test_str)
        )
        raise FileNotFoundError(err_msg)

    # Checking train/dr1
    if not os.path.exists(data_folder + train_str):
        err_msg = (
                "the folder %s does not exist (it is expected in "
                "the TIMIT dataset)" % (data_folder + train_str)
        )
        raise FileNotFoundError(err_msg)


def dataio_prep_timit(hparams, label_encoder=None):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions."""
    if label_encoder is None:
        label_encoder = sb.dataio.encoder.CTCTextEncoder()
        special_labels = {
            "bos_label": hparams["bos_index"],
            "eos_label": hparams["eos_index"],
            "blank_label": hparams["blank_index"],
        }
    else:
        special_labels = {}

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

    # 3. Define text pipeline:
    @sb.utils.data_pipeline.takes("phn")
    @sb.utils.data_pipeline.provides(
        "phn_list",
        "phn_encoded_list",
        "phn_encoded",
        "phn_encoded_eos",
        "phn_encoded_bos",
    )
    def text_pipeline(phn):
        phn_list = phn.strip().split()
        yield phn_list
        phn_encoded_list = label_encoder.encode_sequence(phn_list)
        yield phn_encoded_list
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

    # 4. Fit encoder:
    # Load or compute the label encoder
    lab_enc_file = os.path.join(hparams["save_folder"], "label_encoder.txt")
    label_encoder.load_or_create(
        path=lab_enc_file,
        from_didatasets=[train_data],
        output_key="phn_list",
        special_labels=special_labels,
        sequence_input=True,
    )

    # 5. Set output:
    sb.dataio.dataset.set_output_keys(
        datasets,
        ["id", "sig", "phn_encoded", "phn_encoded_eos", "phn_encoded_bos"],
    )

    return train_data, valid_data, test_data, label_encoder
