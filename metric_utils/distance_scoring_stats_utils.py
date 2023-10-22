import sys

from collections import defaultdict
from speechbrain.utils.edit_distance import _batch_to_dict_format


def scoring_distance_details_for_batch(ids, predict, target, distances, phones):
    """Convenient batch interface for ``scoring_distance_details_by_utterance``.

    Arguments
    ---------
    ids : list, torch.tensor
        Utterance ids for the batch.
    predict: list, torch.tensor
        Predicted scores.
    target: list, torch.tensor
        Actual scores.
    distances : list, torch.tensor
        Scoring distances.
    phones : list, torch.tensor
        Phoneme labels.

    Returns
    -------
    list
        See ``scoring_distance_details_by_utterance``
    """
    distances = _batch_to_dict_format(ids, distances)
    phones = _batch_to_dict_format(ids, phones)
    predict = _batch_to_dict_format(ids, predict)
    target = _batch_to_dict_format(ids, target)

    return scoring_distance_details_by_utterance(predict, target, distances, phones)


def print_scoring_distance_summary(scoring_distance_details, file=sys.stdout):
    """Prints out scoring distance summary details in human-readable format.

    Arguments
    ---------
    scoring_distance_details : dict
        Dict of scoring distance summary details,
    file : stream
        Where to write. (default: sys.stdout)
    """
    print("%MSE {MSE:.2f}".format(**scoring_distance_details), file=file)
    print("%SER {SER:.2f} [ {num_erroneous_sents} / {num_scored_sents} ]".format(**scoring_distance_details), file=file)
    print("=" * 80, file=file)
    print("Average phoneme scoring distances:", file=file)
    print('\n'.join([f"{token}: {scoring_dist}" for token, scoring_dist
                     in scoring_distance_details["token_avg_scoring_dist"].items()]), file=file)


def _print_scoring_distance_header(scoring_distance_details, file=sys.stdout):
    print("=" * 80, file=file)
    print(
        "{key}, %MSE {MSE:.2f} [ {num_erroneous_predictions} / {num_tokens} ]".format(  # noqa
            **scoring_distance_details
        ),
        file=file,
    )


def _print_utt_scoring_distances(predictions, targets, distances, phones, separator=" ; ", file=sys.stdout):
    # First, get equal length text for all:
    predictions_padded = []
    targets_padded = []
    distances_padded = []
    phones_padded = []
    for prediction, target, distance, phone in zip(predictions, targets, distances, phones):  # i indexes a, j indexes b
        prediction_string = str(prediction)
        target_string = str(target)
        distance_string = str(distance)
        phone_string = str(phone)
        # NOTE: the padding does not actually compute printed length,
        # but hopefully we can assume that printed length is
        # at most the str len
        pad_length = max(len(prediction_string), len(target_string), len(distance_string), len(phone_string))
        predictions_padded.append(prediction_string.center(pad_length))
        targets_padded.append(target_string.center(pad_length))
        distances_padded.append(distance_string.center(pad_length))
        phones_padded.append(phone_string.center(pad_length))
    # Then print, in the order phones, distances
    print(separator.join(phones_padded), file=file)
    print(separator.join(predictions_padded), file=file)
    print(separator.join(targets_padded), file=file)
    print(separator.join(distances_padded), file=file)


def print_scoring_distances(details_by_utterance,
                            file=sys.stdout,
                            separator=" ; ",
                            sample_separator=None):
    """Print scoring distances summary.

    Arguments
    ---------
    details_by_utterance : list
        List of score distance details by utterance
    file : stream
        Where to write. (default: sys.stdout)
    separator : str
        String that separates each token in the output. Note the spaces in the
        default.
    sample_separator: str
        A separator to put between samples (optional)
    """
    for dets in details_by_utterance:
        if dets["scored"]:
            _print_scoring_distance_header(dets, file=file)
            _print_utt_scoring_distances(
                dets["predict"],
                dets["target"],
                dets["distances"],
                dets["phones"],
                file=file,
                separator=separator,
            )
            if sample_separator:
                print(sample_separator, file=file)
    pass


def scoring_distance_details_by_utterance(predict, target, distances, phones_dict):
    """Computes scoring distance info about each single utterance.

    This info can then be used to compute summary details (MSE, SER).

    Arguments
    ---------
    predict : dict
        Should be indexable by utterance ids, and return the precited scores
        for each utterance id as iterable
    target : dict
        Should be indexable by utterance ids, and return the target scores
        for each utterance id as iterable
    distances : dict
        Should be indexable by utterance ids, and return the scoring distances
        for each utterance id as iterable
    phones_dict : dict
        Should be indexable by utterance ids, and return
        the phoneme labels for each utterance id as iterable

    Returns
    -------
    list
        A list with one entry for every reference utterance. Each entry is a
        dict with keys:

        * "key": utterance id
        * "scored": (bool) Whether utterance was scored.
        * "num_tokens": (int) Number of tokens in the reference.
        * "predict": (iterable) The predicted scores.
        * "target": (iterable) The target scores.
        * "distances": (iterable) The scoring distances.
        * "phones": (iterable) the phoneme labels.
    """
    details_by_utterance = []
    for key, utt_distance in distances.items():
        utt_predict = predict[key]
        utt_target = target[key]
        mse = sum([d ** 2 for d in utt_distance]) / len(utt_distance)
        num_erroneous_preds = len([d for d in utt_distance if round(d) > 0])
        # Initialize utterance_details
        utterance_details = {
            "key": key,
            "scored": True,
            "num_tokens": len(utt_distance),
            "predict": utt_predict,
            "target": utt_target,
            "distances": utt_distance,
            "phones": phones_dict[key],
            "num_erroneous_predictions": num_erroneous_preds,
            "MSE": mse
        }
        details_by_utterance.append(utterance_details)
    return details_by_utterance


def scoring_distance_summary(details_by_utterance):
    """
    Computes summary stats from the output of scoring_distance_details_by_utterance

    Summary stats like MSE

    Arguments
    ---------
    details_by_utterance : list
        See the output of scoring_distance_details_by_utterance

    Returns
    -------
    dict
        Dictionary with keys:

        * "MSE": (float) Mean squared error.
        * "SER": (float) Sentence Error Rate (percentage of utterances
          which had at least one error).
        * "num_scored_tokens": (int) Total number of tokens in scored
          tokens
        * "num_erraneous_sents": (int) Total number of utterances
          which had at least one error.
        * "num_scored_sents": (int) Total number of utterances
          which were scored.
    """
    # Build the summary details:
    num_scored_tokens = num_scored_sents = num_erraneous_sents = squared_distances = 0
    token_total_scoring_dist = defaultdict(float)
    token_count = defaultdict(int)
    for dets in details_by_utterance:
        if dets["scored"]:
            num_scored_sents += 1
            num_scored_tokens += dets["num_tokens"]
            squared_distances += sum([d ** 2 for d in dets["distances"]])
            if round(sum(dets["distances"]) / dets["num_tokens"]) > 0:
                num_erraneous_sents += 1
            for token, dist in zip(dets["phones"], dets["distances"]):
                token_total_scoring_dist[token] += dist
                token_count[token] += 1
    token_avg_scoring_dist = {token: token_total_scoring_dist[token] / token_count[token] for token in token_count}
    mse_details = {
        "MSE": 100.0 * squared_distances / num_scored_tokens,
        "SER": 100.0 * num_erraneous_sents / num_scored_sents,
        "num_scored_tokens": num_scored_tokens,
        "num_erroneous_sents": num_erraneous_sents,
        "num_scored_sents": num_scored_sents,
        "token_avg_scoring_dist": token_avg_scoring_dist
    }
    return mse_details
