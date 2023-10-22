import torch

from speechbrain.utils.data_utils import undo_padding
from speechbrain.utils.metric_stats import MetricStats
from metric_utils.distance_scoring_stats_utils import scoring_distance_details_for_batch,\
    print_scoring_distance_summary, print_scoring_distances, scoring_distance_summary


class DistanceScoringStats(MetricStats):
    """A class for tracking scoring distance statistics.
    """

    def __init__(self):
        self.ids = []
        self.scores = []
        self.summary = {}

    def append(self, ids, predict, target, phones, utt_len, ind2lab):
        """Add stats to the relevant containers.

        * See MetricStats.append()

        Arguments
        ---------
        ids : list
            List of ids corresponding to utterances.
        predict : torch.tensor
            A predicted output, for comparison with the target output
        target : torch.tensor
            The correct reference output, for comparison with the prediction.
        phones: torch.tensor
            The correct phoneme labels as indices.
        utt_len : torch.tensor
            The relative lengths of the predictions, targets and phones,
            used to undo padding if there is padding present.
        ind2lab : callable
            Callable that maps from indices to labels, operating on batches,
            for writing alignments.
        """
        self.ids.extend(ids)
        if utt_len is not None:
            distance = torch.absolute(predict - target)
            distance = undo_padding(distance.squeeze(), utt_len)
            predict = undo_padding(predict.squeeze(), utt_len)
            target = undo_padding(target.squeeze(), utt_len)
            phones = undo_padding(phones, utt_len)

        if ind2lab is not None:
            phones = ind2lab(phones)

        scores = scoring_distance_details_for_batch(ids, predict, target, distance, phones)
        self.scores.extend(scores)

    def write_stats(self, filestream):
        """Write all relevant info (e.g., scoring distances) to file.
        * See MetricStats.write_stats()
        """
        if not self.summary:
            self.summarize()

        print_scoring_distance_summary(self.summary, filestream)
        print_scoring_distances(self.scores, filestream)

    def summarize(self, field=None):
        """Summarize the scoring distances and return relevant statistics.

        * See MetricStats.summarize()
        """
        self.summary = scoring_distance_summary(self.scores)
        if field is not None:
            return self.summary[field]
        return self.summary
