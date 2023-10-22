from models.apr.feat_based_apr import APRFeatBased
from models.apr.wav2vec2_apr import APRWav2vec2
from models.scoring.feat_based_lstm_scoring import ScorerFeatBasedLSTM
from models.scoring.feat_based_scoring import ScorerFeatBased
from models.scoring.wav2vec2_lstm_scoring import ScorerWav2vec2LSTM
from models.scoring.wav2vec2_scoring import ScorerWav2vec2

brain_classes = {
    "apr": {
        "feat_based": {
            "pure": APRFeatBased,
        },
        "wav2vec2.0": {
            "pure": APRWav2vec2,
        },
        "hubert": {
            "pure": APRWav2vec2,
        },
    },
    "scoring": {
        "feat_based": {
            "pure": ScorerFeatBased,
            "lstm": ScorerFeatBasedLSTM,
        },
        "wav2vec2.0": {
            "pure": ScorerWav2vec2,
            "lstm": ScorerWav2vec2LSTM,
        },
        "hubert": {
            "pure": ScorerWav2vec2,
        }
    }
}


def get_brain_class(hparams):
    model_task = hparams["model_task"]
    model_type = hparams["model_type"]
    if model_task == "apr":
        if "multi_task" in hparams and hparams["multi_task"]:
            return brain_classes[model_task][model_type]["multi_task"]
        else:
            return brain_classes[model_task][model_type]["pure"]

    else:
        if "network_type" in hparams and hparams["network_type"] == "lstm":
            return brain_classes[model_task][model_type]["lstm"]
        else:
            return brain_classes[model_task][model_type]["pure"]
