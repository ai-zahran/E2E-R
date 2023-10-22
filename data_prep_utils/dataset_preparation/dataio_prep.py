from data_prep_utils.dataset_preparation.speechocean762_prepare import dataio_prep_speechocean762
from data_prep_utils.dataset_preparation.timit_prepare import dataio_prep_timit


dataio_prep_funcs = {"timit": dataio_prep_timit,
                     "speechocean762": dataio_prep_speechocean762
                     }


def dataio_prep(hparams, label_encoder=None):
    dataio_prep_func = dataio_prep_funcs[hparams["dataset_name"]]
    if label_encoder is None:
        return dataio_prep_func(hparams)
    else:
        return dataio_prep_func(hparams, label_encoder)
