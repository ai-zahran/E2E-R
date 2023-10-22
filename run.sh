#!/bin/bash

STAGE=0
MODEL_TYPE="w2v2"

SSL_TYPE="wavlm"  # The type of the SSL. Choose from ["hubert", "wavlm", "wav2vec2"]
# The HuggingFace name of the SSL. Choose from ["facebook/hubert-large-ls960-ft", "microsoft/wavlm-large",
# "facebook/wav2vec2-large-lv60"]
WAV2VEC2_HUB="microsoft/wavlm-large"

DATA_DIR=/data  # Change this path to the path where you keep your data.
TIMIT_DATA_FOLDER=$DATA_DIR/TIMIT/
SO762_DATA_FOLDER=$DATA_DIR/speechocean762/

# Result logging
RESULTS_FOLDER=$DATA_DIR/results/
EXP_METADATA_FILE=${RESULTS_FOLDER}/exp_metadata.csv

TIMIT_APR_RESULTS_FILE=${RESULTS_FOLDER}/results_apr_timit.csv
SO762_APR_RESULTS_FILE=${RESULTS_FOLDER}/results_apr_so762.csv
SO762_SCORING_RESULTS_FILE=${RESULTS_FOLDER}/results_scoring_so762.csv

EPOCH_RESULTS_DIR=${RESULTS_FOLDER}/epoch_results
PARAMS_DIR=${RESULTS_FOLDER}/params
EXP_DESCRIPTION=""
TRAINING_TYPE="apr"

SCORING_TYPE=""

# Prepare datasets

if [ ! -d $DATA_DIR/speechocean762 ]; then
    echo "Preparing speechocean762 dataset.";
    mkdir $DATA_DIR/speechocean762
    tar -xzmf speechocean762.tar.gz -C $DATA_DIR/speechocean762;
    echo "Finished preparing speechocean762 dataset.";
fi

if [ ! -d $DATA_DIR/TIMIT ]; then
    echo "Preparing TIMIT dataset.";
    unzip -DDq $DATA_DIR/TIMIT.zip -d $DATA_DIR/timit;
    echo "Finished preparing TIMIT dataset.";
fi

#######################################################################################################################

# Experiments


# Experiment #1: Train APR on native speech and train scorer using transfer learning.

EXP_DESCRIPTION="Train APR then scorer"
# Train phoneme recognition model.
if [ $STAGE -le 1 ]; then
    echo "##### Training phoneme recognition model on TIMIT. ######"

    APR_MODEL_DIR="results/apr/${MODEL_TYPE}/crdnn_${MODEL_TYPE}_timit_apr"
    APR_HPARAM_FILE="hparams/apr/${MODEL_TYPE}/train_${MODEL_TYPE}_timit_apr.yaml"

    [ -d $APR_MODEL_DIR ] && rm -r $APR_MODEL_DIR && echo "Removed already existing $APR_MODEL_DIR directory.";

    python3 train.py $APR_HPARAM_FILE \
        --data_folder=$TIMIT_DATA_FOLDER \
        --exp_folder=$APR_MODEL_DIR \
        --exp_metadata_file=$EXP_METADATA_FILE \
        --results_file=$TIMIT_APR_RESULTS_FILE \
        --epoch_results_dir=$EPOCH_RESULTS_DIR \
        --params_dir=$PARAMS_DIR \
        --exp_description="$EXP_DESCRIPTION";
fi

# Train the scorer (Best version).
if [ $STAGE -le 2 ]; then
    echo "##### Training pronunciation scoring on speechocean762 with no adaptation to non-native speech. ######"

    SCORING_MODEL_DIR="results/scoring/${MODEL_TYPE}/crdnn_${MODEL_TYPE}_so762${SCORING_TYPE}_scoring_aug_no_round_no_pre_train"
    PRETRAINED_MODEL_DIR="results/apr/${MODEL_TYPE}/crdnn_${MODEL_TYPE}_timit_apr/1234"
    SCORING_HPARAM_FILE="hparams/scoring/${MODEL_TYPE}/train_${MODEL_TYPE}_so762${SCORING_TYPE}_scoring.yaml"

    [ -d $SCORING_MODEL_DIR ] && rm -r $SCORING_MODEL_DIR && echo "Removed existing $SCORING_MODEL_DIR directory.";

    python3 train.py $SCORING_HPARAM_FILE \
        --data_folder=$SO762_DATA_FOLDER \
        --batch_size=2 \
        --pretrained_model_folder=$PRETRAINED_MODEL_DIR \
        --use_augmentation=True \
        --round_scores=False \
        --exp_folder=$SCORING_MODEL_DIR \
        --exp_metadata_file=$EXP_METADATA_FILE \
        --results_file=$SO762_SCORING_RESULTS_FILE\
        --epoch_results_dir=$EPOCH_RESULTS_DIR \
        --params_dir=$PARAMS_DIR \
        --exp_description="Step 1 then 3 directly (aug, no round)";
fi

#######################################################################################################################

# Experiment #2: Fine-tune APR on non-native correct pronunciation, then train scorer (i.e.: 3-step training).

# Fine-tune phoneme recognition model on in-domain data (correct speech from non-native speakers).
if [ $STAGE -le 3 ]; then
    echo "##### Fine-tuning phoneme recognition model on correct speechocean762 utterances. ######"

    APR_MODEL_DIR="results/apr/${MODEL_TYPE}/crdnn_${MODEL_TYPE}_timit_apr_wavlm/1234"
    FINE_TUNED_APR_MODEL_DIR="results/apr/${MODEL_TYPE}/crdnn_${MODEL_TYPE}_so762_${TRAINING_TYPE}"
    FINE_TUNED_APR_HPARAM_FILE="hparams/apr/${MODEL_TYPE}/train_${MODEL_TYPE}_so762_${TRAINING_TYPE}.yaml"

    [ -d $FINE_TUNED_APR_MODEL_DIR ] && rm -r $FINE_TUNED_APR_MODEL_DIR \
        && echo "Removed existing $FINE_TUNED_APR_MODEL_DIR directory.";

    python3 train.py $FINE_TUNED_APR_HPARAM_FILE \
        --data_folder=$SO762_DATA_FOLDER \
        --pretrained_model_folder=$APR_MODEL_DIR \
        --exp_folder=$FINE_TUNED_APR_MODEL_DIR \
        --exp_metadata_file=$EXP_METADATA_FILE \
        --results_file=$SO762_APR_RESULTS_FILE \
        --epoch_results_dir=$EPOCH_RESULTS_DIR \
        --params_dir=$PARAMS_DIR \
        --exp_description="$EXP_DESCRIPTION";
fi

# Train scorer using fine-tuned model.
if [ $STAGE -le 4 ]; then
    echo "##### Training pronunciation scoring model on speechocean762 with augmentation (i.e.: +aug). ######"

    SCORING_MODEL_DIR="results/scoring/${MODEL_TYPE}/crdnn_${MODEL_TYPE}_so762${SCORING_TYPE}_scoring"
    PRETRAINED_MODEL_DIR="results/apr/${MODEL_TYPE}/crdnn_${MODEL_TYPE}_so762_${TRAINING_TYPE}/1234/"
    SCORING_HPARAM_FILE="hparams/scoring/${MODEL_TYPE}/train_${MODEL_TYPE}_so762${SCORING_TYPE}_scoring.yaml"

    [ -d $SCORING_MODEL_DIR ]  && rm -r $SCORING_MODEL_DIR && echo "Removed existing $SCORING_MODEL_DIR directory.";

    python3 train.py $SCORING_HPARAM_FILE \
        --data_folder=$SO762_DATA_FOLDER \
        --batch_size=2 \
        --pretrained_model_folder=$PRETRAINED_MODEL_DIR \
        --use_augmentation=True \
        --round_scores=False \
        --exp_folder=$SCORING_MODEL_DIR \
        --exp_metadata_file=$EXP_METADATA_FILE \
        --results_file=$SO762_SCORING_RESULTS_FILE \
        --epoch_results_dir=$EPOCH_RESULTS_DIR \
        --params_dir=$PARAMS_DIR \
        --exp_description="Augmentation and no score rounding.";

fi

#######################################################################################################################

# Experiment #3: Training the scorer with decoder sizes from 128 to 1024.

dec_size=128
EXP_DESCRIPTION="Decoder ${dec_size} +aug -score_rnd"
# Train phoneme recognition model.
if [ $STAGE -le 5 ]; then
    echo "##### Training phoneme recognition model on TIMIT with decoder size ${dec_size}. ######"

    APR_MODEL_DIR="results/apr/${MODEL_TYPE}/crdnn_${MODEL_TYPE}_timit_apr_${dec_size}"
    APR_HPARAM_FILE="hparams/apr/${MODEL_TYPE}/train_${MODEL_TYPE}_timit_apr.yaml"

    [ -d $APR_MODEL_DIR ] && rm -r $APR_MODEL_DIR && echo "Removed already existing $APR_MODEL_DIR directory.";

    python3 train.py $APR_HPARAM_FILE \
        --data_folder=$TIMIT_DATA_FOLDER \
        --exp_folder=$APR_MODEL_DIR \
        --dec_neurons=$dec_size \
        --exp_metadata_file=$EXP_METADATA_FILE \
        --results_file=$TIMIT_APR_RESULTS_FILE \
        --epoch_results_dir=$EPOCH_RESULTS_DIR \
        --params_dir=$PARAMS_DIR \
        --exp_description="$EXP_DESCRIPTION";
fi

# Train scorer
if [ $STAGE -le 6 ]; then
    echo "##### Training pronunciation scoring model on speechocean762 with aug and decoder size 128 and no native fine-tuning. ######"

    SCORING_MODEL_DIR="results/scoring/${MODEL_TYPE}/crdnn_${MODEL_TYPE}_so762${SCORING_TYPE}_scoring_${dec_size}"
    PRETRAINED_MODEL_DIR="results/apr/${MODEL_TYPE}/crdnn_${MODEL_TYPE}_timit_apr_${dec_size}/1234"
    SCORING_HPARAM_FILE="hparams/scoring/${MODEL_TYPE}/train_${MODEL_TYPE}_so762${SCORING_TYPE}_scoring.yaml"

    [ -d $SCORING_MODEL_DIR ]  && rm -r $SCORING_MODEL_DIR && echo "Removed existing $SCORING_MODEL_DIR directory.";

    python3 train.py $SCORING_HPARAM_FILE \
        --data_folder=$SO762_DATA_FOLDER \
        --exp_folder=$SCORING_MODEL_DIR \
        --batch_size=2 \
        --pretrained_model_folder=$PRETRAINED_MODEL_DIR \
        --use_augmentation=True \
        --round_scores=False \
        --dec_neurons=$dec_size \
        --exp_metadata_file=$EXP_METADATA_FILE \
        --results_file=$SO762_SCORING_RESULTS_FILE \
        --epoch_results_dir=$EPOCH_RESULTS_DIR \
        --params_dir=$PARAMS_DIR \
        --exp_description="${EXP_DESCRIPTION} -fine-tuning";
fi

#######################################################################################################################

# Experiment #4: Training using HuBERT and WavLM.

EXP_DESCRIPTION="Train using ${SSL_TYPE}"

# Train phoneme recognition model.
if [ $STAGE -le 7 ]; then
    echo "##### Training phoneme recognition model on TIMIT using ${SSL_TYPE}. ######"

    APR_MODEL_DIR="results/apr/${MODEL_TYPE}/crdnn_${MODEL_TYPE}_timit_apr_${SSL_TYPE}"
    APR_HPARAM_FILE="hparams/apr/${MODEL_TYPE}/train_${MODEL_TYPE}_timit_apr.yaml"

    [ -d $APR_MODEL_DIR ] && rm -r $APR_MODEL_DIR && echo "Removed already existing $APR_MODEL_DIR directory.";

    python3 train.py $APR_HPARAM_FILE \
        --data_folder=$TIMIT_DATA_FOLDER \
        --exp_folder=$APR_MODEL_DIR \
        --wav2vec2_hub=$WAV2VEC2_HUB \
        --exp_metadata_file=$EXP_METADATA_FILE \
        --results_file=$TIMIT_APR_RESULTS_FILE \
        --epoch_results_dir=$EPOCH_RESULTS_DIR \
        --params_dir=$PARAMS_DIR \
        --exp_description="$EXP_DESCRIPTION";
fi

# Train the scorer (Best version).
if [ $STAGE -le 8 ]; then
    echo "##### Training pronunciation scoring on speechocean762 with no adaptation to non-native speech. ######"

    SCORING_MODEL_DIR="results/scoring/${MODEL_TYPE}/crdnn_${MODEL_TYPE}_so762${SCORING_TYPE}_scoring_${SSL_TYPE}"
    PRETRAINED_MODEL_DIR="results/apr/${MODEL_TYPE}/crdnn_${MODEL_TYPE}_timit_apr_${SSL_TYPE}/1234"
    SCORING_HPARAM_FILE="hparams/scoring/${MODEL_TYPE}/train_${MODEL_TYPE}_so762${SCORING_TYPE}_scoring.yaml"

    [ -d $SCORING_MODEL_DIR ] && rm -r $SCORING_MODEL_DIR && echo "Removed existing $SCORING_MODEL_DIR directory.";

    python3 train.py $SCORING_HPARAM_FILE \
        --data_folder=$SO762_DATA_FOLDER \
        --batch_size=2 \
        --pretrained_model_folder=$PRETRAINED_MODEL_DIR \
        --wav2vec2_hub=$WAV2VEC2_HUB \
        --use_augmentation=True \
        --round_scores=False \
        --exp_folder=$SCORING_MODEL_DIR \
        --exp_metadata_file=$EXP_METADATA_FILE \
        --results_file=$SO762_SCORING_RESULTS_FILE\
        --epoch_results_dir=$EPOCH_RESULTS_DIR \
        --params_dir=$PARAMS_DIR \
        --exp_description="$EXP_DESCRIPTION";
fi

#######################################################################################################################

# Experiment #5: Training the scorer with decoder MAE loss.

if [ $STAGE -le 9 ]; then
    echo "##### Training pronunciation scoring model on speechocean762 with aug and cosine similarity and no native fine-tuning. ######"

    SCORING_MODEL_DIR="results/scoring/${MODEL_TYPE}/crdnn_${MODEL_TYPE}_so762${SCORING_TYPE}_scoring_mae"
    PRETRAINED_MODEL_DIR="results/apr/${MODEL_TYPE}/crdnn_${MODEL_TYPE}_timit_apr_wavlm/1234"
    SCORING_HPARAM_FILE="hparams/scoring/${MODEL_TYPE}/train_${MODEL_TYPE}_so762${SCORING_TYPE}_scoring.yaml"

    [ -d $SCORING_MODEL_DIR ]  && rm -r $SCORING_MODEL_DIR && echo "Removed existing $SCORING_MODEL_DIR directory.";

    python3 train.py $SCORING_HPARAM_FILE \
        --data_folder=$SO762_DATA_FOLDER \
        --exp_folder=$SCORING_MODEL_DIR \
        --batch_size=2 \
        --score_cost=!name:speechbrain.nnet.losses.l1_loss \
        --pretrained_model_folder=$PRETRAINED_MODEL_DIR \
        --use_augmentation=True \
        --round_scores=False \
        --exp_metadata_file=$EXP_METADATA_FILE \
        --results_file=$SO762_SCORING_RESULTS_FILE \
        --epoch_results_dir=$EPOCH_RESULTS_DIR \
        --params_dir=$PARAMS_DIR \
        --exp_description="MAE Loss";
fi

#######################################################################################################################

# Experiment #6: Training the scorer with NED similarity scoring.

if [ $STAGE -le 10 ]; then
    echo "##### Training pronunciation scoring model on speechocean762 with normalized Euclidean similarity (NES) (+aug, -rnd). ######"

    SCORING_MODEL_DIR="results/scoring/${MODEL_TYPE}/crdnn_${MODEL_TYPE}_so762${SCORING_TYPE}_scoring_nes"
    PRETRAINED_MODEL_DIR="results/apr/${MODEL_TYPE}/crdnn_${MODEL_TYPE}_timit_apr_wavlm/1234"
    SCORING_HPARAM_FILE="hparams/scoring/${MODEL_TYPE}/train_${MODEL_TYPE}_so762${SCORING_TYPE}_scoring.yaml"

    [ -d $SCORING_MODEL_DIR ]  && rm -r $SCORING_MODEL_DIR && echo "Removed existing $SCORING_MODEL_DIR directory.";

    python3 train.py $SCORING_HPARAM_FILE \
        --data_folder=$SO762_DATA_FOLDER \
        --exp_folder=$SCORING_MODEL_DIR \
        --batch_size=2 \
        --similarity_calc="euclidean" \
        --pretrained_model_folder=$PRETRAINED_MODEL_DIR \
        --use_augmentation=True \
        --round_scores=False \
        --exp_metadata_file=$EXP_METADATA_FILE \
        --results_file=$SO762_SCORING_RESULTS_FILE \
        --epoch_results_dir=$EPOCH_RESULTS_DIR \
        --params_dir=$PARAMS_DIR \
        --exp_description="NES scoring";
fi

#######################################################################################################################

# Experiment #7: Train simple LSTM-based model

if [ $STAGE -le 11 ]; then
    echo "##### Training simple LSTM-based model. ######"

    # Produce and process alignments for the speechocean762 dataset.
    kaldi_eg_dir=kaldi/egs/gop_speechocean762/s5/;
    output_alignments_file_path=$SO762_DATA_FOLDER/resource/alignments.txt;
    kaldi_phones_file_path=${kaldi_eg_dir}/data/lang_nosp/phones.txt;
    output_phones_file_path=$SO762_DATA_FOLDER/resource/phones.txt;

    touch $output_alignments_file_path;
    bash align_speechocean762.sh;

    for subset in train test; do
        cat ${kaldi_eg_dir}/exp/ali_${subset}/merged_alignments.txt >> $output_alignments_file_path;
    done

    cp $kaldi_phones_file_path $output_phones_file_path
fi

if [ $STAGE -le 12 ]; then
    # Run LSTM experiment
    SCORING_MODEL_DIR="results/scoring/${MODEL_TYPE}/${MODEL_TYPE}_so762_lstm_scorer"
    PRETRAINED_MODEL_DIR="results/apr/${MODEL_TYPE}/crdnn_${MODEL_TYPE}_timit_apr_wavlm/1234"
    SCORING_HPARAM_FILE="hparams/scoring/${MODEL_TYPE}/train_${MODEL_TYPE}_so762_lstm_scoring.yaml"

    [ -d $SCORING_MODEL_DIR ] && rm -r $SCORING_MODEL_DIR && echo "Removed existing $SCORING_MODEL_DIR directory.";

    python3 train.py $SCORING_HPARAM_FILE \
        --data_folder=$SO762_DATA_FOLDER \
        --batch_size=2 \
        --use_augmentation=True \
        --training_type="training" \
        --pretrained_model_folder=$PRETRAINED_MODEL_DIR \
        --round_scores=False \
        --exp_folder=$SCORING_MODEL_DIR \
        --exp_metadata_file=$EXP_METADATA_FILE \
        --results_file=$SO762_SCORING_RESULTS_FILE \
        --epoch_results_dir=$EPOCH_RESULTS_DIR \
        --params_dir=$PARAMS_DIR \
        --exp_description="LSTM-based";
fi

#######################################################################################################################

# Experiment #8: Training the scorer with similarity scorer sizes from 128 to 2084.

# Train scorer
scorer_dnn_neurons=2048
if [ $STAGE -le 13 ]; then
    echo "##### Training pronunciation scoring model on speechocean762 with aug and decoder size 128 and no native fine-tuning. ######"

    SCORING_MODEL_DIR="results/scoring/${MODEL_TYPE}/crdnn_${MODEL_TYPE}_so762${SCORING_TYPE}_scoring_sim_dnn_${scorer_dnn_neurons}"
    PRETRAINED_MODEL_DIR="results/apr/${MODEL_TYPE}/crdnn_${MODEL_TYPE}_timit_apr/1234"
    SCORING_HPARAM_FILE="hparams/scoring/${MODEL_TYPE}/train_${MODEL_TYPE}_so762${SCORING_TYPE}_scoring.yaml"

    [ -d $SCORING_MODEL_DIR ]  && rm -r $SCORING_MODEL_DIR && echo "Removed existing $SCORING_MODEL_DIR directory.";

    python3 train.py $SCORING_HPARAM_FILE \
        --data_folder=$SO762_DATA_FOLDER \
        --exp_folder=$SCORING_MODEL_DIR \
        --batch_size=2 \
        --pretrained_model_folder=$PRETRAINED_MODEL_DIR \
        --use_augmentation=True \
        --round_scores=False \
        --scorer_dnn_neurons=$scorer_dnn_neurons \
        --exp_metadata_file=$EXP_METADATA_FILE \
        --results_file=$SO762_SCORING_RESULTS_FILE \
        --epoch_results_dir=$EPOCH_RESULTS_DIR \
        --params_dir=$PARAMS_DIR \
        --exp_description="Training scorer with ${scorer_dnn_neurons} neurons";
fi

#######################################################################################################################
