#!/bin/bash
#
# A batch script to extract alignments for the speechocean762 dataset.
# This recipe uses code from https://eleanorchodroff.com/tutorial/kaldi/forced-alignment.html

MODEL_URL=https://kaldi-asr.org/models/13/0013_librispeech_v1_chain.tar.gz
IVECTOR_EXTRACTOR_URL=https://kaldi-asr.org/models/13/0013_librispeech_v1_extractor.tar.gz
LANG_URL=https://kaldi-asr.org/models/13/0013_librispeech_v1_lm.tar.gz
MODEL_DIR=exp/nnet3_cleaned/tdnn_sp
IVECTOR_EXTRACTOR_DIR=exp/nnet3_cleaned/extractor
LANG_DIR=data/lang
KALDI_ALIGNMENT_SCRIPT=${PWD}/align_with_kaldi.sh


# Copy the Kaldi directory.
if [ ! -d kaldi ]; then
    echo "##### Copying Kaldi from the Kaldi image. #####";
    cp -r /opt/kaldi/ ./ || { echo "Failed to copy Kaldi directory to ${PWD}/kaldi."; exit 1; };
    echo "##### Kaldi successfully copied from the Kaldi image. #####";
fi

cd ./kaldi/egs/librispeech/s5

# Download Librispeech ASR model files.
echo "##### Preparing Librispeech ASR model files. #####"

if [ ! -d $MODEL_DIR ]; then
    echo "Preparing ASR chain model.";
    wget --no-check-certificate -q $MODEL_URL || { echo "Failed to download ASR chain model to ${PWD}."; exit 1; };
    tar -xvzf 0013_librispeech_v1_chain.tar.gz \
        && mkdir exp/nnet3_cleaned/ \
        && mv exp/chain_cleaned/tdnn_1d_sp $MODEL_DIR \
        || { echo "Failed to prepare ASR chain model in $MODEL_DIR"; exit 1; };
fi

if [ ! -d $IVECTOR_EXTRACTOR_DIR ]; then
    echo "Preparing i-vector extractor.";
    wget --no-check-certificate -q $IVECTOR_EXTRACTOR_URL || { echo "Failed to download i-vector extractor to ${PWD}."; \
        exit 1; };
    tar -xvzf 0013_librispeech_v1_extractor.tar.gz \
        || { echo "Failed to prepare i-vector extractor in $IVECTOR_EXTRACTOR_DIR"; exit 1; };
fi

if [ ! -d $LANG_DIR ]; then
    echo "Preparing language model.";
    wget --no-check-certificate -q $LANG_URL || { echo "Failed to download language model to ${PWD}."; exit 1; };
    tar -xvzf 0013_librispeech_v1_lm.tar.gz \
        && mv data/lang_test_tgsmall data/lang \
        || { echo "Failed to prepare language model in $LANG_DIR"; exit 1; };
fi

# Run GOP speechocean762 eg
echo "##### Running Kaldi's GOP speechocean762 script. #####"
cd ../../gop_speechocean762/s5/
bash $KALDI_ALIGNMENT_SCRIPT --data /scratch/users/aiz2/ --stage 2 || { echo "Failed to run Kaldi's GOP speechocean762 script"; \
    exit 1; };

# Collect alignments.
echo "##### Processing alignment files. #####"
for subset in train test; do
    ali_dir=exp/ali_${subset}
    for i in  ${ali_dir}/ali.*.gz;
        do ../../../src/bin/ali-to-phones --ctm-output ../../librispeech/s5/exp/nnet3_cleaned/tdnn_sp/final.mdl \
            ark:"gunzip -c $i|" -> ${i%.gz}.ctm \
            || { echo "Failed to process alignment files in $ali_dir"; exit 1; };
    done;
    cat ${ali_dir}/*.ctm > ${ali_dir}/merged_alignments.txt \
        || { echo "Failed to merge alignemnt files in $ali_dir"; exit 1; };
done;
