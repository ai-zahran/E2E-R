#!/usr/bin/env bash

# Copyright      2019  Junbo Zhang
#           2020-2021  Xiaomi Corporation (Author: Junbo Zhang, Yongqing Wang)
# Apache 2.0

# This script shows how to calculate Goodness of Pronunciation (GOP) and
# use the GOP-based feature to do phone-level mispronunciations detection.
# Read ../README.md or the following paper for details:
#
# "Hu et al., Improved mispronunciation detection with deep neural network
# trained acoustic models and transfer learning based logistic regression
# classifiers, 2015."


# Set this to somewhere where you want to put your data, or where someone 
# else has already put it.  You'll want to change this if you're not on
# the Xiaomi's grid.
data=/home/storage07/zhangjunbo/data

# Base url for downloads.
data_url=www.openslr.org/resources/101
stage=1
nj=25

# You might not want to do this for interactive shells.
set -e

. ./cmd.sh
. ./path.sh
. parse_options.sh

# This recipe depends on the model trained in the librispeech recipe.
# For training it:
#   cd $KALDI_ROOT/egs/librispeech/s5
#   ./run.sh && local/nnet3/run_tdnn.sh

# Check librispeech's models
librispeech_eg=../../librispeech/s5
model=$librispeech_eg/exp/nnet3_cleaned/tdnn_sp
ivector_extractor=$librispeech_eg/exp/nnet3_cleaned/extractor
lang=$librispeech_eg/data/lang

for d in $model $ivector_extractor $lang; do
  [ ! -d $d ] && echo "$0: no such path $d" && exit 1;
done

if [ $stage -le 1 ]; then
  # Download data and untar
  local/download_and_untar.sh $data_url $data
fi

if [ $stage -le 2 ]; then
  # Prepare data
  for part in train test; do
    local/data_prep.sh $data/speechocean762/$part data/$part
  done

  mkdir -p data/local
  cp $data/speechocean762/resource/* data/local
fi

if [ $stage -le 3 ]; then
  # Create high-resolution MFCC features
  for part in train test; do
    steps/make_mfcc.sh --nj $nj --mfcc-config conf/mfcc_hires.conf \
      --cmd "$cmd" data/$part || exit 1;
    steps/compute_cmvn_stats.sh data/$part || exit 1;
    utils/fix_data_dir.sh data/$part
  done
fi

if [ $stage -le 4 ]; then
  # Extract ivector
  for part in train test; do
    steps/online/nnet2/extract_ivectors_online.sh --cmd "$cmd" --nj $nj \
      data/$part $ivector_extractor data/$part/ivectors || exit 1;
  done
fi

if [ $stage -le 5 ]; then
  # Compute Log-likelihoods
  for part in train test; do
    steps/nnet3/compute_output.sh --cmd "$cmd" --nj $nj \
      --online-ivector-dir data/$part/ivectors data/$part $model exp/probs_$part
  done
fi

if [ $stage -le 6 ]; then
  # Prepare lang
  local/prepare_dict.sh data/local/lexicon.txt data/local/dict_nosp

  utils/prepare_lang.sh --phone-symbol-table $lang/phones.txt \
    data/local/dict_nosp "<UNK>" data/local/lang_tmp_nosp data/lang_nosp
fi

if [ $stage -le 7 ]; then
  # Split data and make phone-level transcripts
  for part in train test; do
    utils/split_data.sh data/$part $nj
    for i in `seq 1 $nj`; do
      utils/sym2int.pl -f 2- data/lang_nosp/words.txt \
        data/$part/split${nj}/$i/text \
        > data/$part/split${nj}/$i/text.int
    done

    utils/sym2int.pl -f 2- data/lang_nosp/phones.txt \
      data/local/text-phone > data/local/text-phone.int
  done
fi

if [ $stage -le 8 ]; then
  # Make align graphs
  for part in train test; do
    $cmd JOB=1:$nj exp/ali_$part/log/mk_align_graph.JOB.log \
      compile-train-graphs-without-lexicon \
        --read-disambig-syms=data/lang_nosp/phones/disambig.int \
        $model/tree $model/final.mdl \
        "ark,t:data/$part/split${nj}/JOB/text.int" \
        "ark,t:data/local/text-phone.int" \
        "ark:|gzip -c > exp/ali_$part/fsts.JOB.gz"   || exit 1;
    echo $nj > exp/ali_$part/num_jobs
  done
fi

if [ $stage -le 9 ]; then
  # Align
  for part in train test; do
    steps/align_mapped.sh --cmd "$cmd" --nj $nj --graphs exp/ali_$part \
      data/$part exp/probs_$part $lang $model exp/ali_$part
  done
fi

if [ $stage -le 10 ]; then
  # Make a map which converts phones to "pure-phones"
  # "pure-phone" means the phone whose stress and pos-in-word markers are ignored
  # eg. AE1_B --> AE, EH2_S --> EH, SIL --> SIL
  local/remove_phone_markers.pl $lang/phones.txt \
    data/lang_nosp/phones-pure.txt data/lang_nosp/phone-to-pure-phone.int
fi

if [ $stage -le 11 ]; then
  # Convert transition-id to phone-id
  for part in train test; do
    $cmd JOB=1:$nj exp/ali_$part/log/ali_to_phones.JOB.log \
      ali-to-phones --per-frame=true $model/final.mdl \
        "ark,t:gunzip -c exp/ali_$part/ali.JOB.gz|" \
        "ark,t:|gzip -c >exp/ali_$part/ali-phone.JOB.gz"   || exit 1;
  done
fi
