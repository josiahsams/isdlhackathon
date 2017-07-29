#!/usr/bin/env sh
# Compute the mean image from the imagenet training lmdb
# N.B. this is available in data/catdog


if [[ $# -ne 3 ]]; then
        echo "Not sufficient arguments"
        exit
fi

prefix=$1

# output directory for LMDB
EXAMPLE=$3
#EXAMPLE=/home/hack17/joe

# .txt file location
DATA=$2
#DATA=/home/hack17/joe

# Check if TOOLS is unset
if [ -z ${TOOLS+x} ];
then
TOOLS=./build/tools
fi


$TOOLS/compute_image_mean $EXAMPLE/${prefix}_train_lmdb \
  $DATA/${prefix}_mean.binaryproto

echo "Done."
