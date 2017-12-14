#!/usr/bin/env bash
# Create the imagenet lmdb inputs
# N.B. set the path to the imagenet train + val data dirs
set -e

if [[ $# -ne 8 ]]; then 
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

TRAIN_DATA_ROOT=$4
TEST_DATA_ROOT=$5

if [[ $6 -eq 'true' ]]; then
    RESIZE=true
else
    RESIZE=false
fi
IMG_HEIGHT=$7
IMG_WIDTH=$8
#TRAIN_DATA_ROOT=/home/hack17/cd/train/
#TEST_DATA_ROOT=/home/hack17/cd/train/

# Set RESIZE=true to resize the images to 256x256. Leave as false if images have
# already been resized using another tool.

if $RESIZE; then
  RESIZE_HEIGHT=$IMG_HEIGHT
  RESIZE_WIDTH=$IMG_WIDTH
else
  RESIZE_HEIGHT=0
  RESIZE_WIDTH=0
fi

if [ ! -d "$TRAIN_DATA_ROOT" ]; then
  echo "Error: TRAIN_DATA_ROOT is not a path to a directory: $TRAIN_DATA_ROOT"
  echo "Set the TRAIN_DATA_ROOT variable in create_imagenet.sh to the path" \
       "where the ImageNet training data is stored."
  exit 1
fi

if [ ! -d "$TEST_DATA_ROOT" ]; then
  echo "Error: TEST_DATA_ROOT is not a path to a directory: $TEST_DATA_ROOT"
  echo "Set the TEST_DATA_ROOT variable in create_imagenet.sh to the path" \
       "where the ImageNet validation data is stored."
  exit 1
fi

echo "Creating train lmdb..."

rm -rf $EXAMPLE/${prefix}_train_lmdb
GLOG_logtostderr=1 $TOOLS/convert_imageset \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    --shuffle \
    $TRAIN_DATA_ROOT \
    $DATA/train.txt \
    $EXAMPLE/${prefix}_train_lmdb

echo "Creating val lmdb..."

rm -rf $EXAMPLE/${prefix}_test_lmdb
GLOG_logtostderr=1 $TOOLS/convert_imageset \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    --shuffle \
    $TEST_DATA_ROOT \
    $DATA/test.txt \
    $EXAMPLE/${prefix}_test_lmdb

echo "Done."
