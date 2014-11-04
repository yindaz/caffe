#!/usr/bin/env sh
# Create the imagenet lmdb inputs
# N.B. set the path to the imagenet train + val data dirs
GOOGLE_LOG_DIR=models/positive_pure/log \
/u/yindaz/caffe/build/tools/convert_imageset \
--resize_height=256 \
--resize_width=256 \
--shuffle \
/ \
/n/fs/lsun/PLACE205/mixdata/all_train_data_inuse.txt \
/n/fs/lsun/PLACE205/all_train_data_inuse_lmdb
