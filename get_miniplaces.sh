#!/usr/bin/env bash
# This scripts downloads the MIT mini-places challenge data and unzips it.
# See http://places.csail.mit.edu/ for more information.

echo "Downloading..."

if [ ! -f data.tar.gz ]; then
  wget http://dl.caffe.berkeleyvision.org/mit_mini_places/data.tar.gz
  tar -xzvf data.tar.gz
fi

if [ ! -f development_kit.tar.gz ]; then
  wget http://dl.caffe.berkeleyvision.org/mit_mini_places/development_kit.tar.gz
  tar -xzvf development_kit.tar.gz
fi

# create test data file test.txt
pushd development_kit/data
cp val.txt test.txt
sed -i 's/^val/test/' test.txt
sed -i 's/ .*$//' test.txt
popd
