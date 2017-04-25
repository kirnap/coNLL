#!/usr/bin/env bash

vocab_count="$1_vocab_count"
out_name="$1_out_vocab2"
all_name="$1_100k_vocab2"
head -20000 $vocab_count > temp
awk '{$1="";print $0}' temp > $out_name
rm temp

head -100000 $vocab_count > temp
awk '{$1="";print $0}' temp > $all_name
rm temp
