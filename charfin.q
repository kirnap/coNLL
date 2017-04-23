#!/bin/bash
#$ -N FrenchCONLL
#$ -S /bin/bash
#$ -q ai.q@ahtapot-5-1
#$ -pe smp 1
#$ -cwd
#$ -e /dev/null
#$ -o /dev/null
#$ -M okirnap@ku.edu.tr
#$ -m bea
#$ -l gpu=1
julia charbased_train.jl --trainfile data/French/all_french.txt --vocabfile data/French/French_out_vocab  --wordsfile data/French/French_100k_vocab --charlines 20000 --savefile trained_models/french_chmodel.jld &> logs/char_french.txt

