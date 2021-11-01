#!/bin/bash

python scripts/preprocess/preprocess_sense_embeddings.py \
    --embeddings file ../../Data/embeddings/ares_bert_large.txt \
    --output embeddings/ares_synset_embeddings.txt
    --output_size 512 \
    --log DEBUG