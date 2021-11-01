#!/bin/bash

python3 scripts/preprocess/preprocess_raganato.py \
    --data data/original/semcor/semcor.data.xml \
    --keys data/original/semcor/semcor.gold.key.txt \
    --output data/preprocessed/semcor/semcor.json \
    --use_document_context \
    --replace_sentences

python3 scripts/preprocess/preprocess_raganato.py \
    --data data/original/semeval2007/semeval2007.data.xml \
    --keys data/original/semeval2007/semeval2007.gold.key.txt \
    --output data/preprocessed/semeval2007/semeval2007.json \
    --use_document_context \
    --replace_sentences

python3 scripts/preprocess/preprocess_raganato.py \
    --data data/original/semeval2013/semeval2013.data.xml \
    --keys data/original/semeval2013/semeval2013.gold.key.txt \
    --output data/preprocessed/semeval2013/semeval2013.json \
    --use_document_context \
    --replace_sentences

python3 scripts/preprocess/preprocess_raganato.py \
    --data data/original/semeval2015/semeval2015.data.xml \
    --keys data/original/semeval2015/semeval2015.gold.key.txt \
    --output data/preprocessed/semeval2015/semeval2015.json \
    --use_document_context \
    --replace_sentences

python3 scripts/preprocess/preprocess_raganato.py \
    --data data/original/senseval2/senseval2.data.xml \
    --keys data/original/senseval2/senseval2.gold.key.txt \
    --output data/preprocessed/senseval2/senseval2.json \
    --use_document_context \
    --replace_sentences

python3 scripts/preprocess/preprocess_raganato.py \
    --data data/original/senseval3/senseval3.data.xml \
    --keys data/original/senseval3/senseval3.gold.key.txt \
    --output data/preprocessed/senseval3/senseval3.json \
    --use_document_context \
    --replace_sentences

python3 scripts/preprocess/preprocess_raganato.py \
    --data data/original/all/ALL.data.xml \
    --keys data/original/all/ALL.gold.key.txt \
    --output data/preprocessed/all/all.json \
    --use_document_context \
    --replace_sentences
