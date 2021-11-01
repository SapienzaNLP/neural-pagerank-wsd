#!/bin/bash

python3 scripts/preprocess/preprocess_raganato.py \
    --data data/original/glosses/glosses_main.untagged.data.xml \
    --keys data/original/glosses/glosses_main.untagged.gold.key.txt \
    --untagged_glosses \
    --output data/preprocessed/glosses/glosses.untagged.json

python3 scripts/preprocess/preprocess_raganato.py \
    --data data/original/glosses/glosses_main.data.xml \
    --keys data/original/glosses/glosses_main.gold.key.txt \
    --output data/preprocessed/glosses/glosses.json

python3 scripts/preprocess/preprocess_raganato.py \
    --data data/original/glosses/examples.data.xml \
    --keys data/original/glosses/examples.gold.key.txt \
    --output data/preprocessed/glosses/examples.json

python3 scripts/preprocess/merge_data.py \
    --data data/preprocessed/glosses/glosses.untagged.json data/preprocessed/semcor/semcor.json \
    --output data/preprocessed/glosses/semcor.glosses.untagged.json

python3 scripts/preprocess/merge_data.py \
    --data data/preprocessed/glosses/glosses.json data/preprocessed/glosses/examples.json data/preprocessed/semcor/semcor.json \
    --output data/preprocessed/glosses/semcor.glosses.examples.json
