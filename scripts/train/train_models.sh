#!/bin/bash

# Semcor + relations
python3 train.py --name bert-large --language_model bert-large-cased \
    --include_similar \
    --include_related \
    --include_verb_groups \
    --include_also_see

python3 train.py --name bert-large --language_model bert-large-cased \
    --include_similar \
    --include_related \
    --include_verb_groups \
    --include_also_see \
    --include_hypernyms

python3 train.py --name bert-large --language_model bert-large-cased \
    --include_similar \
    --include_related \
    --include_verb_groups \
    --include_also_see \
    --include_hyponyms

python3 train.py --name bert-large --language_model bert-large-cased \
    --include_similar \
    --include_related \
    --include_verb_groups \
    --include_also_see \
    --include_hypernyms \
    --include_hyponyms

python3 train.py --name bert-large --language_model bert-large-cased \
    --include_similar \
    --include_related \
    --include_verb_groups \
    --include_also_see \
    --include_hypernyms \
    --include_hyponyms \
    --include_instance_hypernyms \
    --include_instance_hyponyms

# Semcor + tagged glosses + examples
python3 train.py --name bert-large --language_model bert-large-cased \
    --train_path data/preprocessed/glosses/semcor.glosses.examples.json \
    --include_similar \
    --include_related \
    --include_verb_groups \
    --include_also_see \
    --include_hypernyms \
    --include_hyponyms