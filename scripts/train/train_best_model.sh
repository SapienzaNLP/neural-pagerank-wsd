#!/bin/bash

python3 train.py --name bert-large --language_model bert-large-cased \
    --train_path data/preprocessed/glosses/semcor.glosses.untagged.json \
    --dev_path data/preprocessed/semeval2007/semeval2007.json \
    --include_similar \
    --include_related \
    --include_verb_groups \
    --include_also_see \
    --include_hypernyms \
    --include_hyponyms \
    --use_synset_embeddings \
    --synset_embeddings_path embeddings/synset_embeddings.txt \
    --alpha 0.15 \
    --power_iterations 10 \
    --loss_type cross_entropy