# Training script
python3 train.py --name <exp_name> \
                 --train_path data/preprocessed/glosses/semcor.glosses.untagged.json
                 --dev_path data/preprocessed/semeval2007/semeval2007.json
                 --include_hypernyms \
                 --include_hyponyms \
                 --include_similar \
                 --include_related \
                 --include_also_see \
                 --include_verb_groups \
                 --include_pertainyms \
                 --include_pagerank \
                 --pagerank_k 10 \
                 --loss_type cross_entropy
                 --synset_embeddings_path data/embeddings/synset_embeddings.txt
                 --use_synset_embeddings
                 --use_graph_convolution
                 --use_trainable_graph
                 --power_iterations 10
                 --alpha 0.15