import argparse
import json
import logging
import os

from nltk.corpus import wordnet as wn
from nltk.corpus.reader import WordNetError
import numpy as np
from sklearn.decomposition import PCA


patching_data = {
    'ddc%1:06:01::': 'dideoxycytosine.n.01.DDC',
    'ddi%1:06:01::': 'dideoxyinosine.n.01.DDI',
    'earth%1:15:01::': 'earth.n.04.earth',
    'earth%1:17:02::': 'earth.n.01.earth',
    'moon%1:17:03::': 'moon.n.01.moon',
    'sun%1:17:02::': 'sun.n.01.Sun',
    'kb%1:23:01::': 'kilobyte.n.02.kB',
    'kb%1:23:03::': 'kilobyte.n.01.kB',
}


def patched_lemma_from_key(key):
    try:
        lemma = wn.lemma_from_key(key)
    except WordNetError as e:
        if key in patching_data:
            lemma = wn.lemma(patching_data[key])
        elif '%3' in key:
            lemma = wn.lemma_from_key(key.replace('%3', '%5'))
        else:
            raise e
    return lemma


def preprocess_embeddings(lmms_path, output_path, n_components=512):
    sense_vectors = {}
    is_noun = lambda s: sense[sense.index('%') + 1] != '1'

    logging.info('Reading embeddings...')
    with open(lmms_path) as f:
        for line in f:
            sense, *values = line.strip().split()
            if not is_noun(sense):
                values = [float(v) for v in values]
                assert len(values) == 2048
                sense_vectors[sense] = values
    
    logging.info('Converting sense embeddings into synset embeddings...')

    synset_indices = {}
    sense_counts = {}
    synset_vectors = np.zeros((len(sense_vectors), 2048))
    for sense, vector in sense_vectors.items():
        synset = patched_lemma_from_key(sense).synset().name()
        if synset not in synset_indices:
            synset_indices[synset] = len(synset_indices)
            sense_counts[synset] = 0
        synset_index = synset_indices[synset]
        synset_vectors[synset_index] += np.array(vector)
        sense_counts[synset] += 1
    
    for synset, synset_index in synset_indices.items():
        synset_vectors[synset_index] /= sense_counts[synset]
    
    logging.info('Running PCA...')
    
    pca = PCA(n_components=n_components, random_state=42)
    reduced_synset_vectors = pca.fit_transform(synset_vectors)

    logging.info('Writing output to file...')

    sorted_synsets = sorted(list(synset_indices.keys()))
    with open(output_path, 'w') as f:
        for synset in sorted_synsets:
            synset_index = synset_indices[synset]
            synset_vector = reduced_synset_vectors[synset_index].tolist()
            synset_vector = [str(v) for v in synset_vector]
            line = '{} {}\n'.format(synset, ' '.join(synset_vector))
            f.write(line)

    logging.info('Done!')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--embeddings_file', type=str, required=True, help='Path to your embeddings file to preprocess.')
    # parser.add_argument('--sensembert', type=str, required=True, help='Path to SensEmBERT.')
    parser.add_argument('--output_size', type=int, default=512, help='Number of components for the output vectors.')
    parser.add_argument('--output', type=str, required=True, dest='output_path', help='Path to the output embedding file.')
    parser.add_argument('--log', type=str, default='WARNING', dest='loglevel', help='Log level. Default = WARNING.')
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.loglevel.upper()))

    logging.info('Preprocessing embeddings...')
    preprocess_embeddings(args.lmms, args.sensembert, args.output_path, n_components=args.output_size)
    logging.info('Done!')