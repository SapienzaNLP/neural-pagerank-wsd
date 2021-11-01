from argparse import ArgumentParser
from pathlib import Path
from nltk.corpus import wordnet
from tqdm import tqdm

def make_offset(synset):
    return 'wn:' + str(synset.offset()).zfill(8) + synset.pos()

def make_edgelist(
        hypernyms=True,
        hyponyms=True,
        similar_tos=True,
        antonyms=True,
        derivationally_related=True,
    ):

    edges = []

    for syn in tqdm(wordnet.all_synsets()):

        from_n = make_offset(syn)

        if hypernyms:
            for syn2 in syn.hypernyms():
                to_n = make_offset(syn2)
                edges.append((from_n, to_n))

        if hyponyms:
            for syn2 in syn.hyponyms():
                to_n = make_offset(syn2)
                edges.append((from_n, to_n))
        
        if similar_tos:
            for syn2 in syn.similar_tos():
                to_n = make_offset(syn2)
                edges.append((from_n, to_n))
                edges.append((to_n, from_n))

        if antonyms:
            for sense in syn.lemmas():
                for sense2 in sense.antonyms():
                    syn2 = sense2.synset()
                    to_n = make_offset(syn2)
                    edges.append((from_n, to_n))
                    edges.append((to_n, from_n))
        
        if derivationally_related:
            for sense in syn.lemmas():
                for sense2 in sense.derivationally_related_forms():
                    syn2 = sense2.synset()
                    to_n = make_offset(syn2)
                    edges.append((from_n, to_n))

    edges = sorted(list(set(edges)))
    return edges

def write_edgelist(path, edgelist):
    with path.open('w') as f:
        for from_n, to_n in edgelist:
            line = f'{from_n}\t{to_n}\n'
            f.write(line)

def make_args():
    parser = ArgumentParser()
    parser.add_argument('output_path', type=Path)
    parser.add_argument('--hypernyms', action='store_true')
    parser.add_argument('--hyponyms', action='store_true')
    parser.add_argument('--similar_tos', action='store_true')
    parser.add_argument('--antonyms', action='store_true')
    parser.add_argument('--derivationally_related', action='store_true')
    args = parser.parse_args()
    return args

def main():
    args = make_args()
    kwargs = vars(args)
    output = kwargs.pop('output_path')
    edgelist = make_edgelist(**kwargs)
    write_edgelist(output, edgelist)

if __name__ == '__main__':
    main()
