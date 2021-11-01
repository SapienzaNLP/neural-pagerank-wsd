import json
from argparse import ArgumentParser

import numpy as np
from scipy import sparse
from fast_pagerank import pagerank_power
from nltk.corpus import wordnet as wn


def get_synsets():
    synset2id = {}
    for synset in wn.all_synsets():
        synset2id[synset] = len(synset2id)
    id2synset = {i: synset for synset, i in synset2id.items()}
    return synset2id, id2synset


def add_synset_relations(synset_id, related_synsets, synset2id, synset_relations):
    num_related_synsets = len(related_synsets)
    for related_synset in related_synsets:
        related_synset_id = synset2id[related_synset]
        synset_relations[synset_id].append({
            'target': related_synset_id,
            'weight': 1. / num_related_synsets
        })


def get_synset_relations(synset2id, id2synset):
    synset_relations = {}
    for synset_id, synset in id2synset.items():
        synset_relations[synset_id] = []

        add_synset_relations(synset_id, synset.hypernyms(), synset2id, synset_relations)
        add_synset_relations(synset_id, synset.hyponyms(), synset2id, synset_relations)
        add_synset_relations(synset_id, synset.also_sees(), synset2id, synset_relations)
        add_synset_relations(synset_id, synset.similar_tos(), synset2id, synset_relations)
        add_synset_relations(synset_id, synset.instance_hypernyms(), synset2id, synset_relations)
        add_synset_relations(synset_id, synset.instance_hyponyms(), synset2id, synset_relations)
        add_synset_relations(synset_id, synset.verb_groups(), synset2id, synset_relations)

        # for lemma in synset.lemmas():
        #     related_lemmas = lemma.derivationally_related_forms()
        #     num_related_lemmas = len(related_lemmas)
        #     for related_lemma in related_lemmas:
        #         related_synset_id = synset2id[related_lemma.synset()]
        #         synset_relations[synset_id].append({
        #             'target': related_synset_id,
        #             'weight': 1. / num_related_lemmas
        #         })

        #     pertainyms = lemma.pertainyms()
        #     num_pertainyms = len(pertainyms)
        #     for pertainym in pertainyms:
        #         pertainym_synset_id = synset2id[pertainym.synset()]
        #         synset_relations[synset_id].append({
        #             'target': pertainym_synset_id,
        #             'weight': 1. / num_pertainyms
        #         })

    return synset_relations


def build_graph(synset_relations, num_synsets):
    from_synset = []
    to_synset = []
    weights = []

    for source, edges in synset_relations.items():
        for edge in edges:
            from_synset.append(source)
            to_synset.append(edge['target'])
            weights.append(edge['weight'])

    graph = sparse.csr_matrix((weights, (from_synset, to_synset)), shape=(num_synsets, num_synsets))
    return graph


def compute_pagerank(graph, id2synset, f):
    num_synsets = len(id2synset)
    for i, (synset_id, synset) in enumerate(id2synset.items()):
        if i % 100 == 0:
            print('  {}/{}'.format(i, num_synsets))
        personalize = np.zeros((num_synsets,))
        personalize[synset_id] = 1.0
        ppr = pagerank_power(graph, p=0.85, personalize=personalize)
        ppr = [(id2synset[s].name(), v) for s, v in enumerate(ppr)]
        sorted_ppr = sorted(ppr, key=lambda kv: kv[1], reverse=True)[:100]
        sorted_ppr = ['{}={}'.format(s, v) for s, v in sorted_ppr if v > 0.]
        line = ' '.join(sorted_ppr)
        f.write('{} {}\n'.format(synset.name(), line))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--output_graph', type=str, default='data/pr_graph.json')
    parser.add_argument('--output_pagerank', type=str, default='data/pr.txt')
    args = parser.parse_args()

    print('  Getting synsets from WordNet...')
    synset2id, id2synset = get_synsets()
    print('  Getting synset relations from WordNet...')
    synset_relations = get_synset_relations(synset2id, id2synset)

    print('  Writing graph to {}...'.format(args.output_graph))
    with open(args.output_graph, 'w') as f:
        output_graph = {}
        output_graph['nodes'] = {i: synset.name() for i, synset in id2synset.items()}
        output_graph['edges'] = synset_relations
        json.dump(output_graph, f, indent=4, sort_keys=True)

    print('  Building graph...')
    num_synsets = len(synset2id)
    graph = build_graph(synset_relations, num_synsets)

    print('  Computing pagerank...')
    with open(args.output_pagerank, 'w') as f:
        compute_pagerank(graph, id2synset, f)

    print('  Done!')
