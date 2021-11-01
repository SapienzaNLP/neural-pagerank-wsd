from argparse import ArgumentParser
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy import sparse
from fast_pagerank import pagerank_power
from tqdm import trange


def read_edgelist(path):
    edges = []
    vocab = {}
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            src, tgt, *rest = line.split()
            src_i = vocab.setdefault(src, len(vocab))
            tgt_i = vocab.setdefault(tgt, len(vocab))
            if rest:
                w = float(rest[0])
            else:
                w = 1.
            edges.append((src_i, tgt_i, w))
    edges = sorted(edges)
    from_node, to_node, weights = zip(*edges)
    from_node = np.array(from_node)
    to_node = np.array(to_node)
    weights = np.array(weights)

    graph = sparse.csr_matrix(
            (weights, (from_node, to_node)), 
            shape=(len(vocab), len(vocab)))

    return graph, vocab

def write_edgelist(path, edgelist, vocab):
    rev_vocab = {i: n for n, i in vocab.items()}
    with path.open('w') as f:
        for from_n, to_n, w in edgelist:
            from_n = rev_vocab[from_n]
            to_n = rev_vocab[to_n]
            line = f'{from_n}\t{to_n}\t{w}\n'
            f.write(line)

def do_personalized_pagerank(to_n, topk=100, alpha=0.15):
    N = graph.shape[0]
    personalization = np.array([0.] * N)
    personalization[to_n] = 1.
    pr = pagerank_power(graph, p=1.-alpha, personalize=personalization, tol=1e-6)
    pr_topk_i = np.argsort(-pr)[:topk]
    pr_topk_v = pr[pr_topk_i]
    pr_topk_v /= pr_topk_v.sum()
    new_edges = [(int(from_n), to_n, float(w)) for from_n, w in zip(pr_topk_i, pr_topk_v) if float(w) > 0.0]
    return new_edges


    return edges

def make_args():
    parser = ArgumentParser()
    parser.add_argument('input_path', type=Path)
    parser.add_argument('output_path', type=Path)
    parser.add_argument('--topk', type=int, default=100)
    parser.add_argument('--alpha', type=float, default=0.15)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = make_args()
    graph, vocab = read_edgelist(args.input_path)

    def process(to_n):
        return do_personalized_pagerank(to_n, topk=args.topk, alpha=args.alpha)

    edgelist = []
    for i, edges in enumerate(map(process, trange(graph.shape[0]))):
        edgelist.extend(edges)

    write_edgelist(args.output_path, edgelist, vocab)
