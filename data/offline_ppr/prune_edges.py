from argparse import ArgumentParser
from pathlib import Path

def read_edges(path):
    edges = {}
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            src, tgt, *rest = line.split()
            out = edges.setdefault(tgt, [])
            if rest:
                w = float(rest[0])
            else:
                w = 1.
            out.append((src, w))
    return edges

def write_edges(path, edges):
    with path.open('w') as f:
        for to_n, out in edges.items():
            for from_n, w in out:
                line = f'{from_n}\t{to_n}\t{w}\n'
                f.write(line)

def prune(edges, topk=10):
    for out in edges.values():
        out[:] = sorted(out, key=lambda e: -e[1])[:topk]
        z = sum([e[1] for e in out])
        out[:] = [(e[0], e[1] / z) for e in out]
    return edges

def make_args():
    parser = ArgumentParser()
    parser.add_argument('input_path', type=Path)
    parser.add_argument('output_path', type=Path)
    parser.add_argument('--topk', type=int, default=10)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = make_args()
    edges = read_edges(args.input_path)
    prune(edges, args.topk)
    write_edges(args.output_path, edges)

