import argparse
import json
import logging
import os


def merge(data_paths, output_path):
    merged = {}

    for data_path in data_paths:
        dataset_name = os.path.basename(data_path).split('.')[0]
        with open(data_path) as f:
            data = json.load(f)
        for sentence_id, sentence in data.items():
            sentence_id = '{}.{}'.format(dataset_name, sentence_id)
            if sentence_id in merged:
                raise ValueError(f'Duplicate sentence ID: {sentence_id}')
            merged[sentence_id] = sentence

    with open(output_path, 'w') as f:
        json.dump(merged, f, sort_keys=True, indent=4)

    return merged


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data',
        type=str,
        nargs='+',
        required=True,
        dest='data_paths',
        help='Paths to the JSON files to merge.')
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        dest='output_path',
        help='Path to the output file.')
    parser.add_argument(
        '--log',
        type=str,
        default='WARNING',
        dest='loglevel',
        help='Log level. Default = WARNING.')
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.loglevel.upper()))

    logging.info('Merging files...')

    merge(args.data_paths, args.output_path)

    logging.info('Done!')