import argparse
import json
import logging
import os
import xml.etree.ElementTree as ET

from nltk.corpus import wordnet as wn


def parse(data_path, keys_path, untagged_glosses, use_document_context, use_document_senses, replace_sentences, min_sentence_length=32):
    all_synsets = set()
    instance2sense = {}
    with open(keys_path) as f:
        for line in f:
            if not line.strip():
                continue
            instance_id, *sense_keys = line.strip().split()
            synsets = [wn.lemma_from_key(k).synset().name() for k in sense_keys]
            all_synsets.update(synsets)
            instance2sense[instance_id] = synsets
    print('# synsets:', len(all_synsets))

    tree = ET.parse(data_path)
    root = tree.getroot()

    data = {}

    for sentence in root.iter('sentence'):
        sentence_id = sentence.attrib['id']
        words = []
        lemmas = []
        pos_tags = []
        instance_ids = {}
        senses = {}

        for i, element in enumerate(sentence):
            if i == 0:
                words.append(element.text.capitalize().replace('_', ' '))
            else:
                words.append(element.text.replace('_', ' '))
            lemmas.append(element.attrib['lemma'])
            pos_tags.append(element.attrib['pos'])
            if element.tag == 'instance':
                instance_id = element.attrib['id']
                if untagged_glosses and instance_id[-4] == 't':
                    continue
                instance_ids[i] = instance_id
                senses[i] = instance2sense[instance_id]

        data[sentence_id] = {
            'words': words,
            'lemmas': lemmas,
            'pos_tags': pos_tags,
            'instance_ids': instance_ids,
            'senses': senses,
        }

    if use_document_context:
        expanded_data = {}
        for sentence_id, sentence in data.items():
            if len(sentence['words']) >= min_sentence_length:
                continue
            parts = sentence_id.split('.')
            doc_id = '.'.join(parts[:-1])
            sentence_number = int(parts[-1][1:])

            # successive_sentence_id = '{}.s{:03d}'.format(doc_id, sentence_number + 1)

            # if successive_sentence_id in data:
            #     successive_sentence = data[successive_sentence_id]
            #     if successive_sentence['instance_ids']:
            #         if not replace_sentences:
            #             new_sentence_id = '{}.s{:03d}.s{:03d}'.format(doc_id, sentence_number, sentence_number + 1)
            #             expanded_data[new_sentence_id] = merge_sentences(sentence, successive_sentence, add_s2_senses=use_document_senses)
            #         else:
            #             expanded_data[sentence_id] = merge_sentences(sentence, successive_sentence, add_s2_senses=use_document_senses)

            previous_sentence_id = '{}.s{:03d}'.format(doc_id, sentence_number - 1)

            if previous_sentence_id in data:
                previous_sentence = data[previous_sentence_id]
                if previous_sentence['instance_ids']:
                    if not replace_sentences:
                        new_sentence_id = '{}.s{:03d}.s{:03d}'.format(doc_id, sentence_number - 1, sentence_number)
                        expanded_data[new_sentence_id] = merge_sentences(previous_sentence, sentence, add_s1_senses=use_document_senses, add_s2_senses=True)
                    else:
                        expanded_data[sentence_id] = merge_sentences(previous_sentence, sentence, add_s1_senses=use_document_senses, add_s2_senses=True)

        data.update(expanded_data)

    return data


def merge_sentences(s1, s2, add_s1_senses=False, add_s2_senses=False):
    sentence = {
        'words': s1['words'] + s2['words'],
        'lemmas': s1['lemmas'] + s2['lemmas'],
        'pos_tags': s1['pos_tags'] + s2['pos_tags'],
    }

    l1 = len(s1['words'])
    instance_ids = {}
    if add_s1_senses:
        instance_ids = {k: v for k, v in s1['instance_ids'].items()}
    if add_s2_senses:
        for k, v in s2['instance_ids'].items():
            instance_ids[k + l1] = v

    senses = {}
    if add_s1_senses:
        senses = {k: v for k, v in s1['senses'].items()}
    if add_s2_senses:
        for k, v in s2['senses'].items():
            senses[k + l1] = v

    sentence['instance_ids'] = instance_ids
    sentence['senses'] = senses
    return sentence


def write_parsed_data(data, path):
    output = json.dumps(data, indent=4, sort_keys=True)

    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))

    with open(path, 'w') as f:
        f.write(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data',
        type=str,
        required=True,
        dest='data_path',
        help='Path to the XLM file to preprocess.')
    parser.add_argument(
        '--keys',
        type=str,
        required=True,
        dest='keys_path',
        help='Path to the keys file to preprocess.')
    parser.add_argument(
        '--use_document_context',
        action='store_true',
        help='Set this flag to use document context.')
    parser.add_argument(
        '--use_document_senses',
        action='store_true',
        help='Set this flag to create longer (annotated) sentences from two sentences (works only if --use_document_context is set).')
    parser.add_argument(
        '--replace_sentences',
        action='store_true',
        help='Set this flag to use replace the original sentences with contextualized sentences (works only if --use_document_context is set).')
    parser.add_argument(
        '--untagged_glosses',
        action='store_true',
        required=False,
        help='Set this flag to preprocess untagged glosses.')
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

    logging.info('Parsing {}...'.format(args.data_path))

    parsed_data = parse(args.data_path, args.keys_path, args.untagged_glosses, args.use_document_context, args.use_document_senses, args.replace_sentences)
    write_parsed_data(parsed_data, args.output_path)

    logging.info('Done!')
