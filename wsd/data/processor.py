from os.path import isfile 
import json
import random

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer

from nltk.corpus import wordnet as wn
from nltk.corpus.reader.wordnet import WordNetError
from tqdm.auto import tqdm


class Processor(object):

    _pos_classes = {
        'NOUN': 'n',
        'VERB': 'v',
        'ADJ': 'a',
        'ADV': 'r',
        'PRT': 'r',
    }

    def __init__(
            self,
            language_model='',
            loss_type='cross_entropy',
            num_negative_samples=-1,
            include_similar_synsets=False,
            include_related_synsets=False,
            include_verb_group_synsets=False,
            include_syntags=False,
            include_hypernym_synsets=False,
            include_hyponym_synsets=False,
            include_instance_hypernym_synsets=False,
            include_instance_hyponym_synsets=False,
            include_also_see_synsets=False,
            include_pertainym_synsets=False,
            include_pagerank_synsets=False,
            pagerank_path='data/pr.txt',
            offline_pagerank_path=None,
            pagerank_k=10,
            use_synder=False,
            unknown_token='<unk>',
            graph_file_path='data/wn_graph.json',
            _load_from_config=False,
            **kwargs):

        super(Processor, self).__init__()

        self.loss_type = loss_type
        self.num_negative_samples = num_negative_samples
        self.include_similar_synsets = include_similar_synsets
        self.include_related_synsets = include_related_synsets
        self.include_verb_group_synsets = include_verb_group_synsets
        self.include_hypernym_synsets = include_hypernym_synsets
        self.include_syntag = include_syntags
        self.include_hyponym_synsets = include_hyponym_synsets
        self.include_instance_hypernym_synsets = include_instance_hypernym_synsets
        self.include_instance_hyponym_synsets = include_instance_hyponym_synsets
        self.include_also_see_synsets = include_also_see_synsets
        self.include_pertainym_synsets = include_pertainym_synsets
        self.include_pagerank_synsets = include_pagerank_synsets
        self.pagerank_k = pagerank_k
        self.padding_target_id = -1
        self.synset_offset2sense_key = self._fetch_synsetoffset2sensekey_mappings()

        self.language_model = language_model
        if self.language_model:
            self.tokenizer = AutoTokenizer.from_pretrained(self.language_model)
            self.padding_token_id = self.tokenizer.pad_token_id
            self.unknown_token_id = self.tokenizer.unk_token_id

        if not _load_from_config:
            maps = Processor._build_maps(pagerank_path, pagerank_k)
            self.synset2id = maps['synset2id']
            self.id2synset = maps['id2synset']
            self.unknown_synset_id = self.synset2id[unknown_token]
            self.num_synsets = len(self.synset2id)
            self.word2synsets = maps['word2synsets']
            self.synset2hypernyms = maps['synset2hypernyms']
            self.synset2hyponyms = maps['synset2hyponyms']
            self.synset2similars = maps['synset2similars']
            self.synset2groups = maps['synset2groups']
            self.synset2related = maps['synset2related']
            self.synset2instance_hypernyms = maps['synset2instance_hypernyms']
            self.synset2instance_hyponyms = maps['synset2instance_hyponyms']
            self.synset2also_see = maps['synset2also_see']
            self.synset2pertainyms = maps['synset2pertainyms']
            self.synset2pagerank = maps['synset2pagerank']
            self.synset2syntags = maps['synset2syntags']
            if offline_pagerank_path:
                self._build_graph_offline_ppr(offline_pagerank_path, graph_file_path)
            else:
                Processor._build_graph_wrapper(self.synset2id, self.synset2similars, self.synset2groups,
                                               self.synset2related, self.synset2hypernyms, self.synset2hyponyms,
                                               self.synset2also_see, self.synset2pertainyms, self.synset2syntags,
                                               graph_file_path, use_synder)

    def encode_sentence(self, sentence, MAX_LENGTH=500):
        word_ids = []
        subword_indices = []
        sequence_length = len(sentence['words']) + len({'[CLS]', '[SEP]'})
        tokenized_sequence_length = sequence_length

        tokenized_sentence = []
        subword_indices = [1]  # CLS index = 1
        for word_index, word in enumerate(sentence['words']):
            tokenized_word = self.tokenizer.tokenize(word)
            tokenized_sentence.extend(tokenized_word)
            subword_indices.extend([word_index + 2] * len(tokenized_word))
            if len(tokenized_sentence) > MAX_LENGTH:
                tokenized_sentence = tokenized_sentence[:MAX_LENGTH]
                subword_indices = subword_indices[:MAX_LENGTH]
                sequence_length = word_index + 2
                break
        # SEP index = sequence_length + PAD + CLS + 1
        subword_indices.append(word_index + 3)
        word_ids = self.tokenizer.encode(
            tokenized_sentence) + [self.padding_token_id]
        subword_indices.append(0)  # PAD index = 0
        tokenized_sequence_length = len(word_ids) - 1  # Exclude PAD

        return {
            'word_ids': torch.as_tensor(word_ids),
            'subword_indices': torch.as_tensor(subword_indices),
            'sequence_length': sequence_length,
            'tokenized_sequence_length': tokenized_sequence_length,
        }

    def encode_labels(self, sentence):
        synset_ids = []
        synset_values = []
        synset_indices = []
        synset_weights = []
        synset_candidates = []
        negative_samples = []

        for synset_index, synsets in sentence['senses'].items():

            lemma = sentence['lemmas'][synset_index]
            pos = sentence['pos_tags'][synset_index]
            if pos not in Processor._pos_classes:
                print('Invalid POS ({}) for lemma ({}).'.format(pos, lemma))
                continue

            pos = Processor._pos_classes[pos]

            if lemma.lower() not in self.word2synsets:
                print('Invalid lemma ({}).'.format(lemma))
                continue

            _synset_candidates = self.word2synsets[lemma.lower()][pos]

            synset_indices.append(synset_index + 1)
            synset_candidates.append(_synset_candidates)

            _synset_ids = []
            _synset_values = []
            _synset_weights = []

            if self.loss_type == 'cross_entropy':
                synsets = synsets[:1]

            for synset in synsets:
                synset_id = self.synset2id[synset] if synset in self.synset2id else self.unknown_synset_id
                _synset_ids.append(synset_id)
                _synset_values.append(1.0)
                _synset_weights.append(1.0)

                if self.loss_type != 'cross_entropy' and synset_id != self.unknown_synset_id:

                    if self.include_similar_synsets:
                        num_similar_synsets = len(
                            self.synset2similars[synset_id])
                        for similar_id in self.synset2similars[synset_id]:
                            if similar_id not in _synset_candidates and similar_id not in _synset_ids:
                                _synset_ids.append(similar_id)
                                _synset_values.append(1.0)
                                _synset_weights.append(
                                    1. / num_similar_synsets)

                    if self.include_related_synsets:
                        num_related_synsets = len(
                            self.synset2related[synset_id])
                        for related_id in self.synset2related[synset_id]:
                            if related_id not in _synset_candidates and related_id not in _synset_ids:
                                _synset_ids.append(related_id)
                                _synset_values.append(1.0)
                                _synset_weights.append(
                                    1. / num_related_synsets)

                    if self.include_syntag:
                        num_syntags = len(self.synset2syntags[synset_id])
                        for syntag_id in self.synset2syntags[synset_id]:
                            if syntag_id not in _synset_candidates and syntag_id not in _synset_ids:
                                _synset_ids.append(syntag_id)
                                _synset_values.append(1.0)
                                _synset_weights.append(1. / num_syntags)

                    if self.include_verb_group_synsets:
                        num_verb_group_synsets = len(
                            self.synset2groups[synset_id])
                        for group_id in self.synset2groups[synset_id]:
                            if group_id not in _synset_candidates and group_id not in _synset_ids:
                                _synset_ids.append(group_id)
                                _synset_values.append(1.0)
                                _synset_weights.append(
                                    1. / num_verb_group_synsets)

                    if self.include_hypernym_synsets:
                        num_hypernym_synsets = len(
                            self.synset2hypernyms[synset_id])
                        for hypernym_id in self.synset2hypernyms[synset_id]:
                            if hypernym_id not in _synset_candidates and hypernym_id not in _synset_ids:
                                _synset_ids.append(hypernym_id)
                                _synset_values.append(1.0)
                                _synset_weights.append(
                                    1. / num_hypernym_synsets)
                                # _num_hypernym_synsets = len(self.synset2hypernyms[hypernym_id])
                                # for _hypernym_id in self.synset2hypernyms[hypernym_id]:
                                #     if _hypernym_id not in _synset_candidates and _hypernym_id not in _synset_ids:
                                #         _synset_ids.append(_hypernym_id)
                                #         _synset_values.append(1.0)
                                #         _synset_weights.append(1. / (num_hypernym_synsets * _num_hypernym_synsets))

                    if self.include_hyponym_synsets:
                        num_hyponym_synsets = len(
                            self.synset2hyponyms[synset_id])
                        for hyponym_id in self.synset2hyponyms[synset_id]:
                            if hyponym_id not in _synset_candidates and hyponym_id not in _synset_ids:
                                _synset_ids.append(hyponym_id)
                                _synset_values.append(1.0)
                                _synset_weights.append(
                                    1. / num_hyponym_synsets)
                                # _num_hyponym_synsets = len(self.synset2hyponyms[hyponym_id])
                                # for _hyponym_id in self.synset2hyponyms[hyponym_id]:
                                #     if _hyponym_id not in _synset_candidates and _hyponym_id not in _synset_ids:
                                #         _synset_ids.append(_hyponym_id)
                                #         _synset_values.append(1.0)
                                #         _synset_weights.append(1. / (num_hyponym_synsets * _num_hyponym_synsets))

                    if self.include_instance_hypernym_synsets:
                        num_instance_hypernym_synsets = len(
                            self.synset2instance_hypernyms[synset_id])
                        for hypernym_id in self.synset2instance_hypernyms[synset_id]:
                            if hypernym_id not in _synset_candidates and hypernym_id not in _synset_ids:
                                _synset_ids.append(hypernym_id)
                                _synset_values.append(1.0)
                                _synset_weights.append(
                                    1. / num_instance_hypernym_synsets)

                    if self.include_instance_hyponym_synsets:
                        num_instance_hyponym_synsets = len(
                            self.synset2instance_hyponyms[synset_id])
                        for hyponym_id in self.synset2instance_hyponyms[synset_id]:
                            if hyponym_id not in _synset_candidates and hyponym_id not in _synset_ids:
                                _synset_ids.append(hyponym_id)
                                _synset_values.append(1.0)
                                _synset_weights.append(
                                    1. / num_instance_hyponym_synsets)

                    if self.include_also_see_synsets:
                        num_also_see_synsets = len(
                            self.synset2also_see[synset_id])
                        for also_see in self.synset2also_see[synset_id]:
                            if also_see not in _synset_candidates and also_see not in _synset_ids:
                                _synset_ids.append(also_see)
                                _synset_values.append(1.0)
                                _synset_weights.append(
                                    1. / num_also_see_synsets)

                    if self.include_pertainym_synsets:
                        num_pertainym_synsets = len(
                            self.synset2pertainyms[synset_id])
                        for pertainym in self.synset2pertainyms[synset_id]:
                            if pertainym not in _synset_candidates and pertainym not in _synset_ids:
                                _synset_ids.append(pertainym)
                                _synset_values.append(1.0)
                                _synset_weights.append(
                                    1. / num_pertainym_synsets)

                    if self.include_pagerank_synsets:
                        num_pagerank_synsets = len(
                            self.synset2pagerank[synset_id])
                        for neighbor in self.synset2pagerank[synset_id]:
                            if neighbor not in _synset_candidates and neighbor not in _synset_ids:
                                _synset_ids.append(neighbor)
                                _synset_values.append(1.0)
                                _synset_weights.append(
                                    1. / num_pagerank_synsets)

            synset_ids.append(_synset_ids)
            synset_values.extend(_synset_values)
            synset_weights.extend(_synset_weights)

            num_synsets = len(self.synset2id) - 1
            _negative_samples = []
            while len(_negative_samples) < self.num_negative_samples:
                _negative_sample = random.randint(0, num_synsets)
                if _negative_sample not in _negative_samples and _negative_sample not in _synset_ids and _negative_sample not in _synset_candidates:
                    _negative_samples.append(_negative_sample)
            negative_samples.append(_negative_samples)

        return {
            'synsets': synset_ids,
            'synset_values': synset_values,
            'synset_indices': synset_indices,
            'synset_weights': synset_weights,
            'synset_candidates': synset_candidates,
            'negative_samples': negative_samples,
        }

    def decode(self, x, y):
        instance_ids = x['instance_ids']
        synset_indices = list(map(list, zip(*x['synset_indices'])))

        mask = torch.zeros_like(y['synsets']) - 50.0
        mask[x['synset_candidates']] = 0.0

        synset_ids = torch.argmax(y['synsets'] + mask, dim=-1).tolist()

        synsets = {}
        for (sentence_index, synset_index), synset_id in zip(synset_indices, synset_ids):
            instance_id = instance_ids[sentence_index][synset_index - 1]
            synset = self.id2synset[synset_id]
            synsets[instance_id] = synset

        return synsets

    def collate_sentences(self, sentences):
        batched_x = {
            'sentence_ids': [],
            'instance_ids': [],
            'instance_lemmas': [],
            'synset_indices': [[], []],
            'synset_candidates': [[], []],

            'word_ids': [],
            'subword_indices': [],
            'sequence_lengths': [],
            'tokenized_sequence_lengths': [],
        }

        batched_y = {
            'synsets': [[], []],
            'synset_values': [],
            'synset_weights': [],
            'negative_samples': [[], []],
        }

        max_sequence_length = 0
        num_synsets = 0
        for sentence_index, sentence in enumerate(sentences):
            encoded_sentence = self.encode_sentence(sentence)
            encoded_labels = self.encode_labels(sentence)

            sequence_length = encoded_sentence['sequence_length']
            max_sequence_length = max(max_sequence_length, sequence_length)

            batched_x['sentence_ids'].append(sentence['sentence_id'])
            batched_x['instance_ids'].append(sentence['instance_ids'])
            batched_x['instance_lemmas'].append({l_i: l for l_i, l in enumerate(
                sentence['lemmas']) if l_i in sentence['instance_ids']})
            batched_x['synset_indices'][0].extend(
                [sentence_index] * len(encoded_labels['synset_indices']))
            batched_x['synset_indices'][1].extend(
                encoded_labels['synset_indices'])

            for synsets, synset_candidates, negative_samples in zip(encoded_labels['synsets'], encoded_labels['synset_candidates'], encoded_labels['negative_samples']):
                batched_y['synsets'][0].extend([num_synsets] * len(synsets))
                batched_y['synsets'][1].extend(synsets)
                batched_y['negative_samples'][0].extend(
                    [num_synsets] * len(negative_samples))
                batched_y['negative_samples'][1].extend(negative_samples)
                batched_x['synset_candidates'][0].extend(
                    [num_synsets] * len(synset_candidates))
                batched_x['synset_candidates'][1].extend(synset_candidates)
                num_synsets += 1

            batched_y['synset_values'].extend(encoded_labels['synset_values'])
            batched_y['synset_weights'].extend(
                encoded_labels['synset_weights'])

            batched_x['word_ids'].append(encoded_sentence['word_ids'])
            batched_x['subword_indices'].append(
                encoded_sentence['subword_indices'])
            batched_x['sequence_lengths'].append(
                encoded_sentence['sequence_length'])
            batched_x['tokenized_sequence_lengths'].append(
                encoded_sentence['tokenized_sequence_length'])

        batched_x['word_ids'] = pad_sequence(
            batched_x['word_ids'], batch_first=True, padding_value=self.padding_token_id)
        batched_x['sequence_lengths'] = torch.as_tensor(
            batched_x['sequence_lengths'])
        batched_x['tokenized_sequence_lengths'] = torch.as_tensor(
            batched_x['tokenized_sequence_lengths'])

        batched_x['subword_indices'] = pad_sequence(
            batched_x['subword_indices'], batch_first=True, padding_value=0)
        batched_y['synsets'] = torch.as_tensor(batched_y['synsets'])
        batched_y['synset_values'] = torch.as_tensor(
            batched_y['synset_values'])
        batched_y['synset_weights'] = torch.as_tensor(
            batched_y['synset_weights'])

        return batched_x, batched_y

    def save_config(self, path):
        config = {
            'padding_target_id': self.padding_target_id,
            'language_model': self.language_model,
            'loss_type': self.loss_type,
            'num_negative_samples': self.num_negative_samples,

            'padding_token_id': self.padding_token_id,
            'unknown_token_id': self.unknown_token_id,

            'synset2id': self.synset2id,
            'unknown_synset_id': self.unknown_synset_id,
            'num_synsets': self.num_synsets,

            'word2synsets': self.word2synsets,
            'synset2hypernyms': self.synset2hypernyms,
            'synset2hyponyms': self.synset2hyponyms,
            'synset2similars': self.synset2similars,
            'synset2groups': self.synset2groups,
            'synset2related': self.synset2related,
            'synset2instance_hypernyms': self.synset2instance_hypernyms,
            'synset2instance_hyponyms': self.synset2instance_hyponyms,
            'synset2also_see': self.synset2also_see,
            'synset2pertainyms': self.synset2pertainyms,
            'synset2pagerank': self.synset2pagerank,
        }

        with open(path, 'w') as f:
            json.dump(config, f, sort_keys=True, indent=4)

    def _fetch_synsetoffset2sensekey_mappings(self, file_path='data/sensekey2offsets.txt'):
        synset_offset2sense_key = {}
        with open(file_path, encoding='utf-8', mode='r') as mappings_file:
            mappings = mappings_file.read().splitlines()

        for mapping in mappings:
            sensekey, offset = mapping.split()
            synset_offset2sense_key[offset] = sensekey
        return synset_offset2sense_key

    @staticmethod
    def from_config(path):
        with open(path) as f:
            config = json.load(f)

        processor = Processor(_load_from_config=True)
        processor.padding_target_id = config['padding_target_id']
        processor.language_model = config['language_model']
        processor.loss_type = config['loss_type']
        processor.num_negative_samples = config['num_negative_samples']

        processor.padding_token_id = config['padding_token_id']
        processor.unknown_token_id = config['unknown_token_id']

        processor.synset2id = config['synset2id']
        processor.id2synset = {
            int(id): synset for synset, id in processor.synset2id.items()}
        processor.unknown_synset_id = config['unknown_synset_id']
        processor.num_synsets = config['num_synsets']

        processor.word2synsets = config['word2synsets']
        processor.synset2hypernyms = {int(synset): [int(
            h) for h in hypernyms] for synset, hypernyms in config['synset2hypernyms'].items()}
        processor.synset2hyponyms = {int(synset): [int(
            h) for h in hyponyms] for synset, hyponyms in config['synset2hyponyms'].items()}
        processor.synset2similars = {int(synset): [int(
            s) for s in similars] for synset, similars in config['synset2similars'].items()}
        processor.synset2groups = {int(synset): [int(
            g) for g in groups] for synset, groups in config['synset2groups'].items()}
        processor.synset2related = {int(synset): [int(
            r) for r in related] for synset, related in config['synset2related'].items()}
        processor.synset2instance_hypernyms = {int(synset): [int(
            h) for h in hypernyms] for synset, hypernyms in config['synset2instance_hypernyms'].items()}
        processor.synset2instance_hyponyms = {int(synset): [int(
            h) for h in hyponyms] for synset, hyponyms in config['synset2instance_hyponyms'].items()}
        processor.synset2also_see = {int(synset): [int(
            a) for a in also_see] for synset, also_see in config['synset2also_see'].items()}
        processor.synset2pertainyms = {int(synset): [int(
            p) for p in pertainyms] for synset, pertainyms in config['synset2pertainyms'].items()}
        processor.synset2pagerank = {int(synset): [int(
            p) for p in pagerank] for synset, pagerank in config['synset2pagerank'].items()}

        processor.tokenizer = AutoTokenizer.from_pretrained(
            processor.language_model)

        return processor

    @staticmethod
    def _extract_syntagmatic_edges(file_path='data/SYNTAGNET_1.0.txt'):
        with open(file_path, encoding='utf-8', mode='r') as file_:
            lines = file_.read().splitlines()[2:]

        syntag_synsets = {}
        for line in lines:
            synset1, synset2, _, _, _, _ = line.split('\t')
            # synset1, synset2, lemma1, pos1, lemma2, pos2 = line.split('\t')
            if synset1 in syntag_synsets:
                syntag_synsets[synset1].append(synset2)
            else:
                syntag_synsets[synset1] = [synset2]

        return syntag_synsets

    @staticmethod
    def _build_maps(pagerank_path, pagerank_k):
        synset2id = {'<unk>': 0}

        for synset in wn.all_synsets():
            synset = synset.name()
            if synset not in synset2id:
                synset2id[synset] = len(synset2id)

        id2synset = {id: synset for synset, id in synset2id.items()}

        word2synsets = {}

        syntags = Processor._extract_syntagmatic_edges()

        for word in wn.words():
            word2synsets[word] = {}
            for pos in Processor._pos_classes.values():
                word2synsets[word][pos] = []
                for synset in wn.synsets(word, pos=pos):
                    synset_id = synset2id[synset.name()]
                    word2synsets[word][pos].append(synset_id)

        synset2hypernyms = {synset2id['<unk>']: []}
        synset2hyponyms = {synset2id['<unk>']: []}
        synset2similars = {synset2id['<unk>']: []}
        synset2groups = {synset2id['<unk>']: []}
        synset2related = {synset2id['<unk>']: []}
        synset2instance_hypernyms = {synset2id['<unk>']: []}
        synset2instance_hyponyms = {synset2id['<unk>']: []}
        synset2also_see = {synset2id['<unk>']: []}
        synset2pertainyms = {synset2id['<unk>']: []}
        synset2syntags = {synset2id['<unk>']: []}
        for synset in wn.all_synsets():
            synset_id = synset2id[synset.name()]

            synset2syntags[synset_id] = []
            _synset = f"{synset._offset:08}{synset._pos}"
            if _synset in syntags:
                for syntag in syntags[_synset]:
                    syntag_ = wn.of2ss(syntag)
                    syntag_id = synset2id[syntag_.name()]
                    synset2syntags[synset_id].append(syntag_id)

            synset2hypernyms[synset_id] = []
            for hypernym in synset.hypernyms():
                hypernym_id = synset2id[hypernym.name()]
                synset2hypernyms[synset_id].append(hypernym_id)

            synset2hyponyms[synset_id] = []
            for hyponym in synset.hyponyms():
                hyponym_id = synset2id[hyponym.name()]
                synset2hyponyms[synset_id].append(hyponym_id)

            synset2similars[synset_id] = []
            for similar in synset.similar_tos():
                similar_id = synset2id[similar.name()]
                synset2similars[synset_id].append(similar_id)

            synset2groups[synset_id] = []
            for group in synset.verb_groups():
                group_id = synset2id[group.name()]
                synset2groups[synset_id].append(group_id)

            synset2related[synset_id] = []
            synset2pertainyms[synset_id] = []
            for lemma in synset.lemmas():
                for related_lemma in lemma.derivationally_related_forms():
                    related_synset_id = synset2id[related_lemma.synset(
                    ).name()]
                    if related_synset_id not in synset2related[synset_id]:
                        synset2related[synset_id].append(related_synset_id)
                for pertainym in lemma.pertainyms():
                    pertainym_id = synset2id[pertainym.synset().name()]
                    if pertainym_id not in synset2pertainyms[synset_id]:
                        synset2pertainyms[synset_id].append(pertainym_id)

            synset2instance_hypernyms[synset_id] = []
            for instance_hypernym in synset.instance_hypernyms():
                instance_hypernym_id = synset2id[instance_hypernym.name()]
                synset2instance_hypernyms[synset_id].append(
                    instance_hypernym_id)

            synset2instance_hyponyms[synset_id] = []
            for instance_hyponym in synset.instance_hyponyms():
                instance_hyponym_id = synset2id[instance_hyponym.name()]
                synset2instance_hyponyms[synset_id].append(instance_hyponym_id)

            synset2also_see[synset_id] = []
            for also_see in synset.also_sees():
                also_see_id = synset2id[also_see.name()]
                synset2also_see[synset_id].append(also_see_id)

        synset2pagerank = {}
        with open(pagerank_path) as f:
            for line in f:
                synset, *values = line.strip().split()
                synset_id = synset2id[synset]
                synset2pagerank[synset_id] = []
                for value in values:
                    other_synset, _ = value.split('=')
                    if other_synset != synset:
                        other_synset_id = synset2id[other_synset]
                        synset2pagerank[synset_id].append(other_synset_id)
                        if len(synset2pagerank[synset_id]) == pagerank_k:
                            break

        print(f'# synsets: {len(synset2id) - 1}',
              f'# words: {len(word2synsets)}', sep='\n')

        return {
            'synset2id': synset2id,
            'id2synset': id2synset,
            'word2synsets': word2synsets,
            'synset2hypernyms': synset2hypernyms,
            'synset2hyponyms': synset2hyponyms,
            'synset2similars': synset2similars,
            'synset2groups': synset2groups,
            'synset2related': synset2related,
            'synset2instance_hypernyms': synset2instance_hypernyms,
            'synset2instance_hyponyms': synset2instance_hyponyms,
            'synset2also_see': synset2also_see,
            'synset2pertainyms': synset2pertainyms,
            'synset2pagerank': synset2pagerank,
            'synset2syntags': synset2syntags,
        }

    def fetch_synset_name(self, synset):
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

        key = self.synset_offset2sense_key[synset]
        try:
            lemma = wn.lemma_from_key(key)
        except WordNetError as e:
            if key in patching_data:
                lemma = wn.lemma(patching_data[key])
            elif '%3' in key:
                lemma = wn.lemma_from_key(key.replace('%3', '%5'))
            else:
                raise e
        return lemma.synset().name()

    def load_synset_embeddings(self, synset_embeddings_path):
        vectors = {}

        with open(synset_embeddings_path) as f:
            for line in f:
                synset, *vector = line.strip().split()
                # To account for using ares or senesembert + lmms
                synset_ = self.fetch_synset_name(
                    synset) if 'ares' in synset_embeddings_path else synset
                assert synset_ in self.synset2id
                vectors[synset_] = [float(v) for v in vector]

        n_components = len(vector)
        np_vectors = np.zeros((self.num_synsets, n_components))
        for synset, vector in vectors.items():
            synset_id = self.synset2id[synset]
            np_vectors[synset_id] = np.array(vector)

        return torch.as_tensor(np_vectors)

    @staticmethod
    def _build_graph_wrapper(synset2id, synset2similars, synset2groups,
                             synset2related, synset2hypernyms, synset2hyponyms,
                             synset2also_see, synset2pertainyms, synset2syntags,
                             output_path, use_synder):
        if use_synder:
            Processor._build_synder_graphs(synset2id, synset2similars,
                                           synset2groups, synset2related,
                                           synset2hypernyms, synset2hyponyms,
                                           synset2also_see, synset2pertainyms,
                                           synset2syntags, output_path)
        else:
            Processor._build_graph(synset2id, synset2similars, synset2groups,
                                   synset2related, synset2hypernyms,
                                   synset2hyponyms, synset2also_see,
                                   synset2pertainyms, synset2syntags,
                                   output_path)

    @staticmethod
    def _build_graph(synset2id, synset2similars, synset2groups,
                     synset2related, synset2hypernyms, synset2hyponyms,
                     synset2also_see, synset2pertainyms, synset2syntags, output_path):
        synset_indices = [[], []]
        synset_values = []

        for synset in synset2id.values():
            degree = len(synset2groups[synset]) + len(synset2similars[synset]) + \
                len(synset2hypernyms[synset]) + len(synset2hyponyms[synset]) + \
                len(synset2related[synset]) + len(synset2also_see[synset]) + \
                len(synset2pertainyms[synset]) + len(synset2syntags[synset])

            for also_see in synset2also_see[synset]:
                synset_indices[0].append(synset)
                synset_indices[1].append(also_see)
                synset_values.append(1. / degree)

            for pertainym in synset2pertainyms[synset]:
                synset_indices[0].append(synset)
                synset_indices[1].append(pertainym)
                synset_values.append(1. / degree)

            for group in synset2groups[synset]:
                synset_indices[0].append(synset)
                synset_indices[1].append(group)
                synset_values.append(1. / degree)

            for similar in synset2similars[synset]:
                synset_indices[0].append(synset)
                synset_indices[1].append(similar)
                synset_values.append(1. / degree)

            for hypernym in synset2hypernyms[synset]:
                synset_indices[0].append(synset)
                synset_indices[1].append(hypernym)
                synset_values.append(1. / degree)

            for hyponym in synset2hyponyms[synset]:
                synset_indices[0].append(synset)
                synset_indices[1].append(hyponym)
                synset_values.append(1. / degree)

            for syntag in synset2syntags[synset]:
                synset_indices[0].append(synset)
                synset_indices[1].append(syntag)
                synset_values.append(1. / degree)

            for related in synset2related[synset]:
                synset_indices[0].append(synset)
                synset_indices[1].append(related)
                synset_values.append(1. / degree)

        graph = {
            'n': len(synset2id),
            'indices': synset_indices,
            'values': synset_values,
        }

        with open(output_path, 'w') as f:
            json.dump(graph, f)
    
    @staticmethod
    def _build_synder_graphs(synset2id, synset2similars, synset2groups,
                             synset2related, synset2hypernyms, synset2hyponyms,
                             synset2also_see, synset2pertainyms,
                             synset2syntags, output_path):
        synset_indices = [[], []]
        synset_values = []

        synder_indices = [[], []]
        synder_values = []

        for synset in synset2id.values():
            degree = len(synset2groups[synset]) + len(synset2similars[synset]) + \
                len(synset2hypernyms[synset]) + len(synset2hyponyms[synset]) + \
                len(synset2related[synset]) + len(synset2also_see[synset]) + \
                len(synset2pertainyms[synset]) + len(synset2syntags[synset])

            for also_see in synset2also_see[synset]:
                synset_indices[0].append(synset)
                synset_indices[1].append(also_see)
                synset_values.append(1. / degree)

            for pertainym in synset2pertainyms[synset]:
                synset_indices[0].append(synset)
                synset_indices[1].append(pertainym)
                synset_values.append(1. / degree)

            for group in synset2groups[synset]:
                synset_indices[0].append(synset)
                synset_indices[1].append(group)
                synset_values.append(1. / degree)

            for similar in synset2similars[synset]:
                synset_indices[0].append(synset)
                synset_indices[1].append(similar)
                synset_values.append(1. / degree)

            for hypernym in synset2hypernyms[synset]:
                synset_indices[0].append(synset)
                synset_indices[1].append(hypernym)
                synset_values.append(1. / degree)

            for hyponym in synset2hyponyms[synset]:
                synset_indices[0].append(synset)
                synset_indices[1].append(hyponym)
                synset_values.append(1. / degree)

            for syntag in synset2syntags[synset]:
                synder_indices[0].append(synset)
                synder_indices[1].append(syntag)
                synder_values.append(1. / degree)

            for related in synset2related[synset]:
                synder_indices[0].append(synset)
                synder_indices[1].append(related)
                synder_values.append(1. / degree)

        graph = {
            'n': len(synset2id),
            'indices': synset_indices,
            'values': synset_values,
        }

        with open(output_path, 'w') as f:
            json.dump(graph, f)

        synder_graph = {
            'n': len(synset2id),
            'indices': synder_indices,
            'values': synder_values,
        }

        synder_output_path = 'data/synder_graph.json'
        with open(synder_output_path, 'w') as f:
            json.dump(synder_graph, f)

    def _build_graph_offline_ppr(self, offline_pagerank_path, output_path):
        if isfile(f'{offline_pagerank_path}_serialized.json'):
            with open(f'{offline_pagerank_path}_serialized.json', 'r') as f:
                graph = json.load(f)
        else:
            with open(offline_pagerank_path, encoding='utf-8', mode='r') as file_:
                file_content = file_.read().splitlines()

            synset_indices = [[], []]
            synset_weights = []
            for line in tqdm(file_content, desc='Parsing Offline PPR file', leave=False):
                synset1_, synset2_, weight = line.strip().split('\t')
                synset1_idx = self.synset2id[self.fetch_synset_name(synset1_)]
                synset2_idx = self.synset2id[self.fetch_synset_name(synset2_)]
                synset_indices[0].append(synset1_idx)
                synset_indices[1].append(synset2_idx)
                synset_weights.append(float(weight))

            graph = {
                'n': len(self.synset2id),
                'indices': synset_indices,
                'values': synset_weights,
            }

            with open(f'{offline_pagerank_path}_serialized.json', 'w') as f:
                json.dump(graph, f)

        with open(output_path, 'w') as f:
            json.dump(graph, f)
