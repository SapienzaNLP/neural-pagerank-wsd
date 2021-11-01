import json

from torch.utils.data import Dataset


class WordSenseDisambiguationDataset(Dataset):

    def __init__(self, path_to_data):
        super().__init__()
        self.sentences = WordSenseDisambiguationDataset.load_sentences(path_to_data)

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        return sentence

    @staticmethod
    def load_sentences(path):
        sentences = {}
        with open(path) as json_file:
            sentence_index = 0
            for i, sentence in json.load(json_file).items():
                if not sentence['instance_ids']:
                    continue
                words, lemmas, pos_tags = WordSenseDisambiguationDataset._get_clean_sentence(sentence['words'], sentence['lemmas'], sentence['pos_tags'])
                sentences[sentence_index] = {
                    'sentence_id': i,
                    'words': words,
                    'lemmas': lemmas,
                    'pos_tags': pos_tags,
                    'instance_ids': {
                        int(instance_index): instance_id for instance_index, instance_id in sentence['instance_ids'].items()
                    },
                    'senses': {
                        int(word_index): senses for word_index, senses in sentence['senses'].items()
                    },
                }
                sentence_index += 1
        return sentences

    @staticmethod
    def _get_clean_sentence(words, lemmas, pos_tags):
        clean_words, clean_lemmas, clean_pos_tags = [], [], []
        for i in range(len(words)):
            if not ((i < len(words) - 2 and words[i] == '@' and words[i + 2] == '@') or (words[i] == '@' and words[i - 2] == '@')):
                clean_words.append(words[i])
                clean_lemmas.append(lemmas[i])
                clean_pos_tags.append(pos_tags[i])

        return clean_words, clean_lemmas, clean_pos_tags
