import math
import json
import torch

from tqdm import tqdm
from collections import OrderedDict

from data.document import Token, Entity, Document
from data import sampling

class Batch_Iterator:
    def __init__(self, objects, batch_size, order=None, truncate=False):
        self._objects = objects
        self._batch_size = batch_size
        self._truncate = truncate
        self._length = len(self._objects)
        self._order = order

        if order is None:
            self._order = list(range(len(self._entities)))

        self._i = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self._truncate and self._i + self._batch_size > self._length:
            raise StopIteration
        elif not self._truncate and self._i >= self._length:
            raise StopIteration
        else:
            objects = [self._objects[i] for i in self._order[self._i:self._i + self._batch_size]]
            self._i += self._batch_size
            return objects


class NER_Dataset(torch.utils.data.Dataset):

    Train_Mode = "train"
    Eval_Mode = "eval"
    Third_Mode = "third" # for building node embeds

    @staticmethod
    def read_data_from_json(
        cfg, json_file_path,
        dataset_label, entity_type2idx,
        tokenizer, span_cands, node_cands,
        node_str2idx, idx2entity_type,
        graph, node2type_idx
    ):
        dataset = NER_Dataset(
            cfg, dataset_label, entity_type2idx,
            tokenizer, span_cands, node_cands,
            node_str2idx, idx2entity_type,
            graph, node2type_idx
        )
        documents = json.load(open(json_file_path))
        if dataset_label == "train":
            documents = documents[:cfg.train_size]
        if dataset_label == "valid":
            documents = documents[:cfg.valid_size]
        if dataset_label == "test":
            documents = documents[:cfg.test_size]
        for document in tqdm(documents, desc=f"Parse dataset '{dataset_label}'"):
            dataset.add_document(document)

        return dataset

    def __init__(
        self, cfg, label, entity_type2idx,
        tokenizer, span_cands, node_cands,
        node_str2idx, idx2entity_type, graph, node2type_idx
    ):
        self.cfg = cfg
        self._label = label
        self._entity_type2idx = entity_type2idx
        self._tokenizer = tokenizer
        self._graph = graph
        self._node2type_idx = node2type_idx

        self._span_cands = span_cands
        self._node_cands = node_cands
        self._node_str2idx = node_str2idx
        self._idx2entity_type = idx2entity_type

        self._documents = OrderedDict()
        self._entities = OrderedDict()

        # current ids
        self._doc_id = 0
        self._ent_id = 0
        self._tok_id = 0

        # default mode
        self._mode = NER_Dataset.Train_Mode

        self._cased_char = cfg.cased_char
        self._cased_word = cfg.cased_word
        self._use_gcn = cfg.use_gcn
        self._use_char_encoder = cfg.use_char_encoder
        self._node_idx2bpe_idx = None

    def set_char_ids(self, char2idx):
        total_n_ids, total_n_unk_ids = 0, 0
        for doc in self._documents.values():
            n_ids, n_unk_ids = doc.set_char_ids(char2idx=char2idx)
            total_n_ids += n_ids
            total_n_unk_ids += n_unk_ids
        return total_n_ids, total_n_unk_ids

    def set_word_ids(self, word2idx):
        total_n_ids, total_n_unk_ids = 0, 0
        for doc in self._documents.values():
            n_ids, n_unk_ids = doc.set_word_ids(word2idx=word2idx)
            total_n_ids += n_ids
            total_n_unk_ids += n_unk_ids
        return total_n_ids, total_n_unk_ids

    def create_entity(self, entity_type_id, tokens, phrase, start, end):
        entity = Entity(self._ent_id, entity_type_id, tokens, phrase, start, end)
        self._entities[self._ent_id] = entity
        self._ent_id += 1

        return entity

    def create_token(self, idx, span_start, span_end, phrase):
        """
        idx (Int): token index in a document
        phrase (Str): token string
        """
        token = Token(self._tok_id, idx, span_start, span_end, phrase)
        self._tok_id += 1
        return token

    def __len__(self):
        return len(self._documents)

    def __getitem__(self, index):
        doc = self._documents[index]
        span_cands = None
        if self._use_gcn:
            span_cands = self._span_cands[index] # candidates for each instance

        if self._mode == NER_Dataset.Train_Mode:
            return sampling.create_sample(
                self.cfg, doc, span_cands, self._node_cands, self._graph,
                self._node2type_idx, self._node_idx2bpe_idx, sample_type="train",
            )
        elif self._mode == NER_Dataset.Eval_Mode:
            return sampling.create_sample(
                self.cfg, doc, span_cands, self._node_cands, self._graph,
                self._node2type_idx, self._node_idx2bpe_idx, sample_type="eval",
            )
        elif self._mode == NER_Dataset.Third_Mode:
            return sampling.create_third_sample(
                self.cfg, doc, self._node_str2idx, self._idx2entity_type
            )
        else:
            raise NotImplementedError

    def build_node_idx2bpe_idx(self, node_str2idx, idx2entity_type):

        node_idx2bpe_idx = dict()
        for doc in self._documents.values():
            bpe_ids = doc.bpe_ids
            for e in doc.entities:
                node_str = e.phrase + " " + idx2entity_type[e.entity_type_id]
                node_idx = node_str2idx[node_str]
                start, end = e.span
                start += 1 # for [CLS]
                end += 1 # for [CLS]
                node_idx2bpe_idx[node_idx] = bpe_ids[start:end+1]

        return node_idx2bpe_idx

    def set_node_idx2bpe_idx(self, node_idx2bpe_idx):
        self._node_idx2bpe_idx = node_idx2bpe_idx

    def build_char2idx(self, special_tokens=[]):

        idx2char = []

        # Add special tokens
        for special_token in special_tokens:
            idx2char.append(special_token)

        new_idx2char = []
        for doc in self._documents.values():
            for each_chars in doc.chars:
                new_idx2char += each_chars
        idx2char = idx2char + list(set(new_idx2char))

        char2idx = {x: i for i, x in enumerate(idx2char)}

        return char2idx, idx2char

    def switch_mode(self, mode):
        self._mode = mode

    def add_document(self, raw_document):
        """
        Input:
            raw_document: Dict[]
        """

        raw_words = raw_document['tokens']
        raw_entities = raw_document['entities']

        # parse raw words
        bpe_tokens, bpe_ids, bpe_strs = self._parse_tokens(raw_words)
        chars = None
        if self._use_char_encoder:
            chars = self._get_chars(raw_words, self._cased_char)
        words = self._get_words(raw_words, self._cased_word)

        # parse raw entities
        entities = self._parse_entities(raw_entities, bpe_tokens)

        # create document
        document = Document(
            self._doc_id, bpe_tokens, entities, bpe_ids, bpe_strs, chars, words
        )
        self._documents[self._doc_id] = document
        self._doc_id += 1

        return document

    def _parse_tokens(self, raw_words):
        """
        Input:
            raw_words: List[Str]
        Output:

        """

        # encoding without special tokens ([CLS] and [SEP])
        bpe_tokens = []

        bpe_ids = []
        # encoding with special tokens ([CLS] and [SEP]) and BPEs of original tokens
        # bpe_ids += [self._tokenizer.convert_tokens_to_ids('[CLS]')]

        # parse tokens
        for token_idx, token_phrase in enumerate(raw_words):
            cur_bpe_ids = self._tokenizer.encode(token_phrase, add_special_tokens=False)
            len_cur_bpe_ids = len(cur_bpe_ids)
            if len(cur_bpe_ids) == 0: # "" token exists
                len_cur_bpe_ids = 1
                # print(len(cur_bpe_ids))
                # print(cur_bpe_ids)
                # print(token_phrase)
                # print(raw_words)
                # exit(0)
            assert len_cur_bpe_ids > 0
            span_start, span_end = (len(bpe_ids), len(bpe_ids) + len_cur_bpe_ids - 1) # inclusive

            bpe_token = self.create_token(token_idx, span_start, span_end, token_phrase)

            bpe_tokens.append(bpe_token)
            bpe_ids += cur_bpe_ids

        bpe_strs = self._tokenizer.convert_ids_to_tokens(bpe_ids)

        cls_id = self._tokenizer.convert_tokens_to_ids('[CLS]')
        sep_id = self._tokenizer.convert_tokens_to_ids('[SEP]')
        bpe_ids = [cls_id] + bpe_ids + [sep_id]

        return bpe_tokens, bpe_ids, bpe_strs

    def _parse_entities(self, raw_entities, bpe_tokens):
        entities = []

        for entity_idx, jentity in enumerate(raw_entities):
            start, end = jentity['start'], jentity['end'] # inclusive
            entity_type_id = self._entity_type2idx[jentity['type']]

            # create entity mention
            tokens = bpe_tokens[start:end+1]
            phrase = " ".join([t.phrase for t in tokens])

            entity = self.create_entity(
                entity_type_id, tokens, phrase, start, end
            )
            entities.append(entity)

        return entities

    def _get_chars(self, raw_words, cased_char):
        if cased_char:
            words = raw_words
        else:
            words = [w.lower() for w in raw_words]
        chars = [[c for c in w] for w in words]
        return chars

    def _get_words(self, raw_words, cased_word):
        if cased_word:
            words = raw_words
        else:
            words = [w.lower() for w in raw_words]
        return words

    @property
    def label(self):
        return self._label

    @property
    def documents(self):
        return list(self._documents.values())

    @property
    def entities(self):
        return list(self._entities.values())

    @property
    def document_count(self):
        return len(self._documents)

    @property
    def entity_count(self):
        return len(self._entities)
