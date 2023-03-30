import re
import torch
import random
import pickle as pkl
from tqdm import tqdm
from collections import OrderedDict

from config import Config


class Entity:

    def __init__(
        self, entity_type, entity_type_id, chars, char_ids,
        span_start, span_end
    ):

        """
        entity_type: str
        tokens: List[Str]

        entity_type_id: int
        token_ids: List[Int]

        """

        self.entity_type = entity_type
        self.chars = chars

        self.entity_type_id = entity_type_id
        self.char_ids = char_ids

        self.span_start = span_start
        self.span_end = span_end
        self.span = (span_start, span_end)

    @property
    def span_start_end_typeid(self):
        return (int(self.span_start), int(self.span_end), int(self.entity_type_id))

class Document:

    def __init__(
        self, chars, labels, char_ids, label_ids, entity_type2idx,
        segs=None, bichars=None, seg_ids=None, bichar_ids=None, # may not use them, just my settings
    ):
        self.chars = chars
        self.labels = labels
        self.char_ids = char_ids
        self.label_ids = label_ids

        self.segs = segs
        self.bichars = bichars
        self.seg_ids = seg_ids
        self.bichar_ids = bichar_ids

        assert len(chars) == len(labels) == len(segs) == len(bichars) == len(self)
        assert len(char_ids) == len(label_ids) == len(seg_ids) == len(bichar_ids) == len(self)

        self.entity_type2idx = entity_type2idx

        self.entities = self.get_entities()

    def __len__(self):
        return len(self.char_ids)

    def get_entities(self):
        entities = []

        span_start, span_end = 0, 0
        for i in range(len(self.labels)):
            label = self.labels[i]
            if label.startswith('O') or label.startswith('I'):
                continue
            elif label.startswith('S'):
                span_start = i
                span_end = i
                entities.append(
                    Entity(
                        entity_type=label[2:],
                        entity_type_id=self.entity_type2idx[label[2:]],
                        chars=self.chars[span_start:span_end+1],
                        char_ids=self.char_ids[span_start:span_end+1],
                        span_start=span_start,
                        span_end=span_end
                    )
                )
            elif label.startswith('B'):
                span_start = i
            elif label.startswith('E'):
                span_end = i
                entities.append(
                    Entity(
                        entity_type=label[2:],
                        entity_type_id=self.entity_type2idx[label[2:]],
                        chars=self.chars[span_start:span_end + 1],
                        char_ids=self.char_ids[span_start:span_end + 1],
                        span_start=span_start,
                        span_end=span_end
                    )
                )

        return entities

class NER_Dataset(torch.utils.data.Dataset.TorchDataset):

    Train_Mode = "train"
    Eval_Mode = "eval"

    def __init__(self, neg_entity_count, max_span_size):

        self._mode = NER_Dataset.Train_Mode

        self._neg_entity_count = neg_entity_count
        self._max_span_size = max_span_size

        self._documents = OrderedDict()
        self._entities = OrderedDict()

        # current ids
        self._doc_id = 0
        self._ent_id = 0
        self._tok_id = 0

    def create_document(self, tokens, entities, doc_encoding):
        document = Document(self._doc_id, tokens, entities, doc_encoding)
        self._documents[self._doc_id] = document
        self._doc_id += 1

        return document

    def create_entity(self, entity_type, tokens, phrase) -> Entity:
        entity = Entity(self._ent_id, entity_type, tokens, phrase)
        self._entities[self._ent_id] = entity
        self._ent_id += 1

        return entity

    def __len__(self):
        return len(self._documents)

    def switch_mode(self, mode):
        self._mode = mode




class Instances:

    pad = Config.PAD
    bos = Config.BOS
    eos = Config.EOS
    unk = Config.UNK

    def __init__(self, inst_list):
        self.inst_list = inst_list

    def __len__(self):
        return len(self.inst_list)

    def __getitem__(self, item):
        """return a tuple: (char label seg bichar)"""
        return self.inst_list[item]

    @staticmethod
    def _load_bin(file, max_length, entity_type2idx):
        with open(file, "rb") as f:
            data = pkl.load(f)

        num_skip = 0
        inst_list = []
        for chars, labels, segs, bichars, \
            char_ids, label_ids, seg_ids, bichar_ids in data:
            if len(char_ids) <= 0:
                num_skip += 1
                continue
            if max_length < 0 or len(char_ids) < max_length:
                inst_list.append(
                    Instance(
                        chars, labels, segs, bichars,
                        char_ids, label_ids, seg_ids, bichar_ids,
                        entity_type2idx
                    )
                )
            else:
                num_skip += 1
        print(f"Loading {file}, max length: {max_length}, skip {num_skip} sentences.")
        return inst_list

    @staticmethod
    def load_data(data_dir, max_length, entity_type2idx):
        """
        Data requirement:
            1. BIOES label mode
            2. length is not so long
            3. utf-8 encoding
        Data type:
            1. *.pkl file
            2. List[Tuple]: [(char_ids, label_ids, seg_ids, bichar_ids), ...]
        """

        # load binary
        train_inst_list = Instances._load_bin(f"{data_dir}/train.pkl", max_length, entity_type2idx)
        valid_inst_list = Instances._load_bin(f"{data_dir}/valid.pkl", max_length, entity_type2idx)
        test_inst_list = Instances._load_bin(f"{data_dir}/test.pkl", max_length, entity_type2idx)

        # creat objects
        train_insts = Instances(train_inst_list)
        valid_insts = Instances(valid_inst_list)
        test_insts = Instances(test_inst_list)

        print(f"Sentences: train={len(train_insts)}, valid={len(valid_insts)}, test={len(test_insts)}")

        return train_insts, valid_insts, test_insts

    def remove_middle(self, idx2label, label2idx):

        for inst in self.inst_list:
            label_ids = inst.label_ids
            seq_len = len(label_ids)
            for i in range(seq_len):
                cur_label = idx2label[label_ids[i]]
                if cur_label.startswith('I'):
                    inst.label_ids[i] = label2idx['O']

    def shuffle(self):
        random.shuffle(self.inst_list)

    def batch_to_tensors(self, batch_size, use_seg, use_bichar):

        num_batch = (len(self) // batch_size)
        assert num_batch >= 1
        num_batch += 1 if len(self) % batch_size != 0 else 0

        batches = []
        for i in range(num_batch):
            batch_inst = self.inst_list[i * batch_size: (i+1) * batch_size]
            batch_tensor = self._map_inst_to_tensor(batch_inst, use_seg, use_bichar)
            batches.append(batch_tensor)
        return batches

    def _map_inst_to_tensor(self, inst_list, use_seg, use_bichar):
        batch_size = len(inst_list)

        char_lengths = torch.LongTensor([len(inst) for inst in inst_list])
        seq_len = char_lengths.max()

        chars = torch.zeros([batch_size, seq_len], dtype=torch.long)
        labels = torch.zeros([batch_size, seq_len], dtype=torch.long)
        segs = torch.zeros([batch_size, seq_len], dtype=torch.long) if use_seg else None
        bichars = torch.zeros([batch_size, seq_len], dtype=torch.long) if use_bichar else None
        for i in range(batch_size):
            chars[i, :char_lengths[i]] = torch.LongTensor(inst_list[i].char_ids)
            labels[i, :char_lengths[i]] = torch.LongTensor(inst_list[i].label_ids)
            if use_seg:
                segs[i, :char_lengths[i]] = torch.LongTensor(inst_list[i].seg_ids)
            if use_bichar:
                bichars[i, :char_lengths[i]] = torch.LongTensor(inst_list[i].bichar_ids)

        num_sentences = batch_size
        num_tokens = torch.sum(char_lengths).data.item()

        gold_spans = [[] for _ in range(len(inst_list))]
        for i in range(len(inst_list)):
            inst = inst_list[i]
            for e in inst.entities:
                gold_spans[i].append(e.span_start_end_typeid)

        result_dict = {
            "chars": chars,
            "labels": labels,
            "segs": segs,
            "bichars": bichars,
            "char_lengths": char_lengths,
            "num_sentences": num_sentences,
            "num_tokens": num_tokens,
            "inst_list": inst_list,
            "gold_spans": gold_spans
        }

        return result_dict