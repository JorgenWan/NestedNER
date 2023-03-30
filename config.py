import os
import six
import math
import json
import numpy as np
import pickle as pkl

import torch
import torch.nn as nn

from transformers import BertTokenizer

import utils
from models import MODEL_WITH_ARCHI

def pad_sequences(sequences, maxlen=None, dtype='int32',
                  padding='post', truncating='post', value=0.):

    num_samples = len(sequences)

    lengths = []
    for x in sequences:
        try:
            lengths.append(len(x))
        except TypeError:
            raise ValueError('`sequences` must be a list of iterables. '
                             'Found non-iterable: ' + str(x))

    if maxlen is None:
        maxlen = np.max(lengths)

    # take the sample shape from the first non empty sequence
    # checking for consistency in the main loop below.
    sample_shape = tuple()
    for s in sequences:
        if len(s) > 0:
            sample_shape = np.asarray(s).shape[1:]
            break

    is_dtype_str = np.issubdtype(dtype, np.str_) or np.issubdtype(dtype, np.unicode_)
    if isinstance(value, six.string_types) and dtype != object and not is_dtype_str:
        raise ValueError("`dtype` {} is not compatible with `value`'s type: {}\n"
                         "You should set `dtype=object` for variable length strings."
                         .format(dtype, type(value)))

    x = np.full((num_samples, maxlen) + sample_shape, value, dtype=dtype)
    for idx, s in enumerate(sequences):
        if not len(s):
            continue  # empty list/array was found
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError('Truncating type "%s" '
                             'not understood' % truncating)

        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError('Shape of sample %s of sequence at position %s '
                             'is different from expected shape %s' %
                             (trunc.shape[1:], idx, sample_shape))

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError('Padding type "%s" not understood' % padding)
    return x

class PreEmbeddedLM(nn.Module):

    def __init__(self, dev, emb_path, need_cls):
        super().__init__()

        self.dev = dev
        self.need_cls = need_cls

        with open(emb_path, 'rb') as f:
            self.emb_dict = pkl.load(f)

    def forward(self, batch_tokens, maxlen=None):

        embs = []
        cls_embs = None
        if self.need_cls:
            cls_embs = []
        for tokens in batch_tokens:
            lm_emb = self.emb_tokens(tokens)
            # embs.append(lm_emb["word_emb"][:, :1024])
            embs.append(lm_emb["word_emb"][:, :768])
            # embs.append(lm_emb["word_emb"])
            if self.need_cls:
                cls_embs.append(lm_emb["cls_emb"])
        # embs = np.stack(embs, axis=0) # (B, T, H)
        embs_padded = pad_sequences(embs, maxlen=maxlen, dtype='float32',
                                    padding='post', truncating='post', value=0.)
        embs_padded = torch.from_numpy(embs_padded).float()

        mask = torch.zeros(embs_padded.shape[:2]).bool()
        for i, emb in enumerate(embs):
            mask[i, :len(emb)] = True

        embs_padded = embs_padded.to(self.dev)
        mask = mask.to(self.dev)
        if self.need_cls:
            cls_embs = torch.FloatTensor(cls_embs).to(self.dev)

        return embs_padded, mask, cls_embs

    def emb_tokens(self, tokens):
        tokens = tuple(tokens)
        if tokens not in self.emb_dict:
            raise Exception(f'{tokens} not pre-emb')

        return self.emb_dict[tokens]



class Config:

    PAD = "<PAD>"
    BOS = "<BOS>"
    EOS = "<EOS>"
    UNK = "<UNK>"

    def __init__(self, args):

        for k, v in args.__dict__.items():
            setattr(self, k, v)

        if self.debug:
            self.cache_dir = f"{self.data_dir}/cache-debug"
            self.graph_dir = f"{self.data_dir}/Graph-debug"
            self.train_ratio = 0.01
            self.valid_ratio = 0.01
            self.test_ratio = 0.01
        else:
            self.cache_dir = f"{self.data_dir}/cache"
            self.graph_dir = f"{self.data_dir}/Graph"

        self.use_cuda = False
        if torch.cuda.is_available() and not args.use_cpu:
            self.use_cuda = True
            torch.cuda.set_device(0)

        self.dev = torch.device("cuda:0" if self.use_cuda else "cpu")

        self.model_name = self.get_model_name(args.archi)

        self.data_name = args.data_dir.split("/")[-2]

        self.tokenizer = BertTokenizer.from_pretrained(
            args.pretrained_model, do_lower_case=args.do_lower_case
        )

        self.entity_type2idx = utils.build_entity_type2idx(f"{args.data_dir}/entity_types.json", {"None": 0})
        self.idx2entity_type = utils.get_idx2sth([self.entity_type2idx])[0]

        self.special_tokens = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]', '<unk>', '<pad>']

        self.num_nodes = None
        self.graph = None
        self.node_str2idx = None
        self.node2type_idx = None
        self.node_idx2bpe_idx = None

        self.node2char_ids = None
        self.node2word_ids = None
        self.node2lm_spans = None
        self.node2word_strs = None
        self.node2size = None

        if self.use_char_encoder:
            self.idx2char, self.char2idx = [], {}
        if self.use_word_encoder:
            self.word_embeds, self.idx2word, self.word2idx = None, [], {}

        if self.use_lm_embed:
            print("loading pretrained language model embeddings ...")
            # self.lm_embed = PreEmbeddedLM(
            #     dev=self.dev, emb_path=self.lm_emb_path, need_cls=True
            # )
            self.lm_embed = None
            self.ent_lm_embed = PreEmbeddedLM(
                dev=self.dev, emb_path=self.ent_lm_emb_path, need_cls=False
            )
            # exit(0)

        total_train_data_size = len(json.load(open(f"{self.data_dir}/train.json")))
        total_valid_data_size = len(json.load(open(f"{self.data_dir}/valid.json")))
        total_test_data_size = len(json.load(open(f"{self.data_dir}/test.json")))
        self.train_size = math.ceil(total_train_data_size * self.train_ratio)
        self.valid_size = math.ceil(total_valid_data_size * self.valid_ratio)
        self.test_size = math.ceil(total_test_data_size * self.test_ratio)

        # 32 is not ok for ace04
        # self.ent_maxlen = 64

    def load_word_embed(self, file_path, word_dim, special_tokens):

        # Read GloVe vectors
        glove = {}
        with open(file_path, 'r') as f:
            for line in f:
                # values = line.decode().split()
                values = line.strip().split(' ')
                word = values[0]
                glove[word] = np.array([float(x) for x in values[1:]], dtype=np.float32)

        # Build word embedding matrix
        word_embeds = np.zeros((len(glove) + len(special_tokens), word_dim),
                              dtype=np.float)

        idx2word = []
        word2idx = {}

        # Add special tokens and randomly initialize them
        for special_token in special_tokens:
            token_idx = len(idx2word)
            word2idx[special_token] = token_idx
            idx2word.append(special_token)
            word_embeds[token_idx] = np.random.normal(size=word_dim)

        for word in glove.keys():
            word2idx[word] = len(idx2word)
            idx2word.append(word)

        for idx, word in enumerate(idx2word[len(special_tokens):]):
            word_embeds[idx] = np.asarray(glove[word], dtype=np.float)

        self.idx2word, self.word2idx, self.word_embeds = idx2word, word2idx, word_embeds

    def get_model_name(self, archi_name):

        model_name = ""
        for k, v in MODEL_WITH_ARCHI.items():
            if archi_name in v:
                model_name = k
                break

        return model_name

    def reset_data_table(self, label2idx, idx2label, char2idx, idx2char, entity_type2idx, idx2entity_type):
        """
            currently only load label2idx, char2idx
            other dicts: seg2idx, bichar2idx
        """

        self.label2idx = label2idx
        self.idx2label = idx2label
        self.char2idx = char2idx
        self.idx2char = idx2char
        self.entity_type2idx = entity_type2idx
        self.idx2entity_type = idx2entity_type

        # self.seg2idx = seg2idx
        # self.idx2seg = idx2seg
        # self.bichar2idx = bichar2idx
        # self.idx2bichar = idx2bichar

        self.label_size = len(self.idx2label)
        self.num_chars = len(self.idx2char)

        self.pad_idx = self.label2idx[Config.PAD]
        self.bos_idx = self.label2idx[Config.BOS]
        self.eos_idx = self.label2idx[Config.EOS]
        self.unk_idx = self.char2idx[Config.UNK]

        self.O_idx = self.label2idx["O"]

        self.begin_label_ids, self.end_label_ids = utils.get_begin_end_label_ids(self.idx2label)

    def get_num_update_epoch(self, data_size):
        self.max_epoch = self.max_epoch
        self.data_size = data_size
        self.batch_size = self.max_sentences
        self.update_freq = self.update_freq

        self.t_steps = (self.data_size // (self.batch_size * self.update_freq) +
                        int(self.data_size % (self.batch_size * self.update_freq) != 0)) * self.max_epoch


