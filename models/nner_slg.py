import os
import json
import logging
import itertools
import numpy as np

from time import time

import dgl
from dgl.nn.pytorch import GraphConv

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from transformers import BertModel

from models import register_model, register_model_architecture
from models.utils import calculate_ner_match, calculate_prf, is_nan
from models.base_model import Base_Model

from modules.utils import Cross_Entropy, Embedding
from modules.graph_conv import Graph_Conv_v2

logger = logging.getLogger(__name__)

def compute_kernel_bias(vecs, n_comp):
    """
        function: y = (x + bias).dot(kernel)
        vecs: 2d np.array
        n_comp: Int
    """
    mu = vecs.mean(axis=0, keepdims=True)
    cov = np.cov(vecs.T)
    u, s, vh = np.linalg.svd(cov)
    W = np.dot(u, np.diag(1 / np.sqrt(s)))
    return W[:, :n_comp], -mu

def transform_and_normalize(vecs, kernel=None, bias=None):
    """
        vecs: 2d np.array, shape = (n1, n2)
        kernel: 2d np.array, shape = (n2, n_comp)
        bias: 2d np.array, shape = (n1, n_comp)
    """
    if not (kernel is None or bias is None):
        vecs = (vecs + bias).dot(kernel)
    return vecs / (vecs**2).sum(axis=1, keepdims=True)**0.5

def whitening(arr, n_comp):
    """
        arr: 2d np.array, shape = (n1, n2)
        n_comp: Int
        new_arr: 2d np.array, shape = (n1, n_comp)
    """

    kernel, bias = compute_kernel_bias(arr, n_comp=n_comp)
    new_arr = transform_and_normalize(arr, kernel=kernel, bias=bias)
    return new_arr

def flat1(ll):
    l = list(itertools.chain.from_iterable(ll))
    return l


def flat2(data, tensor=False, dev="cpu"):
    res = [d for dd in data for d in dd]
    if tensor:
        res = torch.stack(res).to(dev)
    return res

def sim_matrix(a, b, eps=1e-8):
    """
        a: matrix
        b: matrix
        eps: for numerical stability
    """
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt

def accumu_sum(l):
    """
        l: List, eg [None, 2, 3, 1]
        new_l: List, eg [0, 2, 5, 6]
    """
    cur_sum = 0
    new_l = []
    for n in l:
        if n:
            cur_sum += int(n)
        new_l.append(cur_sum)
    return new_l

def get_attn_mask(ll):
    """
        ll: List[List[Int]]
    """
    _lens = [None] + [len(l) for l in ll]
    _idx = accumu_sum(_lens)

    dim1 = len(ll)
    dim2 = sum([len(l) for l in ll])
    attn_mask = torch.zeros(dim1, dim2, dtype=torch.float)
    for i, l in enumerate(ll):
        attn_mask[i, _idx[i]:_idx[i+1]] = 1
    return attn_mask

def get_token(h: torch.tensor, x: torch.tensor, token: int):
    """ Get specific token embedding (e.g. [CLS]) """
    emb_size = h.shape[-1]

    token_h = h.view(-1, emb_size)
    flat = x.contiguous().view(-1)

    # get contextualized embedding of given token
    token_h = token_h[flat == token, :]

    return token_h

class DGL_GCN(nn.Module):
    def __init__(self, input_dim, layers, dropout, use_graph_weight, gcn_norm_method):
        super().__init__()

        self.input_dim = input_dim
        self.gcn_layers = layers
        self.gcn_dropout = dropout
        self.use_graph_weight = use_graph_weight
        self.gcn_norm_method = gcn_norm_method

        self.setup_layers()

    @property
    def out_dim(self):
        return self.gcn_layers[-1]

    def setup_layers(self):
        """
        Creating the layers based on the args.
        """
        self.layers = []
        gcn_layers = [self.input_dim] + self.gcn_layers
        for i, _ in enumerate(gcn_layers[:-1]):
            if self.use_graph_weight:
                self.layers.append(
                    Graph_Conv_v2(
                        in_feats=gcn_layers[i],
                        out_feats=gcn_layers[i+1],
                        activation=F.relu,
                        allow_zero_in_degree=True,
                        weight=True,
                        norm=self.gcn_norm_method
                    )
                ) # default: edge['w'] to store graph weight
            else:
                self.layers.append(
                    GraphConv(
                        in_feats=gcn_layers[i],
                        out_feats=gcn_layers[i + 1],
                        activation=F.relu,
                        weight=True,
                        norm=self.gcn_norm_method
                    )
                )

            """
            dgl._ffi.base.DGLError: 
                There are 0-in-degree nodes in the graph, 
                output for those nodes will be invalid. 
                This is harmful for some applications, 
                causing silent performance regression. 
                Adding self-loop on the input graph by calling 
                `g = dgl.add_self_loop(g)` will resolve the issue. 
                Setting ``allow_zero_in_degree`` to be `True` 
                when constructing this module will suppress the check 
                and let the code run.
            """
        self.layers = ListModule(*self.layers)

    def forward(self, g, features, readout_weights=None):
        """
        Args:
            g: batched_graph
            features: FloatTensor(real_num_nodes, feature_dim)
            readout_weights: FloatTensor(real_num_nodes)
        """
        # g = dgl.add_self_loop(g) # todo: self-loop?

        h = features.float()
        for i, layer in enumerate(self.layers):
            h = F.dropout(h, p=self.gcn_dropout, training=self.training)
            h = layer(g, h)
        h = F.dropout(h, p=self.gcn_dropout, training=self.training)

        return h

class ListModule(torch.nn.Module):
    """
        Abstract list layer class.
    """
    def __init__(self, *args):
        """
            Module initializing.
        """
        super(ListModule, self).__init__()
        idx = 0
        for module in args:
            self.add_module(str(idx), module)
            idx += 1

    def __getitem__(self, idx):
        """
            Getting the indexed layer.
        """
        if idx < 0 or idx >= len(self._modules):
            raise IndexError('index {} is out of range'.format(idx))
        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def __iter__(self):
        """
            Iterating on the layers.
        """
        return iter(self._modules.values())

    def __len__(self):
        """
            Number of layers.
        """
        return len(self._modules)


class Char_Encoder(nn.Module):

    def __init__(self, n_chars, char_dim=60, char_hid_dim=100, device=None):
        super().__init__()
        self.dev = device

        self.char_embeds = nn.Embedding(num_embeddings=n_chars, embedding_dim=char_dim)
        self.lstm = nn.LSTM(input_size=char_dim, hidden_size=char_hid_dim,
                            bidirectional=True, batch_first=True).to(device=self.dev)

    def forward(self, char_ids):
        """
        Input:
            char_ids: B * max_n_word * max_char_len
        Output:
            char_hids: B * max_n_word, 2 * char_hid_dim
        """

        # assert char_ids is not None
        mask = (char_ids != 0)
        lens = mask.sum(-1) # B * max_n_word

        x = self.char_embeds(char_ids)
        bsz, w_maxlen, c_maxlen, _ = x.shape
        x = x.view(bsz * w_maxlen, c_maxlen, -1)
        lens = lens.view(bsz * w_maxlen)

        word_seq_lens = lens + (lens == 0).long()  # avoid length == 0
        word_rep = x
        sorted_seq_len, permIdx = word_seq_lens.sort(0, descending=True)
        _, recover_idx = permIdx.sort(0, descending=False)
        sorted_seq_tensor = word_rep[permIdx]

        packed_words = pack_padded_sequence(sorted_seq_tensor, sorted_seq_len.cpu(), True)
        lstm_out, (h, _) = self.lstm(packed_words, None)
        # lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)
        # outputs = lstm_out[recover_idx]
        hidden = torch.cat([h[-2, :, :], h[-1, :, :]], dim=-1)
        hidden = hidden[recover_idx]
        out = hidden.view(bsz, w_maxlen, -1)

        # char_hids = []
        # for seq in x:
        #     _, (h, _) = self.lstm(seq) # 2, max_n_word, H
        #     x_hid = torch.cat([h[-2, :, :], h[-1, :, :]], dim=-1) # max_n_word, 2 * H
        #     char_hids.append(x_hid)
        #
        # out = torch.stack(char_hids).to(device=self.dev) # max_n_word, 2 * H

        return out

def word_lstm_forward(lstm, x, mask):
    lens = mask.sum(-1)  # B

    bsz, w_maxlen, hid_dim = x.shape

    word_seq_lens = lens + (lens == 0).long()  # avoid length == 0
    word_rep = x
    sorted_seq_len, permIdx = word_seq_lens.sort(0, descending=True)
    _, recover_idx = permIdx.sort(0, descending=False)
    sorted_seq_tensor = word_rep[permIdx]

    packed_words = pack_padded_sequence(sorted_seq_tensor, sorted_seq_len.cpu(), True)
    lstm_out, (h, _) = lstm(packed_words, None)
    lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)
    outs = lstm_out[recover_idx]
    # hidden = torch.cat([h[-2, :, :], h[-1, :, :]], dim=-1)
    # out = hidden[recover_idx]

    # out = torch.zeros(
    #     [bsz, w_maxlen, outs.shape[2]],
    #     requires_grad=True, dtype=torch.float
    # ).to(outs.device)  # B, w_maxlen, hid_dim
    # out[:, :outs.shape[1], :] = outs

    outs = torch.cat(
        [
            outs,
            torch.zeros([bsz, w_maxlen-outs.shape[1], outs.shape[2]]).to(outs.device)
        ],
        dim=1
    )

    return outs

@register_model("nner_slg")
class NNER_SLG(Base_Model):

    @classmethod
    def build_model(cls, cfg):
        return cls(cfg)

    def __init__(self, cfg):
        super().__init__(cfg)

        self.hidden_dim = cfg.lm_dim
        self.size_embed_dim = cfg.size_embed_dim
        self.neg_entity_count = cfg.neg_entity_count
        self.max_span_size = cfg.max_span_size
        self.entity_type2idx = cfg.entity_type2idx
        self.idx2entity_type = cfg.idx2entity_type
        self.entity_type_size = len(cfg.idx2entity_type)
        self._max_batch_nodes = cfg.max_batch_nodes
        self._max_batch_edges = cfg.max_batch_edges
        self._graph = cfg.graph # entity graph
        self._n_neighbor = cfg.n_neighbor
        self._n_hop = cfg.n_hop
        self.num_nodes = cfg.num_nodes
        self.gcn_layers = cfg.gcn_layers

        self.gcn_dropout = cfg.gcn_dropout
        self.concat_span_hid = cfg.concat_span_hid
        self.use_graph_weight = cfg.use_graph_weight
        self.do_extra_eval = cfg.do_extra_eval
        self.save_dir = cfg.save_dir
        self.idx2entity_type = cfg.idx2entity_type
        self.alpha = cfg.alpha
        self.gcn_norm_method = cfg.gcn_norm_method

        self.dev = cfg.dev
        self.use_char_encoder = cfg.use_char_encoder
        self.use_word_encoder = cfg.use_word_encoder
        self.use_lm_embed = cfg.use_lm_embed
        self.use_size = cfg.use_size
        self.char_hid_dim = cfg.char_hid_dim
        self.word_embed_dim = cfg.word_embed_dim
        self.wc_lstm_hid_dim = cfg.wc_lstm_hid_dim
        self.sent_enc_dropout = cfg.sent_enc_dropout
        self.node_embed_dim = self.wc_lstm_hid_dim * 2 + int(self.use_size) * self.size_embed_dim
        self.gcn_init_dim = self.wc_lstm_hid_dim * 2 + int(self.use_size) * self.size_embed_dim

        self.max_ent_n_word = cfg.max_ent_n_word
        self.max_n_word = cfg.max_n_word
        self.show_infer_speed = cfg.show_infer_speed
        self.test_n_words = cfg.test_n_words

        self.graph_emb_method = cfg.graph_emb_method
        self.wc_lstm_layers = cfg.wc_lstm_layers

        if self.use_char_encoder:
            self.node2char_ids = cfg.node2char_ids
            self.char_encoder = Char_Encoder(
                n_chars=len(cfg.idx2char), char_dim=cfg.char_embed_dim,
                char_hid_dim=cfg.char_hid_dim, device=cfg.dev
            )

        if self.use_word_encoder:
            self.node2word_ids = cfg.node2word_ids
            word_embeds = torch.FloatTensor(cfg.word_embeds)
            self.word_embeds = nn.Embedding.from_pretrained(word_embeds, freeze=True)

        self.use_gcn = cfg.use_gcn
        if self.use_lm_embed:
            self.cls_embed_dim = cfg.cls_embed_dim
            self.lm_embed = cfg.lm_embed
            self.node2lm_spans = cfg.node2lm_spans
            self.node2word_strs = cfg.node2word_strs
            self.ent_lm_embed = cfg.ent_lm_embed
            self.use_cls = cfg.use_cls
            self.lm_dim = cfg.lm_dim


            self.lm = BertModel.from_pretrained(
                cfg.pretrained_model, output_attentions=True, output_hidden_states=False
            )
        else:
            self.use_cls = False
            self.lm_dim = 0

        if self.use_char_encoder or self.use_word_encoder:
            # enc_hid_dim = cfg.char_hid_dim * int(self.use_char_encoder) * 2 + \
            #             cfg.word_embed_dim * int(self.use_word_encoder) # method 1
            enc_hid_dim = cfg.char_hid_dim * int(self.use_char_encoder) * 2 + \
                          cfg.word_embed_dim * int(self.use_word_encoder) + \
                          self.lm_dim * int(self.use_lm_embed) # method 2
        else:
            if self.use_lm_embed:
                enc_hid_dim = self.lm_dim  # method 2
        self.wc_lstm_enc = nn.LSTM(
            input_size=enc_hid_dim, hidden_size=self.wc_lstm_hid_dim,
            bidirectional=True, batch_first=True, num_layers=self.wc_lstm_layers
        ).to(device=self.dev)
        self.wc_enc_dropout = nn.Dropout(self.sent_enc_dropout).to(device=self.dev)

        # self.lin_hid_size = int(self.use_lm_embed) * self.lm_dim + self.wc_lstm_hid_dim\
        #                     * \
        #                   int(self.use_char_encoder or self.use_word_encoder) * 2
        # self.lin_hid_size = self.wc_lstm_hid_dim \
        #                         * int(self.use_char_encoder or self.use_word_encoder) * 2
        if self.use_char_encoder or self.use_word_encoder:
            self.lin_hid_size = self.wc_lstm_hid_dim * 2 + int(self.use_cls) * self.lm_dim + int(self.use_size) * self.size_embed_dim
        else:
            self.lin_hid_size = self.lm_dim + int(self.use_cls) * self.lm_dim + int(self.use_size) * self.size_embed_dim
        # self.lin_hid_size = self.lm_dim + int(self.use_cls) * self.lm_dim + + int(self.use_size) * self.size_embed_dim
        if self.lin_hid_size > 0:
            # self.encoder_linear = nn.Linear(self.lin_hid_size, self.wc_lstm_hid_dim * 2)
            self.encoder_relu = nn.Tanh()
        else:
            self.encoder_linear = None

        if self.use_size:
            self.size_embeds = nn.Embedding(100, self.size_embed_dim)

        self.node_out_dim = 0
        if self.use_gcn:
            self.gcn = DGL_GCN(
                input_dim=self.gcn_init_dim,
                layers=self.gcn_layers,
                dropout=self.gcn_dropout,
                use_graph_weight=self.use_graph_weight,
                gcn_norm_method=self.gcn_norm_method,
            ) # B n_span (gcn_in_dim+gcn_out_dim)*2+H_ctx+size_emb
            self.node_out_dim = self.gcn.out_dim # gcn

        if not self.use_cls:
            self.span_out_dim = int(self.use_size) * self.size_embed_dim
        else:
            self.span_out_dim = int(self.use_size) * self.size_embed_dim + self.cls_embed_dim
        if self.use_gcn:
            self.span_out_dim += self.node_out_dim + self.gcn.out_dim
        if self.concat_span_hid:
            self.span_out_dim += (2 * self.wc_lstm_hid_dim) # span-level hidden
        self.span_dropout = nn.Dropout(self.dropout).to(device=self.dev)
        # self.fc_out_span = nn.Linear(self.span_out_dim, self.entity_type_size)
        # self.fc_out_span = nn.Linear(
        #     self.lin_hid_size +
        #     int(self.use_size) * self.size_embed_dim +
        #     int(self.use_cls) * self.cls_embed_dim,
        #     self.entity_type_size
        # )
        self.fc_out_span = nn.Linear(
            self.lin_hid_size + int(self.use_gcn) * (self.node_out_dim),
            self.entity_type_size
        )
        if self.use_gcn:
            self.fc_out_node = nn.Linear(self.node_out_dim, self.entity_type_size)

            if self.graph_emb_method == "attn":
                self.W_graph_attn = nn.Linear(self.wc_lstm_hid_dim * 2 + int(self.use_size) * self.size_embed_dim, self.node_out_dim)

    def _encode_to_span_level(self, h_word_level, entity_masks):
        """
        Input:
            h_word_level: B * max_n_word * H
            entity_masks: B * max_n_span * max_n_word
        Output:
            h_span_level: B * max_n_span * H
        """

        bsz, n_span, seq_len = entity_masks.size()

        # if n_span is large (>= 500), split it into small batches
        max_n_span = 400
        if bsz * n_span >= max_n_span:
            h_span_level = []
            # if n_span <= max_n_span:
            #     for i in range(bsz):
            #         m = (entity_masks[i].unsqueeze(-1) == 0).float() * (-1e30)  # n_span L 1
            #         h = m + h_word_level[i].unsqueeze(0).repeat(n_span, 1, 1)  # n_span L H
            #         h = h.max(dim=1)[0]  # n_span H
            #         h_span_level.append(h)
            # else:
            indices = list(range(0, n_span, max_n_span)) + [n_span]
            for i in range(bsz):
                h_span_level_i = []
                for j in range(len(indices) - 1):
                    s, e = indices[j], indices[j+1]
                    m = (entity_masks[i, s:e, :].unsqueeze(-1) == 0).float() * (-1e30)  # n_span L 1
                    h = m + h_word_level[i].unsqueeze(0).repeat(e-s, 1, 1)  # n_span L H
                    h = h.max(dim=1)[0]  # n_span H
                    h_span_level_i.append(h)
                h_span_level_i = torch.cat(h_span_level_i)
                h_span_level.append(h_span_level_i)
            h_span_level = torch.stack(h_span_level)
        else:
            # max pool entity candidate spans
            m = (entity_masks.unsqueeze(-1) == 0).float() * (-1e30)  # B n_span L 1
            h_span_level = m + h_word_level.unsqueeze(1).repeat(1, n_span, 1, 1) # B n_span L H
            h_span_level = h_span_level.max(dim=2)[0]  # B n_span H

        # m = entity_masks.unsqueeze(-1).float()  # B n_span L 1
        # print(h_word_level.shape, n_span)
        # h_span_level = m * h_word_level.unsqueeze(1).repeat(1, n_span, 1, 1)  # B
        #
        # # n_span L H
        # h_span_level = h_span_level.sum(dim=2)  # B n_span H
        # span_len = torch.sum(entity_masks, dim=-1).unsqueeze(-1).float()  # B n_span
        # h_span_level = torch.div(h_span_level, span_len + 1e-10)  # B n_span H

        return h_span_level

    def _get_gcn_in_features(
            self, sgs_nodes, spans_feature, nodes_feature,
            sgs_nodes_word_strs
        ):
        """
            sgs_nodes: List[List[Int]]
            spans_feature: Tensor with shape (n_span, feature_dim)
            nodes_feature: Tensor with shape (n_span, feature_dim)
        """

        if len(nodes_feature) > 0:
            nodes_feature = torch.cat(nodes_feature, dim=0)

        # for span/node indices
        span_indices = []  # tobe: List[Int]
        nodes_indices = []  # tobe: List[List[Int]]

        n_span = len(sgs_nodes)
        idx = 0
        for i in range(n_span):
            len_sgs_nodes_i = len(sgs_nodes[i])
            nodes_indices.append(list( range(idx, idx + len_sgs_nodes_i) ))
            span_indices.append(idx + len_sgs_nodes_i)
            idx += (len_sgs_nodes_i + 1)

        # for efficient torch.index_selcect
        if len(nodes_feature) > 0:
            n_nodes = len(nodes_feature)
        else:
            n_nodes = 0
        n_spans = len(spans_feature)
        total_node_span = n_nodes + n_spans
        indices = [0 for _ in range(total_node_span)]

        cur_n_span = 0
        for si in span_indices:
            indices[si] = n_nodes + cur_n_span
            cur_n_span += 1

        cur_n_node = 0
        for ni in flat1(nodes_indices):
            indices[ni] = cur_n_node
            cur_n_node += 1

        # for features
        if len(nodes_feature) > 0:
            features = torch.cat([nodes_feature, spans_feature], dim=0)
        else:
            features = spans_feature
        features = torch.index_select(features, dim=0, index=torch.LongTensor(
            indices).to(self.dev))

        return features, span_indices, nodes_indices

    def _span_attn_node(self, h_n, node_indices, h_s):
        """
            h: n_node_span, gcn_out_dim
            node_indices: List[List[Int]]
        """

        h_graph = []
        for i in range(len(node_indices)):
            ni = node_indices[i]
            if len(ni) > 0:
                h = h_n.index_select(0, torch.LongTensor(ni).to(self.dev))
                if self.graph_emb_method == "mean":
                    h_graph.append(h.mean(dim=0))
                elif self.graph_emb_method == "attn":
                    tmp_h = torch.mm(h, self.W_graph_attn.weight)
                    tmp_h = torch.mm(tmp_h, h_s[i, :].unsqueeze(-1)).squeeze(1)
                    attn = F.softmax(tmp_h)
                    attn_h = torch.mm(attn.unsqueeze(0), h)
                    h_graph.append(attn_h.squeeze(0))
            else:
                h_graph.append(torch.zeros(h_n.shape[1]).to(self.dev))
        h_graph = torch.stack(h_graph, dim=0)

        return h_graph

    def _forward_gcn(
        self, span_reprs, node_reprs, batch_sgs,
        batch_sgs_nodes, n_node_for_batch_and_span,
        sgs_nodes_word_strs,
    ):
        """
        Input:
            span_reprs: B n_span h_span+h_size
            node_reprs: B * n_span * n_node * h_span+h_size
            batch_sgs: batch subgraphs List[List[(start, end, Graph)]]
            batch_sgs_nodes: List[List[List[Int]]]
        """

        batch_size, n_span , _ = span_reprs.size()
        # h_span = torch.zeros(
        #     [batch_size, n_span, self.gcn.out_dim],
        #     requires_grad=True, dtype=torch.float
        # ).to(self.dev) # B, n_span, gcn_out_dim

        h_graph = torch.zeros(
            [batch_size, n_span, self.gcn.out_dim],
            requires_grad=True, dtype=torch.float
        ).to(self.dev)  # B, n_span, gcn_out_dim

        h_node = []  # tobe: n_node_in_B, gcn_out_dim (for ease)

        for i in range(batch_size): # i-th batch
            _h_span_inst = []  # tobe: real_n_span, gcn_out_dim
            _h_node_inst = []  # tobe: real_n_node, gcn_out_dim
            _h_graph_inst = []  # tobe: real_n_node, gcn_out_dim
            for j, (b, e, batch_g) in enumerate(batch_sgs[i]): # j-th node

                batch_g = batch_g.to(self.dev)
                # features, span_indices, node_indices = self._get_gcn_in_features(
                #     sgs_nodes=batch_sgs_nodes[i][b: e + 1], # 0 is padding
                #     spans_feature=span_reprs[i, b: e + 1, :],
                #     nodes_feature=node_reprs[i][b: e + 1],
                #     sgs_nodes_word_strs=sgs_nodes_word_strs[i][b: e + 1]
                # ) # n_node_span_in_be, gcn_in_dim

                sgs_nodes = batch_sgs_nodes[i][b: e + 1]
                nodes_indices = []  # tobe: List[List[Int]]
                n_span = len(batch_sgs_nodes[i][b: e + 1])
                idx = 0
                for k in range(n_span):
                    len_sgs_nodes_k = len(sgs_nodes[k])
                    nodes_indices.append(list(range(idx, idx + len_sgs_nodes_k)))
                    idx += len_sgs_nodes_k

                nodes_feature = node_reprs[i][b: e + 1]
                # if len(nodes_feature) > 0:

                features = torch.cat(nodes_feature, dim=0) # n_node_in_be, gcn_in_dim

                h = self.gcn(
                    g=batch_g, features=features, readout_weights=None
                )  # n_node_span_in_be, gcn_out_dim
                # h_cat = torch.cat([h, features], dim=1) # n_node_span_in_be, gcn_out_dim

                _h_node_inst.append(h)  # n_span_in_be, gcn_out_dim

                _h_graph_inst.append(
                    self._span_attn_node(h, nodes_indices, span_reprs[i, b: e + 1, :])
                )  # n_span_in_be, gcn_in_dim + gcn_out_dim

                # _h_span_inst.append(
                #     h.index_select(0, torch.LongTensor(span_indices).to(self.dev))
                # )  # n_span_in_be, gcn_out_dim
                #
                # _h_graph_inst.append(
                #     self._span_attn_node(h, span_indices, node_indices)
                # )  # n_span_in_be, gcn_in_dim + gcn_out_dim
                #
                # _h_node_inst.append(
                #     h.index_select(0, torch.LongTensor(flat1(node_indices)).to(self.dev))
                # )  # n_node_in_be, gcn_out_dim

            # _h_span_inst = torch.cat(_h_span_inst, dim=0)
            # real_n_span = _h_span_inst.shape[0]
            # h_span[i, :real_n_span, :] = _h_span_inst  # real_n_span, gcn_out_dim

            tmp_h_graph = torch.cat(_h_graph_inst, dim=0)  # real_n_span, gcn_in_dim +
            # gcn_out_dim
            h_graph[i, :len(tmp_h_graph), :] = tmp_h_graph
            h_node.extend(_h_node_inst)
        h_node = torch.cat(h_node, dim=0) # n_node_in_B, gcn_in_dim + gcn_out_dim

        # return h_span, h_node, h_graph
        return h_node, h_graph

    def _fine_tune_lm_embed(
        self, lm_spans, lm_bpe_ids, lm_attn_masks, maxlen
    ):
        """
        Input:
            lm_spans: bsz * n_token, List[List[Int]]
            lm_bpe_ids: bsz * max_n_bpe, Tensor[bsz * max_n_bpe]
            lm_attn_masks: bsz * max_n_bpe, Tensor[bsz * max_n_bpe]
            maxlen: int
        Output:
            h_lm:
            lm_mask: None # todo: add lm_mask
            h_cls: None # todo: add h_cls
        """

        lm_out = self.lm(
            input_ids=lm_bpe_ids, attention_mask=lm_attn_masks, output_hidden_states=True
        )  # 1. last hidden 2. pooler of CLS 3. all hidden states
        # bpe_h_lm = torch.stack(lm_out[2][-4:])  # 4 B L H
        # bpe_h_lm = torch.mean(bpe_h_lm, dim=0)  # B L H, mean-pool last 4 hiddens
        bpe_h_lm = lm_out[2][-1]

        bsz = len(lm_spans)

        h_lm = torch.zeros([bsz, maxlen, bpe_h_lm.shape[-1]], requires_grad=True).to(
            self.dev)

        for batch_i in range(len(lm_spans)):
            idx = 1 # 1 for CLS
            for token_j in range(len(lm_spans[batch_i])):
                # torch.mean(bpe_h_lm[batch_i, idx:idx + lm_spans[batch_i][token_j], :], dim=0)
                h_lm[batch_i, token_j, :] = torch.max(
                    bpe_h_lm[batch_i, idx:idx + lm_spans[batch_i][token_j], :], dim=0
                )[0]
                idx += lm_spans[batch_i][token_j]

        lm_mask = None
        # h_cls = lm_out[1]
        h_cls = lm_out[2][-1][:, 0, :]

        return h_lm, lm_mask, h_cls

    def _encode_to_word_level(
        self, char_ids, word_ids, word_strs, is_entity,
        lm_spans=None, lm_bpe_ids=None, lm_attn_masks=None,
    ):
        """
        Input:
            char_ids: B * max_n_word * max_char_len
            word_ids: B * max_n_word
            lm_bpe_ids: B * max_n_bpe
            lm_attn_masks: B * max_n_bpe
            lm_spans: List[List[Int]], word length in bpe
        Output:
            h_word_level: B * max_n_word * (2 * wc_wc_lstm_h)
            h_cls: B * h_lm
        """

        if self.use_char_encoder:
            h_char = self.char_encoder(char_ids)  # B, max_n_word, 2 * char_h
        if self.use_word_encoder:
            h_word = self.word_embeds(word_ids)  # B, max_n_word, word_h

        # method 1
        # if self.use_char_encoder or self.use_word_encoder:
        #     if self.use_char_encoder and self.use_word_encoder:
        #         h_wc = torch.cat(
        #             (h_char, h_word), dim=-1
        #         )  # B, max_n_word, 2 * char_h + word_h
        #     else:
        #         h_wc = h_char if self.use_char_encoder else h_word
        #     h_wc = self.wc_enc_dropout(h_wc)
        #     h_wc = word_lstm_forward(
        #         lstm=self.wc_lstm_enc, x=h_wc, mask=(word_ids != 0)
        #     ) # B, max_n_word, wc_lstm_h
        #
        # h_lm, h_cls = None, None
        # if self.use_lm_embed:
        #     if not is_entity:
        #         h_lm, lm_mask, h_cls = self.lm_embed(word_strs, maxlen=512)
        #     else:
        #         h_lm, lm_mask, h_cls = self.ent_lm_embed(word_strs, maxlen=512)
        #
        # if self.use_char_encoder or self.use_word_encoder:
        #     if self.use_lm_embed:
        #         h_word_level = torch.cat((h_wc, h_lm), dim=-1) # B, max_n_word, lm_h + wc_lstm_h
        #     else:
        #         h_word_level = h_wc # B, max_n_word, lm_h + wc_lstm_h
        # else:
        #     h_word_level = h_lm # B, max_n_word, lm_h



        # method 2
        h_cls = None

        if self.use_char_encoder or self.use_word_encoder:
            if self.use_char_encoder and self.use_word_encoder:
                h_wc = torch.cat(
                    (h_char, h_word), dim=-1
                )  # B, max_n_word, 2 * char_h + word_h
            else:
                h_wc = h_char if self.use_char_encoder else h_word
            if self.use_lm_embed:
                if not is_entity:
                    # h_lm, lm_mask, h_cls = self.lm_embed(word_strs, maxlen=512)
                    h_lm, lm_mask, h_cls = self._fine_tune_lm_embed(
                        lm_spans, lm_bpe_ids, lm_attn_masks, maxlen=self.max_n_word
                    )
                else:
                    h_lm, lm_mask, h_cls = self.ent_lm_embed(word_strs, maxlen=self.max_ent_n_word)
                h_wc = torch.cat(
                    (h_wc, h_lm), dim=-1
                )

            # print("aaa", h_word.shape, h_char.shape, h_wc.shape)

            h_wc = self.wc_enc_dropout(h_wc)
            h_wc = word_lstm_forward(
                lstm=self.wc_lstm_enc, x=h_wc, mask=(word_ids != 0)
            ) # B, max_n_word, wc_lstm_h
        else:
            if self.use_lm_embed:
                if not is_entity:
                    # h_lm, lm_mask, h_cls = self.lm_embed(word_strs, maxlen=512)
                    h_lm, lm_mask, h_cls = self._fine_tune_lm_embed(
                        lm_spans, lm_bpe_ids, lm_attn_masks, maxlen=self.max_n_word
                    )
                else:
                    h_lm, lm_mask, h_cls = self.ent_lm_embed(word_strs, maxlen=self.max_ent_n_word)
                h_wc = h_lm

            # h_wc = self.wc_enc_dropout(h_wc)
            # h_wc = word_lstm_forward(
            #     lstm=self.wc_lstm_enc, x=h_wc, mask=(lm_bpe_ids != 0)
            # )  # B, max_n_word, wc_lstm_h
        h_word_level = h_wc

        # if self.encoder_linear is not None:
        #     h_word_level = self.encoder_linear(h_word_level) # B, max_n_word,
        #     # 2 * wc_lstm_h
        #     h_word_level = self.encoder_relu(h_word_level)
        # h_word_level = self.wc_enc_dropout(h_word_level)

        return h_word_level, h_cls

    def _get_logits(self, sample):
        """
        Input:
            sample
                entity_masks: B n_span L
                entity_sizes: B num_sample
                char_ids: B * max_n_word * max_char_len
                word_ids: B * max_n_word
                word_strs: B * max_n_word
                lm_spans: List[List[Int]], word length in bpe
        Output:
            logits_span, logits_node
        """
        h_word_level, h_cls = self._encode_to_word_level(
            char_ids=sample["char_ids"], word_ids=sample["word_ids"],
            word_strs=sample["word_strs"], is_entity=False,
            lm_spans=sample["lm_spans"], lm_bpe_ids=sample["bpe_ids"],
            lm_attn_masks=sample["lm_attn_masks"]
        ) # B * max_n_word * h_lm (or 2 * wc_lstm_h), B * h_lm

        h_span_level = self._encode_to_span_level(
            h_word_level=h_word_level, entity_masks=sample["entity_masks"]
        ) # B * max_n_span * (2 * wc_lstm_h)

        if self.use_size:
            h_span_size = self.size_embeds(sample["entity_sizes"]) # B max_n_span h_size

        if self.use_gcn:
            # add h_ctx.unsqueeze(1).repeat(1, h_pool.shape[1], 1) or not
            # B * max_n_span * (h_span_level+h_size) # +H_ctx
            if self.use_size:
                span_reprs = torch.cat([h_span_level, h_span_size], dim=2)
            else:
                span_reprs = h_span_level

            bsz = len(sample["char_ids"])
            node_reprs = [] # tobe: B * n_span * n_node * h_span_level+h_size

            # 可能出现bug的边界情况
            # 1. 有一句话中有多个span，但是它们都没有nodes
            # 2. 有一句话中没有一个span

            n_node_for_batch_and_span = []
            for i in range(bsz):
                n_node_for_span = [len(d) for d in sample["sgs_nodes_char_ids"][i]]
                n_node_for_batch_and_span.append(n_node_for_span)

                n_nodes = len([d for dd in sample["sgs_nodes_char_ids"][i] for d in dd])

                if n_nodes == 0:
                    h_span_level_dim = 2 * self.wc_lstm_hid_dim + int(self.use_size) * \
                                       self.size_embed_dim
                    _h_span_level = torch.zeros(0, h_span_level_dim).to(self.dev)

                    # recover nodes in span
                    _cur_span_node_reprs = []
                    n_span = len(sample["sgs_nodes_char_ids"][i])
                    for i_span in range(n_span):
                        _cur_span_node_reprs.append(_h_span_level)
                    node_reprs.append(_cur_span_node_reprs)
                    continue

                flat_word_strs = None
                if self.use_lm_embed:
                    flat_word_strs = flat2(sample["sgs_nodes_word_strs"][i], False)

                # f1 = flat2(sample["sgs_nodes_word_ids"][i], tensor=True)
                # f2 = flat2(sample["sgs_nodes"][i])
                # if len(f1) != len(f2):
                #     print(len(f1), len(f2))
                #     print(sample["sgs_nodes_word_ids"][i], "\n", sample["sgs_nodes"][i])

                _h_word_level, _ = self._encode_to_word_level(
                    char_ids=flat2(sample["sgs_nodes_char_ids"][i], True, self.dev),
                    word_ids=flat2(sample["sgs_nodes_word_ids"][i], True, self.dev),
                    word_strs=flat_word_strs,
                    is_entity=True,
                )  # (n_span * n_node) * max_n_word * (2 * wc_lstm_h), None

                # _h_span_level = _h_word_level.max(dim=1)[0]
                _h_span_level = _h_word_level.mean(dim=1) # (n_span*n_node) * (2*wc_lstm_h)
                if self.use_size:
                    h_size = self.size_embeds(
                        torch.LongTensor(flat2(sample["sgs_nodes_size"][i], False)).to(
                            self.dev)
                    )  # (n_span * n_node) * h_size

                    # (n_span * n_node) * (2 * wc_lstm_h + h_size)
                    _h_span_level = torch.cat([_h_span_level, h_size], dim=1)

                # f2 = flat2(sample["sgs_nodes"][i])
                # if len(_h_span_level) != len(f2):
                #     print(len(f1), len(f2))
                #     print(sample["sgs_nodes_word_ids"][i], "\n", sample["sgs_nodes"][i])
                #     print(len(f1), len(_h_span_level))

                # recover nodes in span
                idx = 0
                _cur_span_node_reprs = []
                n_span = len(sample["sgs_nodes_char_ids"][i])
                for i_span in range(n_span):
                    n_node = len(sample["sgs_nodes_char_ids"][i][i_span])
                    _cur_span_node_reprs.append(_h_span_level[idx: idx+n_node, :])
                    idx += n_node
                node_reprs.append(_cur_span_node_reprs)

            h_gcn_node, h_gcn_graph = self._forward_gcn(
                span_reprs=span_reprs,  # B n_span h_span_level+h_size
                node_reprs=node_reprs,  # B n_span n_node h_span_level+h_size
                batch_sgs=sample["sgs"],  # List[List[Graph]]
                batch_sgs_nodes=sample["sgs_nodes"],  # List[List[List[Int]]]
                n_node_for_batch_and_span=n_node_for_batch_and_span,  # List[List[Int]]
                sgs_nodes_word_strs=sample["sgs_nodes_word_strs"],  # for debug
            )

            # h_gcn_span, h_gcn_node, h_gcn_graph = self._forward_gcn(
            #     span_reprs=span_reprs, # B n_span h_span_level+h_size
            #     node_reprs=node_reprs, # B n_span n_node h_span_level+h_size
            #     batch_sgs=sample["sgs"],  # List[List[Graph]]
            #     batch_sgs_nodes=sample["sgs_nodes"],  # List[List[List[Int]]]
            #     n_node_for_batch_and_span=n_node_for_batch_and_span,  # List[List[Int]]
            #     sgs_nodes_word_strs=sample["sgs_nodes_word_strs"], # for debug
            # )
            # B * n_span * gcn_out_dim
            # n_node_in_B * (gcn_in_dim+gcn_out_dim)
            # B * n_span * (gcn_in_dim+gcn_out_dim)

        # span representations
        if self.concat_span_hid:
            h_span = h_span_level

        if self.use_size:
            h_span = torch.cat([h_span_size, h_span], dim=2) # B n_span H_ctx+H_size

        if self.use_cls:
            h_span = torch.cat(
                [
                    h_cls.unsqueeze(1).repeat(1, h_span.shape[1], 1),  # 1024
                    h_span
                ],
                dim=2
            )  # B n_span H_ctx+H_size

        if self.use_gcn:
            h_span = torch.cat([h_gcn_graph, h_span], dim=2)
        # B n_span h_gcn_span+h_gcn_graph+h_size

        h_span = self.span_dropout(h_span)
        logits_span = self.fc_out_span(h_span) # B n_span label_size

        # node representations
        logits_node = None
        if self.use_gcn:
            logits_node = self.fc_out_node(h_gcn_node) # n_node_in_B label_size

        return logits_span, logits_node

    def _get_span_loss(self, logits_span, entity_types, entity_sample_masks):

        logits_span = logits_span.view(-1, logits_span.shape[-1])
        labels_span = entity_types.view(-1)

        loss_span = Cross_Entropy(
            logits_span, labels_span,
            sample_masks=entity_sample_masks.view(-1),
            is_logits=True, pad_idx=int(1e8)  # 1e8 means no pad
        )

        if loss_span > 1e6:  # Weibo NER dataset has a huge loss in one instance (id=143)
            loss_span *= 0

        sample_size_span = int(entity_sample_masks.sum())
        if sample_size_span > 0:
            loss_span = loss_span / sample_size_span
        else:
            loss_span = 0 * loss_span

        return loss_span, sample_size_span

    def _get_node_loss(self, logits_node, sgs_node_types):
        labels_node = []
        for i in range(len(sgs_node_types)):
            for j in range(len(sgs_node_types[i])):
                labels_node.extend(sgs_node_types[i][j])
        labels_node = torch.LongTensor(labels_node).to(self.dev)
        sample_mask = torch.ones(labels_node.size()).to(self.dev)
        loss_node = Cross_Entropy(
            logits_node, labels_node, sample_masks=sample_mask,
            is_logits=True, pad_idx=int(1e8)  # 1e8 means no pad
        )
        if loss_node > 1e6:  # Weibo NER dataset has a huge loss in one instance (id=143)
            loss_node *= 0

        sample_size_node = logits_node.shape[0]
        if sample_size_node > 0:
            loss_node = loss_node / sample_size_node
        else:
            loss_node = 0 * loss_node

        return loss_node, sample_size_node

    def forward(self, sample):
        """
        Input:
            sample (Dict): from Train_Mode
                encodings: B L
                context_masks: B L
                entity_masks: B sample_size_ent L
                entity_sizes: B sample_size_ent
                entity_types: B sample_size_ent
                entity_sample_masks: B sample_size_ent
        Output:
            log_output (Dict): some losses and infomation
        """

        # loss of spans
        logits_span, logits_node = self._get_logits(sample)
        loss_span, sample_size_span = self._get_span_loss(
            logits_span, entity_types=sample["entity_types"],
            entity_sample_masks=sample["entity_sample_masks"]
        )

        loss_node, sample_size_node = torch.zeros([]), 0
        if self.use_gcn:
            loss_node, sample_size_node = self._get_node_loss(
                logits_node, sgs_node_types=sample["sgs_node_types"]
            )

        loss = loss_span + self.alpha * loss_node

        log_output = {
            "loss": loss.data.item(),
            "loss_span": loss_span.data.item(),
            "loss_node": loss_node.data.item(),
            "sample_size_span": sample_size_span,
            "sample_size_node": sample_size_node,
        }

        return loss, log_output

    def evaluate(self, sample, gt_entities):

        if self.show_infer_speed:
            infer_time = 0

        if self.show_infer_speed:
            start_t = time()
        logits_span, logits_node = self._get_logits(sample)
        if self.show_infer_speed:
            infer_time += time () - start_t

        loss_span, sample_size_span = self._get_span_loss(
            logits_span, entity_types=sample["entity_types"],
            entity_sample_masks=sample["entity_sample_masks"]
        )

        loss_node, sample_size_node = torch.zeros([]), 0
        if self.use_gcn:
            loss_node, sample_size_node = self._get_node_loss(
                logits_node, sgs_node_types=sample["sgs_node_types"]
            )

        loss = loss_span + self.alpha * loss_node

        n_preds, n_golds, n_inters = 0, 0, 0

        if self.do_extra_eval:
            pred_data, gold_data = [], []


        if self.show_infer_speed:
            start_t = time()
        best_ids = logits_span.argmax(dim=-1) * sample["entity_sample_masks"].long()
        if self.show_infer_speed:
            infer_time += time () - start_t

        batch_size = logits_span.shape[0]
        for i in range(batch_size):
            entity_type_ids = best_ids[i]

            # get entities that are not classified as 'None'
            valid_entity_indices = entity_type_ids.nonzero().view(-1)
            valid_entity_types = entity_type_ids[valid_entity_indices]
            valid_entity_spans = sample["entity_spans"][i][valid_entity_indices]

            pred_spans, gold_spans = [], []
            if self.do_extra_eval:
                pred_phrases, gold_phrases = [], []
                node_entity_types = []

            # def _f(bpe_list):
            #     bpe_strings = ' '.join(bpe_list)
            #     bpe_strings = bpe_strings.replace(" ##", "")
            #     return bpe_strings

            for j in range(len(valid_entity_types)):
                span_b = int(valid_entity_spans[j][0])
                span_e = int(valid_entity_spans[j][1])
                entity_type = self.idx2entity_type[int(valid_entity_types[j])]
                pred_spans.append((span_b, span_e, entity_type))
                if self.do_extra_eval:
                    phrase = ' '.join(sample["word_strs"][i][span_b: span_e + 1])
                    pred_phrases.append(phrase)

            for j in range(len(gt_entities[i])):
                span_b, span_e, entity_type_id = gt_entities[i][j]
                entity_type = self.idx2entity_type[int(entity_type_id)]
                gold_spans.append((span_b, span_e, entity_type))
                if self.do_extra_eval:
                    phrase = ' '.join(sample["word_strs"][i][span_b: span_e + 1])
                    gold_phrases.append(phrase)
                    if self.use_gcn:
                        node_entity_types.append(
                            [self.idx2entity_type[t] for t in
                             sample["gold_sgs_node_types"][i][j]]
                        )

            _pred_spans = set(pred_spans)
            _gold_spans = set(gold_spans)
            n_preds += len(_pred_spans)
            n_golds += len(_gold_spans)
            n_inters += len(_pred_spans & _gold_spans)

            if self.do_extra_eval:
                pred_data.append({
                    "spans": list(pred_spans), "phrases": pred_phrases,
                })
                gold_data.append({
                    "spans": list(gold_spans), "phrases": gold_phrases,
                    # "node_entity_types": node_entity_types
                    # todo: 暂时不需要 node_entity_types， 之后可能需要 node entity phrase 等其他，需要时再加
                })

        log_output = {
            "loss": loss.data.item(),
            "loss_span": loss_span.data.item(),
            "loss_node": loss_node.data.item(),
            "sample_size_span": sample_size_span,
            "sample_size_node": sample_size_node,
            "n_preds": n_preds,
            "n_golds": n_golds,
            "n_inters": n_inters,
            "extra_eval_data": {
                "pred_data": pred_data,
                "gold_data": gold_data,
                "word_strs": sample["word_strs"],
            } if self.do_extra_eval else None,
        }

        if not self.show_infer_speed:
            infer_time = 0
        return loss, log_output, infer_time

    def save_extra_eval_data(self, file, log_output):
        data = []
        extra_eval_data = log_output["extra_eval_data"]
        data_size = len(extra_eval_data["pred_data"])

        for i in range(data_size):
            data.append({
                "id": i,
                "word_strs": extra_eval_data["word_strs"][i],
                "pred_data": extra_eval_data["pred_data"][i],
                "gold_data": extra_eval_data["gold_data"][i],
            })

        with open(file, 'w') as f:
            json.dump(data, f, ensure_ascii=False, indent=1)

    def aggregate_log_outputs(self, log_outputs):
        """Aggregate log outputs from data parallel training."""
        loss = sum(log.get('loss', 0) for log in log_outputs)
        loss_span = sum(log.get('loss_span', 0) for log in log_outputs)
        loss_node = sum(log.get('loss_node', 0) for log in log_outputs)
        sample_size_span = sum(log.get('sample_size_span', 0) for log in log_outputs)
        sample_size_node = sum(log.get('sample_size_node', 0) for log in log_outputs)

        n_preds = sum(log.get('n_preds', 0) for log in log_outputs)
        n_golds = sum(log.get('n_golds', 0) for log in log_outputs)
        n_inters = sum(log.get('n_inters', 0) for log in log_outputs)

        p, r, f = calculate_prf(n_preds, n_golds, n_inters)

        if not self.training and self.do_extra_eval:
            extra_eval_data = {
                "pred_data": [],
                "gold_data": [],
                "word_strs": []
            }
            for log in log_outputs:
                data = log["extra_eval_data"]
                extra_eval_data["pred_data"].extend(data["pred_data"])
                extra_eval_data["gold_data"].extend(data["gold_data"])
                extra_eval_data["word_strs"].extend(data["word_strs"])

        aggregate_output = {
            "loss": loss,
            "loss_span": loss_span,
            "loss_node": loss_node,
            "sample_size_span": sample_size_span,
            "sample_size_node": sample_size_node,
            "p": p,
            "r": r,
            "f": f,
            "extra_eval_data": extra_eval_data \
                if not self.training and self.do_extra_eval else None,
        }
        return aggregate_output


    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument('--num_rnn_layers', type=int)
        parser.add_argument('--rnn_dropout', type=float)
        parser.add_argument('--rnn_hidden_dim', type=int)
        parser.add_argument('--bidirectional', action='store_true')
        parser.add_argument('--rnn_type', type=str)
        parser.add_argument('--dropout', type=float)
        parser.add_argument(
            '--pretrained_model',
            choices=[
                "bert-base-cased", "bert-base-uncased", "bert-large-uncased",
                "bert-base-chinese", "dmis-lab/biobert-base-cased-v1.1"
            ],
            help="BERT: https://github.com/huggingface/transformers/blob/"
                 "master/src/transformers/models/bert/tokenization_bert.py"
        )
        parser.add_argument('--neg_entity_count', type=int,
                           help='-1 means using all entity')
        parser.add_argument('--max_span_size', type=int)
        parser.add_argument('--size_embed_dim', default=25, type=int)
        parser.add_argument('--gcn_dropout', type=float)
        parser.add_argument('--gcn_layers', type=str)
        parser.add_argument('--gcn_norm_method', type=str)
        parser.add_argument('--do_lower_case', type=bool)
        parser.add_argument('--wc_lstm_layers', type=int)


@register_model_architecture("nner_slg", "nner_slg")
def nner_slg(args):
    args.char_embed_dim = getattr(args, 'char_embed_dim', 768) # fixed for bert base
    args.num_rnn_layers = getattr(args, 'num_rnn_layers', 0)
    args.rnn_dropout = getattr(args, 'rnn_dropout', 0)
    args.rnn_hidden_dim = getattr(args, 'rnn_hidden_dim', 0)
    args.bidirectional = getattr(args, 'bidirectional', False)
    args.rnn_type = getattr(args, 'rnn_type', 'lstm')
    args.dropout = getattr(args, 'dropout', 0.1)
    args.neg_entity_count = getattr(args, 'neg_entity_count', 100)
    args.size_embed_dim = getattr(args, 'size_embed_dim', 25)
    args.gcn_dropout = getattr(args, 'gcn_dropout', 0.3)
    args.gcn_layers = getattr(args, 'gcn_layers', "16")
    args.gcn_norm_method = getattr(args, 'gcn_norm_method', "right")




@register_model_architecture("nner_slg", "nner_slg_ace04")
def nner_slg_ace04(args):
    args.pretrained_model = getattr(args, 'pretrained_model', "bert-base-cased")
    args.do_lower_case = getattr(args, 'do_lower_case', False)
    args.lm_dim = getattr(args, 'lm_dim', 768)
    args.cls_embed_dim = getattr(args, 'cls_embed_dim', 768)

    # args.pretrained_model = getattr(args, 'pretrained_model', "bert-large-uncased")
    # args.do_lower_case = getattr(args, 'do_lower_case', True)
    # args.lm_dim = getattr(args, 'lm_dim', 1024)
    # args.cls_embed_dim = getattr(args, 'cls_embed_dim', 1024)

    args.max_span_size = getattr(args, 'max_span_size', 10)
    args.max_n_word = 122
    args.max_n_bpe = 137
    args.max_ent_n_word = 58
    args.max_word_len = 68
    args.wc_lstm_layers = 1
    args.test_n_words = 17822
    nner_slg(args)

@register_model_architecture("nner_slg", "nner_slg_ace05")
def nner_slg_ace05(args):
    args.pretrained_model = getattr(args, 'pretrained_model', "bert-base-cased")
    args.do_lower_case = getattr(args, 'do_lower_case', False)
    args.lm_dim = getattr(args, 'lm_dim', 768)
    args.cls_embed_dim = getattr(args, 'cls_embed_dim', 768)

    # args.pretrained_model = getattr(args, 'pretrained_model', "bert-large-uncased")
    # args.do_lower_case = getattr(args, 'do_lower_case', True)
    # args.lm_dim = getattr(args, 'lm_dim', 1024)
    # args.cls_embed_dim = getattr(args, 'cls_embed_dim', 1024)

    args.max_span_size = getattr(args, 'max_span_size', 10)
    args.max_n_word = 104
    args.max_n_bpe = 134
    args.max_ent_n_word = 50
    args.max_word_len = 59
    args.wc_lstm_layers = 1
    args.test_n_words = 17909
    nner_slg(args)

@register_model_architecture("nner_slg", "nner_slg_genia")
def nner_slg_genia(args):
    # args.pretrained_model = getattr(args, 'pretrained_model', "dmis-lab/biobert-large-cased-v1.1")
    # args.do_lower_case = getattr(args, 'do_lower_case', False)
    # args.lm_dim = getattr(args, 'lm_dim', 1024)
    # args.cls_embed_dim = getattr(args, 'cls_embed_dim', 1024)

    args.pretrained_model = getattr(args, 'pretrained_model',
                                    "dmis-lab/biobert-base-cased-v1.1")
    args.do_lower_case = getattr(args, 'do_lower_case', False)
    args.lm_dim = getattr(args, 'lm_dim', 768)
    args.cls_embed_dim = getattr(args, 'cls_embed_dim', 768)

    args.max_span_size = getattr(args, 'max_span_size', 10)
    args.max_n_word = 176
    args.max_n_bpe = 375
    args.max_ent_n_word = 21
    args.max_word_len = 100
    args.wc_lstm_layers = 1
    args.test_n_words = 50182
    nner_slg(args)

