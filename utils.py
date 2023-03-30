import os
import re
import sys
import json
import torch
import random
import logging
import numpy as np
import pickle as pkl
import networkx as nx

from glob import glob
from tqdm import tqdm

from collections import OrderedDict

from data import sampling
from data.node import Node
from models.utils import evaluate_each_entity_type

from time import time

log_formatter = logging.Formatter(
    "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(log_formatter)

file_handler = logging.FileHandler(os.environ.get("LOGFILENAME", "~/tmp.log"))
file_handler.setFormatter(log_formatter)

def set_seed(seed, use_cuda=False):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    # seeds all about torch
    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed(seed) # current gpu
        torch.cuda.manual_seed_all(seed) # all gpus

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

def _load_dict(file, res_dict):
    with open(file, "r", encoding="utf-8") as f:
        for line in f.readlines():
            k, v = line.strip().split()
            res_dict[k] = len(res_dict)
    return res_dict

def _load_ordered_dict(file, res_dict):
    with open(file, "r") as f:
        for i, line in enumerate(f.readlines()):
            if i < 2:
                continue
            k, v = line.strip().split()
            res_dict[k] = len(res_dict)
    return res_dict

def load_dicts(data_dir, is_bert=False):
    """
        currently only load label2idx, char2idx
        other dicts: char2idx, seg2idx, bichar2idx
    """

    label2idx = _load_dict(f"{data_dir}/label_dict.txt", {"<PAD>": 0, "<BOS>": 1, "<EOS>": 2})
    if is_bert:
        char2idx = _load_ordered_dict(f"{data_dir}/char_dict.txt", OrderedDict({"<PAD>": 0, "<UNK>": 1}))
    else:
        char2idx = _load_dict(f"{data_dir}/char_dict.txt", {"<PAD>": 0, "<UNK>": 1})
    entity_type2idx = _load_dict(f"{data_dir}/entity_type_dict.txt", {"O": 0})

    return label2idx, char2idx, entity_type2idx

def load_char_embeds(data_dir):
    with open(f"{data_dir}/char_embed.pkl", "rb") as f:
        char_embeds = pkl.load(f)
    return char_embeds

def load_bichar_embeds(data_dir):
    with open(f"{data_dir}/bichar_embed.pkl", "rb") as f:
        bichar_embeds = pkl.load(f)
    return bichar_embeds

def get_idx2sth(sth2idx_list):
    result = []
    for sth2idx in sth2idx_list:
        idx2sth = [0 for _ in range(len(sth2idx))]
        for k, v in sth2idx.items():
            idx2sth[v] = k
        result.append(idx2sth)
    return result

def get_begin_end_label_ids(idx2label):
    begin_label_ids, end_label_ids = [], []
    for i in range(len(idx2label)):
        label = idx2label[i]
        if label.startswith('B'):
            begin_label_ids.append(i)
        elif label.startswith('E'):
            end_label_ids.append(i)
    return begin_label_ids, end_label_ids

def move_to_cuda(sample):
    new_sample = {}
    for key, value in sample.items():
        if torch.is_tensor(value):
            new_sample[key] = value.cuda()
        elif isinstance(value, list):
            new_list = []
            for v in value:
                if torch.is_tensor(v):
                    new_list.append(v.cuda())
                else:
                    new_list.append(v)
            new_sample[key] = new_list
        else:
            new_sample[key] = value
    return new_sample

def evaluate(
        args,
        model, lr_scheduler, optimizer,
        checker, logger,
        valid_data, test_data,
        epoch, num_updates, cur_lr, total_norm,
        no_incre_on_val, max_val_f,
        best_val, best_test_by_val, best_test,
        entity_types,
        at_epoch=False, at_update=False
):
    assert at_epoch ^ at_update

    def _evaluate(data, model):
        data_itr = torch.utils.data.DataLoader(
            data,
            batch_size=args.eval_max_sentences,
            shuffle=False,
            drop_last=False,
            num_workers=args.sampling_processes,
            collate_fn=sampling.collate_fn_padding
        )

        gt_entities = []
        for doc in data.documents:
            cur_gt_entities = [entity.as_tuple() for entity in doc.entities]
            gt_entities.append(cur_gt_entities)

        model.eval()

        log_outputs = []
        start_idx = 0
        end_idx = 0
        i = 0
        for sample in data_itr:
            start_t = time()
            sample = move_to_cuda(sample)
            end_idx += int(sample["entity_spans"].shape[0])
            loss, log_output = model.evaluate(sample, gt_entities=gt_entities[start_idx:end_idx])
            log_outputs.append(log_output)
            start_idx = end_idx

            # p = log_output["n_inters"] / (log_output["n_preds"] + 1e-8)
            # r = log_output["n_inters"] / (log_output["n_golds"] + 1e-8)
            # f1 = 2 * p * r / (p + r + 1e-8)
            # print(
            #     i,
            #     "p=", round(p*100, 20),
            #     "r=", round(r*100, 20),
            #     "f1=", round(f1*100, 20),
            #     "time=", round(time() - start_t, 2),
            #     flush=True
            # )
            # i += 1

        log_output = model.aggregate_log_outputs(log_outputs)
        return log_output

    valid_log_output = _evaluate(valid_data, model)
    test_log_output = _evaluate(test_data, model)

    # judge overfitting by nll loss in valid set
    save_best = False
    if valid_log_output["f"] > max_val_f:
        max_val_f = valid_log_output["f"]
        save_best = True
        logger.set_best_epoch_update(epoch, num_updates)

    # checker.save_checkpoint(epoch, num_updates, model, lr_scheduler, optimizer, save_best)

    if at_epoch:
        logger.print_at_epoch(epoch, num_updates, cur_lr, total_norm, valid_log_output, test_log_output)
    if at_update:
        logger.print_at_update(epoch, num_updates, cur_lr, total_norm, valid_log_output, test_log_output)

    # for self-test
    if best_val["f"] < valid_log_output["f"]:
        if at_epoch:
            no_incre_on_val = 0
        best_val["p"] = valid_log_output["p"]
        best_val["r"] = valid_log_output["r"]
        best_val["f"] = valid_log_output["f"]

        best_test_by_val["p"] = test_log_output["p"]
        best_test_by_val["r"] = test_log_output["r"]
        best_test_by_val["f"] = test_log_output["f"]
    else:
        if at_epoch:
            no_incre_on_val += 1

    if best_test["f"] < test_log_output["f"]:
        best_test["p"] = test_log_output["p"]
        best_test["r"] = test_log_output["r"]
        best_test["f"] = test_log_output["f"]

    val_f = valid_log_output["f"]

    return no_incre_on_val, max_val_f, best_val, best_test_by_val, best_test, val_f

def reset_args(orig_args, ckpt_args):
    """This is for evaluation."""
    for k, v in ckpt_args.__dict__.items():
        if not hasattr(orig_args, k):
            setattr(orig_args, k, v)

def get_grad_norm(model):
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    return total_norm

def build_entity_type2idx(json_file_path, seed_dict):
    with open(json_file_path, "r") as f:
        entity_types = json.load(f)

    for entity_type in entity_types:
        seed_dict[entity_type] = len(seed_dict)

    return seed_dict

def read_nodes(node_json_file):
    with open(node_json_file, 'r') as f:
        nodes = json.load(f)
    return nodes

def load_nodes(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data

def load_edges(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data

def load_json(file):
    with open(file, 'r') as f:
        data = json.load(f)
    return data

def load_pickle(file):
    with open(file, 'rb') as f:
        data = pkl.load(f)
    return data

def save_pickle(data, file):
    with open(file, 'wb') as f:
        pkl.dump(data, f)


def partition_arg_topK(matrix, K, axis=0):
    """
    perform topK based on np.argpartition
    :param matrix: to be sorted
    :param K: select and sort the top K items
    :param axis: 0 or 1. dimension to be sorted.
    :return:
    """
    a_part = np.argpartition(matrix, K, axis=axis)
    if axis == 0:
        row_index = np.arange(matrix.shape[1 - axis])
        a_sec_argsort_K = np.argsort(matrix[a_part[0:K, :], row_index], axis=axis)
        return a_part[0:K, :][a_sec_argsort_K, row_index]
    else:
        column_index = np.arange(matrix.shape[1 - axis])[:, None]
        a_sec_argsort_K = np.argsort(matrix[column_index, a_part[:, 0:K]], axis=axis)
        return a_part[:, 0:K][column_index, a_sec_argsort_K]

def node_nearest_cands(nodes, edges, n_order_neighbor):
    """
    Filter edges by weights. Select the top n_order_neighbor nodes with weight > 0.
    """

    G = nx.Graph()
    G.add_nodes_from([n["id"] for n in nodes])
    G.add_edges_from([(e["node1"]["id"], e["node2"]["id"], {"weight": e["weight"]}) for e in edges])
    A = nx.to_numpy_matrix(G, weight="weight")
    nodes_topK_cands = partition_arg_topK(-A, n_order_neighbor, axis=1).tolist()

    entities_cands = []
    for id1, node_topK_cand in enumerate(nodes_topK_cands):
        entity_cands = []
        for id2 in node_topK_cand:
            if A[id1, id2] > 0:
                entity_cands.append(id2)
        entities_cands.append(entity_cands)
    return entities_cands

def build_entity2idx(nodes):
    entity2idx = dict()
    for i, node in enumerate(nodes):
        entity_phrase = " ".join(node["tokens"])
        entity2idx[entity_phrase] = i
    return entity2idx

def load_span_cands(span_cands_dir, edge_weight_threshold, train_size, n_cand):
    cands = dict()
    def _get_paths(data_label):
        paths = glob(f'{span_cands_dir}/{data_label}_*.json', recursive=False)
        return paths

    total_span_edges = 0
    actual_span_edges = 0

    for data_label in ["train", "valid", "test"]:
        cands[data_label] = dict()
        paths = _get_paths(data_label)
        for path in tqdm(paths):
            data = load_json(path)
            inst_idx = int(re.search(f'{data_label}_(\d+).json', path).group(1))
            if inst_idx >= train_size:
                continue # instance idx starts at 0
            cands[data_label][inst_idx] = dict()
            for d in data:
                span = " ".join(d["span"])
                cands[data_label][inst_idx][span] = []
                # for c in d["candidates"]:
                for c in d["candidates"][:n_cand]:
                    if c["weight"] > edge_weight_threshold:
                        cands[data_label][inst_idx][span].append(c)

                total_span_edges += len(d["candidates"])
                actual_span_edges += len(cands[data_label][inst_idx][span])

    return cands, total_span_edges, actual_span_edges

if __name__ == "__main__":
    pass