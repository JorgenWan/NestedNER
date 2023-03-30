"""
    Pretrain node/entity embedding by (Cluster-)GCN
"""

import os
import sys
import dgl
import json
import logging
import numpy as np
import time

import torch

from tqdm import tqdm

import utils
from options import get_train_arguments
from config import Config

from data.ner_dataset import NER_Dataset
from data import sampling
from data.iterators import Grouped_Iterator

from models.utils import build_model
from optimizers.utils import build_optimizer
from lr_schedulers.utils import build_lr_scheduler

from jlogger import Train_Logger
from checker import Train_Checker

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    handlers=[utils.console_handler, utils.file_handler]
)

logger = logging.getLogger("train")

def load_graph_data(cfg):
    cache_dir = cfg.cache_dir
    graph_dir = cfg.graph_dir

    os.makedirs(cache_dir, exist_ok=True)

    # load nodes (entity nodes)
    nodes_json_file = f"{graph_dir}/nodes.json"
    nodes_pkl_file = f"{cache_dir}/nodes.pkl"
    if os.path.exists(nodes_pkl_file) and cfg.load_cache:
        logger.info(f"[Graph] loading saved nodes from {nodes_pkl_file} ...")
        nodes = utils.load_pickle(nodes_pkl_file)
    else:
        logger.info(f"[Graph] loading nodes from {nodes_json_file} ...")
        nodes = utils.load_json(nodes_json_file)
        if cfg.save_cache:
            logger.info(f"[Graph] saving nodes to {nodes_pkl_file} ...")
            utils.save_pickle(nodes, nodes_pkl_file)
    logger.info(f"[Graph] entity-entity graph loads {len(nodes)} nodes.")

    # load edges (between entity nodes)
    edges_json_file = f"{graph_dir}/edges.json"
    edges_pkl_file = f"{cache_dir}/edges.pkl"
    if os.path.exists(edges_pkl_file) and cfg.load_cache:
        logger.info(f"[Graph] loading saved edges from {edges_pkl_file} ...")
        edges = utils.load_pickle(edges_pkl_file)
    else:
        logger.info(f"[Graph] loading edges from {edges_json_file} ...")
        edges = utils.load_json(edges_json_file)
        if cfg.save_cache:
            logger.info(f"[Graph] saving edges to {edges_pkl_file} ...")
            utils.save_pickle(edges, edges_pkl_file)
    logger.info(f"[Graph] entity-entity graph loads {len(edges)} (one-way) edges.")

    # build entity graph (with edges filtered by threshold)
    graph_pkl_file = f"{cache_dir}/graph.pkl"
    if os.path.exists(graph_pkl_file) and cfg.load_cache:
        graph = utils.load_pickle(graph_pkl_file)
    else:
        dg = dgl.DGLGraph()  # note: dgl uses digraph as default!
        dg.add_nodes(len(nodes))

        edges_src, edges_tgt, edges_w = [], [], []
        for e in edges:
            if e["weight"] > cfg.edge_weight_threshold:
                edges_src.append(e["node1"]["id"])
                edges_tgt.append(e["node2"]["id"])
                edges_w.append(e["weight"])
        dg.add_edges(edges_src, edges_tgt, data={"w": torch.FloatTensor(edges_w)})
        graph = dgl.to_bidirected(dg)  # undigraph
        if cfg.save_cache:
            utils.save_pickle(graph, graph_pkl_file)
    logger.info(f"[Graph] entity-entity graph keeps {graph.number_of_edges()} (bi-way) edges.")

    # load span candidates
    span_cands_pkl_file = f"{cache_dir}/span_cands.pkl"
    if os.path.exists(span_cands_pkl_file) and cfg.load_cache:
        span_cands, total_n_edges, actual_n_edges = utils.load_pickle(span_cands_pkl_file)
    else:
        span_cands, total_n_edges, actual_n_edges = utils.load_span_cands(
            f"{graph_dir}/candidates", cfg.edge_weight_threshold, train_size=cfg.train_size,
            n_cand=cfg.graph_neighbors[0]
        )  # include entities for training
        if cfg.save_cache:
            utils.save_pickle(
                (span_cands, total_n_edges, actual_n_edges),
                span_cands_pkl_file
            )
    logger.info(f"[Graph] span-entity graph loads {total_n_edges} (one-way) edges.")
    logger.info(f"[Graph] span-entity graph keeps {actual_n_edges} (one-way) edges.")

    # load node candidates (entity)
    node_cands_pkl_file = f"{cache_dir}/node_cands.pkl"
    if os.path.exists(node_cands_pkl_file) and cfg.load_cache:
        node_cands = utils.load_pickle(node_cands_pkl_file)
    else:
        node_cands = utils.node_nearest_cands(nodes, edges, cfg.n_neighbor)
        if cfg.save_cache:
            utils.save_pickle(node_cands, node_cands_pkl_file)

    # load node_str2idx -- Nodes: OrderedDict["id":Int, "tokens":List[Str], "type":Str]
    node_str2id_pkl_file = f"{cache_dir}/node_str2idx.pkl"
    if os.path.exists(node_str2id_pkl_file) and cfg.load_cache:
        node_str2idx = utils.load_pickle(node_str2id_pkl_file)
    else:
        node_str2idx = dict()
        for n in nodes:
            phrase = " ".join(n["tokens"] + [n["type"]])  # without 'type', there is repetition
            assert phrase not in node_str2idx, logger.info(phrase)
            node_str2idx[phrase] = n["id"]
        if cfg.save_cache:
            utils.save_pickle(node_str2idx, node_str2id_pkl_file)

    # load node2type_idx
    node2type_idx_pkl_file = f"{cache_dir}/node2type_idx.pkl"
    if os.path.exists(node2type_idx_pkl_file) and cfg.load_cache:
        node2type_idx = utils.load_pickle(node2type_idx_pkl_file)
    else:
        node2type_idx = dict()
        for n in nodes:
            node2type_idx[n["id"]] = cfg.entity_type2idx[n["type"]]
        if cfg.save_cache:
            utils.save_pickle(node2type_idx, node2type_idx_pkl_file)

    return nodes, edges, graph, span_cands, node_cands, \
           node_str2idx, node2type_idx,

def load_node_data(cfg):
    cache_dir = cfg.cache_dir

    def _set_char_ids(token):
        char_ids = []
        for c in token:
            if c in cfg.char2idx.keys():
                char_ids.append(cfg.char2idx[c])
            else:
                char_ids.append(cfg.char2idx["<unk>"])
        return char_ids

    def _set_word_ids(token):
        if token in cfg.word2idx.keys():
            return cfg.word2idx[token]
        else:
            return cfg.word2idx["<unk>"]

    # load node2char_ids, node2word_ids, node2lm_spans, node2word_strs
    node2char_ids = None
    node2word_ids = None
    node2lm_spans = None
    node2word_strs = None
    node2size = None
    node2char_ids_pkl_file = f"{cache_dir}/node2char_ids.pkl"
    node2word_ids_pkl_file = f"{cache_dir}/node2word_ids.pkl"
    node2lm_spans_pkl_file = f"{cache_dir}/node2lm_spans.pkl"
    node2word_strs_pkl_file = f"{cache_dir}/node2word_strs.pkl"
    node2size_pkl_file = f"{cache_dir}/node2size.pkl"
    if os.path.exists(node2word_strs_pkl_file) and \
        os.path.exists(node2word_ids_pkl_file) and \
        os.path.exists(node2lm_spans_pkl_file) and \
        os.path.exists(node2word_strs_pkl_file) and \
        os.path.exists(node2size_pkl_file) and \
        cfg.load_cache:
        if cfg.use_char_encoder:
            node2char_ids = utils.load_pickle(node2char_ids_pkl_file)
        if cfg.use_word_encoder:
            node2word_ids = utils.load_pickle(node2word_ids_pkl_file)
        node2lm_spans = utils.load_pickle(node2lm_spans_pkl_file)
        node2word_strs = utils.load_pickle(node2word_strs_pkl_file)
        node2size = utils.load_pickle(node2size_pkl_file)
    else:
        if cfg.use_char_encoder:
            node2char_ids = dict()
            for n in nodes:
                node2char_ids[n["id"]] = tuple([_set_char_ids(token) for token in n["tokens"]])

        if cfg.use_word_encoder:
            node2word_ids = dict()
            for n in nodes:
                node2word_ids[n["id"]] = tuple([_set_word_ids(token) for token in n["tokens"]])

        if cfg.use_lm_embed:
            node2lm_spans = dict()
            node2word_strs = dict()
            for n in nodes:
                node2lm_spans[n["id"]] = tuple([len(token) for token in n["tokens"]])
                node2word_strs[n["id"]] = tuple(n["tokens"])

        node2size = dict()
        for n in nodes:
            node2size[n["id"]] = len(n["tokens"])

        if cfg.save_cache:
            utils.save_pickle(node2char_ids, node2char_ids_pkl_file)
            utils.save_pickle(node2word_ids, node2word_ids_pkl_file)
            utils.save_pickle(node2lm_spans, node2lm_spans_pkl_file)
            utils.save_pickle(node2word_strs, node2word_strs_pkl_file)
            utils.save_pickle(node2size, node2size_pkl_file)

    return node2char_ids, node2word_ids, node2lm_spans, node2word_strs, node2size


def process_data(cfg, span_cands, node_cands, graph):
    cache_dir = cfg.cache_dir

    label2mode = {
        "train": NER_Dataset.Train_Mode,
        "valid": NER_Dataset.Eval_Mode,
        "test": NER_Dataset.Eval_Mode
    }
    def _load_dataset(data_label):
        data_pkl_file = f"{cache_dir}/{data_label}.pkl"
        if os.path.exists(data_pkl_file) and cfg.load_cache:
            dataset = utils.load_pickle(data_pkl_file)
        else:
            dataset = NER_Dataset.read_data_from_json(
                cfg=cfg,
                json_file_path=f"{cfg.data_dir}/{data_label}.json", dataset_label=data_label,
                entity_type2idx=cfg.entity_type2idx, tokenizer=cfg.tokenizer,
                span_cands=span_cands[data_label], node_cands=node_cands,
                node_str2idx=cfg.node_str2idx, idx2entity_type=cfg.idx2entity_type,
                graph=graph, node2type_idx=cfg.node2type_idx
            )
            dataset.switch_mode(label2mode[data_label])
            if cfg.save_cache:
                utils.save_pickle(dataset, data_pkl_file)

        return dataset

    train_data = _load_dataset("train")
    valid_data = _load_dataset("valid")
    test_data = _load_dataset("test")

    return train_data, valid_data, test_data

def get_train_itr(cfg, train_data, is_grouped=True):
    train_itr = torch.utils.data.DataLoader(
        train_data, batch_size=cfg.max_sentences,
        shuffle=cfg.shuffle, drop_last=True,
        num_workers=cfg.sampling_processes,
        collate_fn=sampling.collate_fn_padding
    )

    if is_grouped:
        cur_train_itr = Grouped_Iterator(iter(train_itr), len(train_itr), cfg.update_freq)
        return cur_train_itr
    else:
        return train_itr

def evaluate(
        cfg,
        model, lr_scheduler, optimizer,
        checker, train_logger,
        valid_data, test_data,
        epoch, num_updates, cur_lr, total_norm,
        no_incre_on_val, max_val_f,
        best_val, best_test_by_val, best_test,
        entity_types,
        at_epoch=False, at_update=False
):
    assert at_epoch ^ at_update

    def _evaluate(data, model, data_label):
        data_itr = torch.utils.data.DataLoader(
            data, batch_size=cfg.eval_max_sentences,
            shuffle=False, drop_last=False,
            num_workers=cfg.sampling_processes,
            collate_fn=sampling.collate_fn_padding
        )

        gt_entities = []
        for doc in data.documents:
            cur_gt_entities = [entity.as_tuple() for entity in doc.entities]
            gt_entities.append(cur_gt_entities)

        model.eval()

        log_outputs = []
        start_idx, end_idx = 0, 0

        if cfg.show_infer_speed:
            infer_time = 0

        for sample in data_itr:
            sample = utils.move_to_cuda(sample)
            end_idx += int(sample["entity_spans"].shape[0])
            loss, log_output, _infer_time = model.evaluate(sample, gt_entities=gt_entities[start_idx:end_idx])
            log_outputs.append(log_output)
            start_idx = end_idx

            if cfg.show_infer_speed:
                infer_time += _infer_time
        if cfg.show_infer_speed:
            print("infer speed (t/s): ", int(model.test_n_words / infer_time))

        log_output = model.aggregate_log_outputs(log_outputs)

        if cfg.do_extra_eval:
            logger.info(f"saving log: {data_label}_{epoch}_{num_updates}.json")
            os.makedirs(f"{cfg.save_pred_dir} {train_logger.start_t}", exist_ok=True)
            model.save_extra_eval_data(
                file=f"{cfg.save_pred_dir} {train_logger.start_t}/"
                     f"{data_label}_{epoch}_{num_updates}.json",
                log_output=log_output,
            )

        return log_output

    valid_log_output = _evaluate(valid_data, model, 'valid')
    test_log_output = _evaluate(test_data, model, 'test')

    # judge overfitting by nll loss in valid set
    save_best = False
    if valid_log_output["f"] > max_val_f:
        max_val_f = valid_log_output["f"]
        save_best = True
        train_logger.set_best_epoch_update(epoch, num_updates)

    # checker.save_checkpoint(epoch, num_updates, model, lr_scheduler, optimizer, save_best)

    if at_epoch:
        train_logger.print_at_epoch(epoch, num_updates, cur_lr, total_norm, valid_log_output, test_log_output)
    if at_update:
        train_logger.print_at_update(epoch, num_updates, cur_lr, total_norm, valid_log_output, test_log_output)

    # for self-test
    def _assign_dict(d1, d2):
        d1["p"] = d2["p"]
        d1["r"] = d2["r"]
        d1["f"] = d2["f"]

    if best_val["f"] < valid_log_output["f"]:
        if at_epoch: no_incre_on_val = 0
        _assign_dict(best_val, valid_log_output)
        _assign_dict(best_test_by_val, test_log_output)
    else:
        if at_epoch: no_incre_on_val += 1

    if best_test["f"] < test_log_output["f"]: _assign_dict(best_test, test_log_output)

    val_f = valid_log_output["f"]

    return no_incre_on_val, max_val_f, best_val, best_test_by_val, best_test, val_f

# def get_parameters(cfg, model):
#     params = [
#         {"params": model.parameters(), "lr": cfg.lr},
#     ]
#     # print(model.parameters())
#     # exit(0)
#
#     return params

def get_parameters(cfg, model):
    # for pretrained language model
    if cfg.use_lm_embed and hasattr(model, "lm"):
        lm_params = model.lm.parameters()
        lm_param_ids = list(map(id, model.lm.parameters()))
        other_params = list(filter(lambda p: id(p) not in lm_param_ids, model.parameters()))
        params = [
            {"params": lm_params, "lr": cfg.lr},
            {"params": other_params, "lr": cfg.other_lr},
        ]
    else:
        params = [
                {"params": model.parameters(), "lr": cfg.other_lr},
            ]
            # print(model.parameters())
            # exit(0)

    return params

if __name__ != "__main__":
    exit(0)

args = get_train_arguments()
logger.info(args)

cfg = Config(args)
utils.set_seed(cfg.seed, cfg.use_cuda)

# Load graph
span_cands = {"train": None, "valid": None, "test": None}
node_cands, graph = None, None
if cfg.use_gcn:
    nodes, edges, graph, span_cands, node_cands, \
    node_str2idx, node2type_idx = load_graph_data(cfg)
    cfg.num_nodes = len(nodes)
    cfg.graph = graph
    cfg.node_str2idx = node_str2idx
    cfg.node2type_idx = node2type_idx

# Process data
train_data, valid_data, test_data = process_data(cfg, span_cands, node_cands, graph)
cfg.get_num_update_epoch(data_size=len(train_data))
# bpe2idx = train_data.get_bpe2idx()
if cfg.use_gcn:
    cfg.node_idx2bpe_idx = train_data.build_node_idx2bpe_idx(
        node_str2idx=cfg.node_str2idx, idx2entity_type=cfg.idx2entity_type
    )
    train_data.set_node_idx2bpe_idx(cfg.node_idx2bpe_idx)
    valid_data.set_node_idx2bpe_idx(cfg.node_idx2bpe_idx)
    test_data.set_node_idx2bpe_idx(cfg.node_idx2bpe_idx)

if cfg.use_char_encoder:
    cfg.char2idx, cfg.idx2char = train_data.build_char2idx(
        special_tokens=cfg.special_tokens
    )
    n_ids1, n_unk_ids1 = train_data.set_char_ids(cfg.char2idx)
    n_ids2, n_unk_ids2 = valid_data.set_char_ids(cfg.char2idx)
    n_ids3, n_unk_ids3 = test_data.set_char_ids(cfg.char2idx)
    logger.info(f"Using char encoder (char ids): "
    f"  [Train] total={n_ids1} unk={n_unk_ids1} ratio={round(100*n_unk_ids1/n_ids1, 2)}%"
    f"  [Valid] total={n_ids2} unk={n_unk_ids2} ratio={round(100*n_unk_ids2/n_ids2, 2)}%"
    f"  [Test] total={n_ids3} unk={n_unk_ids3} ratio={round(100*n_unk_ids3/n_ids3, 2)}%"
    )

if cfg.use_word_encoder:
    cfg.load_word_embed(
        file_path=cfg.wv_file, word_dim=cfg.word_embed_dim,
        special_tokens=cfg.special_tokens
    )
    n_ids1, n_unk_ids1 = train_data.set_word_ids(cfg.word2idx)
    n_ids2, n_unk_ids2 = valid_data.set_word_ids(cfg.word2idx)
    n_ids3, n_unk_ids3 = test_data.set_word_ids(cfg.word2idx)
    logger.info(f"Using word encoder (word ids): "
    f"  [Train] total={n_ids1} unk={n_unk_ids1} ratio={round(100*n_unk_ids1/n_ids1, 2)}%"
    f"  [Valid] total={n_ids2} unk={n_unk_ids2} ratio={round(100*n_unk_ids2/n_ids2, 2)}%"
    f"  [Test] total={n_ids3} unk={n_unk_ids3} ratio={round(100*n_unk_ids3 / n_ids3, 2)}%"
    )

if cfg.use_gcn:
    node2char_ids, node2word_ids, node2lm_spans, node2word_strs, node2size = load_node_data(cfg)
    cfg.node2char_ids = node2char_ids
    cfg.node2word_ids = node2word_ids
    cfg.node2lm_spans = node2lm_spans
    cfg.node2word_strs = node2word_strs
    cfg.node2size = node2size

# Model
model = build_model(cfg)
logger.info(model)


# Checker: managing checkpoints
checker = Train_Checker(cfg)
checker.load_model(model)

if cfg.use_cuda:
    model = model.cuda()

optimizer = build_optimizer(cfg, params=get_parameters(cfg, model))
lr_scheduler = build_lr_scheduler(cfg, optimizer)

checker.load_optimizer_state(optimizer)
checker.load_lr_scheduler_state(lr_scheduler)
checker.delete_model_state()

# Logger: responsible for logging and printing
train_logger = Train_Logger(cfg)
logger.info(f"Saving dir: {cfg.save_pred_dir} {train_logger.start_t}")

# Train model
start_epoch = checker.start_epoch()
num_updates = checker.num_updates()

max_update = cfg.max_update or np.inf
cur_lr = optimizer.get_lr()

no_incre_on_val = 0
max_val_f = -np.inf
best_val = {"p": 0, "r": 0, "f": 0}
best_test_by_val = {"p": 0, "r": 0, "f": 0}
best_test = {"p": 0, "r": 0, "f": 0}

pbar = tqdm(total=cfg.t_steps, leave=False, dynamic_ncols=True, initial=num_updates)

for epoch in range(start_epoch, cfg.max_epoch + 1):
    if num_updates > max_update or cur_lr < cfg.min_lr:
        break

    train_data.switch_mode(NER_Dataset.Train_Mode)
    train_itr = get_train_itr(cfg, train_data, is_grouped=True)
    train_logger.reset_before_epoch(len(train_itr))

    tmp_idx = 0
    for samples in train_itr:
        # if tmp_idx <= 65 and not cfg.debug:
        #     pbar.update(1)
        #     tmp_idx += 1
        #     continue
        pbar.update(1)
        tmp_idx += 1

        model.train()
        model.zero_grad()

        log_outputs = []
        for sample in samples:
            sample = utils.move_to_cuda(sample)
            loss, log_output = model(sample)
            loss.backward()
            log_outputs.append(log_output)

        log_output = model.aggregate_log_outputs(log_outputs)

        # update model parameters
        model.clip_gradient()
        optimizer.step()
        num_updates += 1

        # update learning rate
        cur_lr = lr_scheduler.step_update(num_updates)  # step after each update

        # log the outputs
        train_logger.add_log_output(log_output)
        total_norm = utils.get_grad_norm(model)
        if cfg.evaluate_per_update > 0 and num_updates % cfg.evaluate_per_update == 0:
            no_incre_on_val, max_val_f, best_val, \
            best_test_by_val, best_test, val_f = evaluate(
                cfg,
                model, lr_scheduler, optimizer,
                checker, train_logger,
                valid_data, test_data,
                epoch, num_updates, cur_lr, total_norm,
                no_incre_on_val, max_val_f,
                best_val, best_test_by_val, best_test,
                entity_types=cfg.idx2entity_type,
                at_update=True
            )
            train_logger.save_log()
        sys.stdout.flush()

    total_norm = utils.get_grad_norm(model)
    no_incre_on_val, max_val_f, best_val, best_test_by_val, best_test, val_f = evaluate(
        cfg,
        model, lr_scheduler, optimizer,
        checker, train_logger,
        valid_data, test_data,
        epoch, num_updates, cur_lr, total_norm,
        no_incre_on_val, max_val_f,
        best_val, best_test_by_val, best_test,
        entity_types=cfg.idx2entity_type,
        at_epoch=True
    )

    train_logger.save_log()

    cur_lr = lr_scheduler.step_epoch(epoch=epoch, val_loss=-val_f)  # step at epoch

    if (
        no_incre_on_val >= cfg.max_no_incre_on_valid or
        epoch == cfg.max_epoch or
        num_updates > max_update or
        cur_lr < cfg.min_lr
    ):
        train_logger.print_and_save_best_result(cfg, best_val, best_test_by_val, best_test)
        break

sys.stdout.flush()


















