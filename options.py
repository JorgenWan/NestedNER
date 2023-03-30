import torch
import argparse

from models import ARCHI_REGISTRY, ARCHI_CONFIG_REGISTRY
from criterions import CRITERION_REGISTRY
from optimizers import OPTIMIZER_REGISTRY
from lr_schedulers import LR_SCHEDULER_REGISTRY

def add_dataset_arguments(parser):
    group = parser.add_argument_group('Dataset')
    group.add_argument('--data_dir',
                       type=str,
                       default='/NAS2020/Workspaces/NLPGroup/juncw/database/NER/Flat_NER/SpanNER_en/weibo',
                       help="directory to dataset, absolute path")
    group.add_argument('--shuffle', type=int, default=0)
    group.add_argument('--max_tokens', type=int, default=99999, help='maximum number of tokens in a batch')
    group.add_argument('--max_sentences', type=int, default=2, help='maximum number of sentences in a batch')
    group.add_argument('--eval_max_sentences', type=int, default=1, help='maximum number of sentences in a batch')
    group.add_argument('--max_length', type=int, default=250, help='maximum length of sentences')
    group.add_argument('--sampling_processes', default=0, type=int, help="for dataloader")
    group.add_argument('--load_cache', type=int, default=0)
    group.add_argument('--save_cache', type=int, default=0)
    group.add_argument('--train_ratio', type=float, default=0.01, help='train data ratio')
    group.add_argument('--valid_ratio', type=float, default=0.01, help='valid data ratio')
    group.add_argument('--test_ratio', type=float, default=0.01, help='test data ratio')

    group.add_argument('--max_n_word', type=int, default=125, help='maximum number of words in sentences')
    group.add_argument('--max_ent_n_word', type=int, default=64, help='maximum number of words in entities')
    group.add_argument('--max_word_len', type=int, default=64, help='maximum number of words in entities')


def add_model_arguments(parser):
    group = parser.add_argument_group('Model')
    group.add_argument('--archi',
                       choices=ARCHI_REGISTRY.keys(),
                       default="nner_slg",
                       help='model architecture')
    group.add_argument('--use_char_encoder', type=int, default=1)
    group.add_argument('--use_word_encoder', type=int, default=1)
    group.add_argument('--use_lm_embed', type=int, default=1)
    group.add_argument('--use_size', type=int, default=1)

    group.add_argument('--char_embed_dim', default=30, type=int)
    group.add_argument('--char_hid_dim', default=50, type=int)
    group.add_argument('--word_embed_dim', default=100, type=int) # the same as wv_file
    group.add_argument('--cls_embed_dim', default=1024, type=int) # the same as LM
    group.add_argument('--wc_lstm_hid_dim', default=200, type=int) # the same as LM
    group.add_argument('--sent_enc_dropout', default=0.5, type=float) # the same as LM
    group.add_argument('--lm_dim',
                       default=5120, type=int,
                       help="768/1024 for base/large model, 4096 for flair embeddings")

    group.add_argument('--wv_file', default="ACE04.glove.6B.100d.txt", type=str)
    group.add_argument('--lm_emb_path', default="ACE04.bert_large_uncased_flair.emb.pkl",  type=str)
    group.add_argument('--ent_lm_emb_path',
                       default="ACE04.ent_bert_large_uncased_flair.emb.pkl",
                       type=str)
    group.add_argument('--cased_char', type=int, default=1)
    group.add_argument('--cased_word', type=int, default=1)
    group.add_argument('--cased_lm', type=int, default=1)

    group.add_argument('--use_gcn', type=int, default=1)
    group.add_argument('--use_cls', type=int, default=0)

    group.add_argument('--debug', type=int, default=0)
    group.add_argument('--show_infer_speed', type=int, default=0)
    group.add_argument('--data_name', type=str)

    # graph
    group.add_argument('--edge_weight_threshold', default=1.0, type=float,
                        help='keep edges whose weight higher than the threshold')
    group.add_argument('--graph_neighbors', default="3 2", type=str,
                       help='number of graph neighbors for each hop from span')
    group.add_argument('--max_batch_nodes', default=3000, type=int)
    group.add_argument('--max_batch_edges', default=1000, type=int)

    group.add_argument('--concat_span_hid', type=int, default=0)
    group.add_argument('--use_graph_weight', action='store_true', default=False)
    group.add_argument('--graph_emb_method', default="mean", type=str, help="optional: mean, attn")

    group.add_argument('--alpha', default=0.1, type=float,
                        help='multi-task learning weight for graph node')

def add_criterion_arguments(parser):
    group = parser.add_argument_group('Criterion')
    group.add_argument('--criterion',
                       choices=CRITERION_REGISTRY.keys(),
                       default="cross_entropy",
                       help='Training Criterion')

def add_optimization_arguments(parser):
    group = parser.add_argument_group('Optimization')
    group.add_argument('--optimizer', default='adam', choices=OPTIMIZER_REGISTRY.keys())
    group.add_argument('--lr', default=0.01, type=float, help='learning rate')
    group.add_argument('--other_lr', default=1e-4, type=float)
    group.add_argument('--clip_norm', default=-1, type=float,
                       help='clip threshold of gradient norm, -1 means no clip')
    group.add_argument('--clip_value', default=-1, type=float,
                       help='clip threshold of gradient value, -1 means no clip')
    group.add_argument('--sentence_average', action='store_true', default=True,
                       help='normalize gradients by the number of sentences in a batch'
                            ' (default is to normalize by number of tokens)')
    group.add_argument('--update_freq', default=1, type=int, help='update parameters every N batches')
    group.add_argument('--min_lr', default=1e-10, type=float)
    group.add_argument('--max_epoch', default=100, type=int)
    group.add_argument('--max_update', default=100000, type=int)
    group.add_argument('--max_no_incre_on_valid', default=50, type=int,
                        help='if use linear_with_warmup scheduler, it is disabled')
    group.add_argument('--print_at_update', action='store_true',
                        help='print at each (model) update')

def add_lr_scheduler_arguments(parser):
    group = parser.add_argument_group('LR Scheduler')
    group.add_argument('--lr_scheduler',
                       choices=LR_SCHEDULER_REGISTRY.keys(),
                       default="linear_with_warmup",
                       help='Learning Rate Scheduler')

def add_checkpoint_arguments(parser):
    group = parser.add_argument_group('Checkpointing')
    group.add_argument('--load_checkpoint', action='store_true',
                       help='load checkpoint to train continuously')
    group.add_argument('--checkpoint_path', default='checkpoints', type=str,
                       help='checkpoint file path')
    group.add_argument('--save_dir', default='checkpoints', type=str,
                       help='directory to save checkpoints')

    group.add_argument('--keep_last_epochs', type=int, default=-1,
                       help='keep last n epoch checkpoints')
    group.add_argument('--keep_last_updates', type=int, default=-1,
                       help='keep the last n update checkpoints')

def get_specific_arguments(parser):
    args, _ = parser.parse_known_args()

    # model-specific configuration
    # only include attributes explicitly given or having default values
    if hasattr(args, "archi"):
        group = parser.add_argument_group('Model-specific', argument_default=argparse.SUPPRESS)
        ARCHI_REGISTRY[args.archi].add_args(group)

    if hasattr(args, "optimizer"):
        group = parser.add_argument_group('Optimizer-specific', argument_default=argparse.SUPPRESS)
        OPTIMIZER_REGISTRY[args.optimizer].add_args(group)

    if hasattr(args, "lr_scheduler"):
        group = parser.add_argument_group('LR_scheduler-specific', argument_default=argparse.SUPPRESS)
        LR_SCHEDULER_REGISTRY[args.lr_scheduler].add_args(group)

    if hasattr(args, "criterion"):
        group = parser.add_argument_group('Criterion-specific', argument_default=argparse.SUPPRESS)
        CRITERION_REGISTRY[args.criterion].add_args(group)

    # architecture-specific configuration
    args, _ = parser.parse_known_args()
    if hasattr(args, "archi"):
        ARCHI_CONFIG_REGISTRY[args.archi](args)

    return args

def post_process_arguments(args):

    if hasattr(args, "gcn_layers"):
        layers = [int(l) for l in args.gcn_layers.split(" ")]
        args.gcn_layers = layers
    if hasattr(args, "graph_neighbors"):
        graph_neighbors = [int(l) for l in args.graph_neighbors.split(" ")]
        args.graph_neighbors = graph_neighbors

        args.n_hop = len(graph_neighbors)
        args.n_neighbor = max(graph_neighbors)

        args.max_ee_graph_neighbors = max(args.graph_neighbors[1:])

    return args


def get_train_arguments():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--use_cpu', action='store_true', default=False)

    parser.add_argument('--save_log_file', type=str, help='file path to save log')
    parser.add_argument('--save_best_result_file', type=str, help='file path to the best result')

    # evaluation during training
    parser.add_argument('--evaluate_per_update', default=100, type=int,
                        help="-1 is no evaluation within a epoch")

    # eval low-frequency entities
    parser.add_argument('--do_extra_eval', type=int, default=0,
                        help="do extra evaluation, including saving "
                             "the predictions, subgraphs, training time, inference time, etc")
    parser.add_argument('--save_pred_dir', type=str)

    add_dataset_arguments(parser)
    add_model_arguments(parser)
    add_criterion_arguments(parser)
    add_lr_scheduler_arguments(parser)
    add_optimization_arguments(parser)
    add_checkpoint_arguments(parser)

    args = get_specific_arguments(parser)

    args = post_process_arguments(args)

    return args

def get_evaluate_arguments():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--use_cpu', action='store_true')
    parser.add_argument('--eval_each_entity_type', action='store_true')

    # checkpoint
    parser.add_argument('--checkpoint_path', type=str, help="absolute path")
    parser.add_argument('--save_log_file', type=str, help='file path to save log')

    add_dataset_arguments(parser)

    args = get_specific_arguments(parser)

    return args