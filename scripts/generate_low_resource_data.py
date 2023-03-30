import os
import json
import random
import argparse

from shutil import copyfile

import sys
sys.path.append("..")
from build_nodes import build_nodes_func
from build_edges import build_edges_func
from build_candidates import build_candidates_func

def gen_low_res(in_train_file, out_train_file, ratio):
    with open(in_train_file, 'r') as f:
        data = json.load(f)

    data_size = int(len(data) * ratio)
    new_data = random.sample(data, data_size)

    with open(out_train_file, 'w') as f:
        json.dump(new_data, f, ensure_ascii=False, indent=1)

if __name__ == "__main__":

    # for candidates
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--max_span_size', default=10, type=int)
    parser.add_argument('--cand_size', default=10, type=int)
    parser.add_argument('--num_process', default=48, type=int)
    args, _ = parser.parse_known_args()
    print(args)

    data_dir = f"/NAS2020/Workspaces/NLPGroup/juncw/database/NER/Flat_NER/SpanNER_en"
    ratios = [0.05, 0.1, 0.3]

    data_names = [ "genia"
        # "scierc", "conll04",  # entity-relation dataset
        # "ace04", "ace05", "genia",  # nested NER dataset
        # "conll03",  # English flat NER dataset
        # "weibo", "resume", "ontonote4", "msra",
        # "people_daily", "ecommerce"  # Chinese flat NER dataset
    ]

    for ratio in ratios:
        for data_name in data_names:
            print(f"\n{data_name} {ratio}")

            low_res_dir = f"{data_dir}/{data_name}/low_res_{ratio}"
            graph_dir = f"{data_dir}/{data_name}/low_res_{ratio}/Graph"
            cand_dir = f"{graph_dir}/candidates"

            nodes_file = f"{graph_dir}/nodes.json"
            edges_file = f"{graph_dir}/edges.json"

            orig_train_file = f"{data_dir}/{data_name}/train.json"
            orig_valid_file = f"{data_dir}/{data_name}/valid.json"
            orig_test_file = f"{data_dir}/{data_name}/test.json"
            orig_entity_types_file = f"{data_dir}/{data_name}/entity_types.json"

            train_file = f"{low_res_dir}/train.json"
            valid_file = f"{low_res_dir}/valid.json"
            test_file = f"{low_res_dir}/test.json"
            entity_types_file = f"{low_res_dir}/entity_types.json"

            save_cand_train_file = f"{graph_dir}/candidates/train"
            save_cand_valid_file = f"{graph_dir}/candidates/valid"
            save_cand_test_file = f"{graph_dir}/candidates/test"

            os.makedirs(low_res_dir, exist_ok=True)
            gen_low_res(orig_train_file, train_file, ratio)
            copyfile(orig_valid_file, valid_file)
            copyfile(orig_test_file, test_file)
            copyfile(orig_entity_types_file, entity_types_file)

            os.makedirs(graph_dir, exist_ok=True)
            build_nodes_func(train_file=train_file, save_node_file=nodes_file)
            build_edges_func(node_file=nodes_file, save_edge_file=edges_file)

            os.makedirs(cand_dir, exist_ok=True)
            build_candidates_func(
                args=args, nodes_file=nodes_file,
                train_file=train_file, valid_file=valid_file, test_file=test_file,
                save_cand_train_file=save_cand_train_file,
                save_cand_valid_file=save_cand_valid_file,
                save_cand_test_file=save_cand_test_file
            )


