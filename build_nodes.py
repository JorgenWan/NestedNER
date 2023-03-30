import os
import math
import json
from tqdm import tqdm
from orderedset import OrderedSet
from data.node import Node


def read_json_data_for_nodes(json_data_file, existing_nodes, train_ratio):
    """
    json_data_file: json
    Read a .json data file.
    Return: each span is a str of tokens, seperated by ' '
    """

    with open(json_data_file, 'r') as f:
        data = json.load(f)

    total_train_size = len(data)
    train_size = math.ceil(total_train_size * train_ratio)
    data = data[:train_size]

    n_nodes_before_reading = len(existing_nodes)
    for inst in tqdm(data, desc="json_data_file"):
        for e in inst["entities"]:
            existing_nodes.add(Node(tokens=e["tokens"], type=e["type"]))
    n_nodes_after_reading = len(existing_nodes)

    print("total train data size:", total_train_size)
    print("used train data size:", train_size)
    print("number of nodes:", int(n_nodes_after_reading - n_nodes_before_reading))

    return existing_nodes

def save_nodes_to_json_file(nodes, save_node_file):
    res = []
    for i, node in enumerate(nodes):
        res.append(
            {
                "id": i,
                "tokens": node.tokens,
                "type": node.type
            }
        )

    with open(save_node_file, 'w') as f:
        json.dump(res, f, ensure_ascii=False, indent=1)

def build_nodes(train_file, save_node_file, train_ratio):
    nodes = OrderedSet()
    nodes = read_json_data_for_nodes(train_file, nodes, train_ratio)
    save_nodes_to_json_file(nodes, save_node_file)

if __name__ == "__main__":
    data_dir = f"/NAS2020/Workspaces/NLPGroup/juncw/database/NER/nested_ner"
    data_names = ["ace04", "ace05", "genia"]
    debug = False

    if debug:
        train_ratio = 0.01
    else:
        train_ratio = 1.0

    for data_name in data_names:
        print(f"{data_name}")

        train_file = f"{data_dir}/{data_name}/train.json"

        if debug:
            save_node_file = f"{data_dir}/{data_name}/Graph-debug/nodes.json"
            os.makedirs(f"{data_dir}/{data_name}/Graph-debug", exist_ok=True)
        else:
            save_node_file = f"{data_dir}/{data_name}/Graph/nodes.json"
            os.makedirs(f"{data_dir}/{data_name}/Graph", exist_ok=True)

        build_nodes(
            train_file=train_file, save_node_file=save_node_file, train_ratio=train_ratio
        )
        print("\n")












