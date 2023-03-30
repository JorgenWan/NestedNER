import os
import json
from tqdm import tqdm
from orderedset import OrderedSet

class Node:
    def __init__(self, tokens, type):

def read_json_data_for_entities(json_data_file, existing_entities):
    """
    json_data_file: json
    Read a .json data file.
    Return: each span is a str of tokens, seperated by ' '
    """

    with open(json_data_file, 'r') as f:
        data = json.load(f)

    for inst in tqdm(data):
        entities = []
        for i in range(len(inst["entities"])):
            entities.append(
                {
                    "type":
                }
            )


        extract_spans(inst["tokens"], max_span_size)
        for span in spans:
            existing_spans.add(' '.join(span))
    return existing_spans

data_dir = f"//NAS2020/Workspaces/NLPGroup/juncw/database/NER/Flat_NER/SpanNER_en"
data_names = [
    "scierc", "conll04",  # entity-relation dataset
    "ace04", "ace05", "genia",  # nested NER dataset
    "conll03",  # English flat NER dataset
    "weibo", "resume", "ontonote4", "msra", "people_daily", "ecommerce"  # Chinese flat NER dataset
]

for data_name in data_names:
    print(data_name)

    train_file = f"{data_dir}/{data_name}/train.json"
    valid_file = f"{data_dir}/{data_name}/valid.json"
    save_node_file = f"{data_dir}/{data_name}/Graph/nodes.csv"

    os.makedirs(f"{data_dir}/{data_name}/Graph", exist_ok=True)

    entities = OrderedSet()
    entities = read_json_data_for_entities(train_file)
    entities = read_json_data_for_entities(valid_file)





