"""
Read the train/valid/test.json file and get entity types.
"""

import json

def get_entity_types(json_file_path):

    with open(json_file_path, "r") as f:
        data = json.load(f)

    entity_types = set()
    for doc in data:
        for e in doc["entities"]:
            entity_types.add(e["type"])
    return entity_types

data_dir = f"/NAS2020/Workspaces/NLPGroup/juncw/database/NER/Flat_NER/SpanNER_en"
data_names = [
    "scierc", "conll04",  # entity-relation dataset
    "ace04", "ace05", "genia",  # nested NER dataset
    "conll03",  # English flat NER dataset
    "weibo", "resume", "ontonote4", "msra", "people_daily", "ecommerce"  # Chinese flat NER dataset
]

for data_name in data_names:

    train_file = f"{data_dir}/{data_name}/train.json"
    valid_file = f"{data_dir}/{data_name}/valid.json"
    test_file = f"{data_dir}/{data_name}/test.json"

    save_entity_types_file = f"{data_dir}/{data_name}/entity_types.json"

    train_entity_types = get_entity_types(train_file)
    valid_entity_types = get_entity_types(valid_file)
    test_entity_types = get_entity_types(test_file)

    entity_types = list(train_entity_types | valid_entity_types | test_entity_types)

    print(f"{data_name}={entity_types}")

    with open(save_entity_types_file, 'w') as f:
        json.dump(entity_types, f, ensure_ascii=False, indent=1)