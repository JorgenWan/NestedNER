import json


json_file_path = "/NAS2020/Workspaces/NLPGroup/juncw/database" \
                 "/NER/nested_ner/ace04/train.json"

sent_len = [3, 3]
n_entity = [0, 100]
n_type = [0, 100]

with open(json_file_path, "r") as f:
    data = json.load(f)

for d in data:
    types = set()
    for e in d["entities"]:
        types.add(e["type"])

    if (sent_len[0] <= len(d["tokens"]) <= sent_len[1] and \
        n_entity[0] <= len(d["entities"]) <= n_entity[1] and \
        n_type[0] <= len(types) <= n_type[1]):
        print(f'sent_id: {d["sent_id"]} \n'
              f'sent: {" ".join(d["tokens"])} \n'
              f'entities: {d["entities"]} \n')

