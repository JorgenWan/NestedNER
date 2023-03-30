import os
import json

def store_json_file(in_file, out_file, sent_start_id):
    """
    in_file: raw file, such as train.txt.clip
    out_file: json file
    """

    with open(in_file, 'r') as f:
        orig_data = json.load(f)

    new_data = []

    num_sentences = len(orig_data)
    for i in range(num_sentences):
        entities = []
        for entity in orig_data[i]["entities"]:
            b = entity["start"]
            e = entity["end"] - 1
            t = entity["type"]
            tokens = orig_data[i]["tokens"][b:e+1]
            entities.append(
                {
                    "type": t,
                    "start": b,
                    "end": e,
                    "tokens": tokens
                }
            )

        new_data.append(
            {
                "sent_id": i + sent_start_id,
                "tokens": orig_data[i]["tokens"],
                "entities": entities
            }
        )

    with open(out_file, 'w') as f:
        json.dump(new_data, f, ensure_ascii=False, indent=1)

    print(num_sentences)

    return num_sentences

data_dir = f"/NAS2020/Workspaces/NLPGroup/juncw/database/NER/Flat_NER/spert/data/datasets"
# digit2zero = True

for data_name in ["scierc", "conll04"]:
    print(data_name)

    train_file = f"{data_dir}/{data_name}/{data_name}_train.json"
    valid_file = f"{data_dir}/{data_name}/{data_name}_dev.json"
    test_file = f"{data_dir}/{data_name}/{data_name}_test.json"

    os.makedirs(f"{data_dir}/{data_name}/SpanNER", exist_ok=True)
    train_save_file = f"{data_dir}/{data_name}/SpanNER/train.json"
    valid_save_file = f"{data_dir}/{data_name}/SpanNER/valid.json"
    test_save_file = f"{data_dir}/{data_name}/SpanNER/test.json"

    num_train_sentences = store_json_file(
        train_file, train_save_file,
        sent_start_id=0,
    )
    num_valid_sentences = store_json_file(
        valid_file, valid_save_file,
        sent_start_id=num_train_sentences,
    )
    num_test_sentences = store_json_file(
        test_file, test_save_file,
        sent_start_id=num_train_sentences + num_valid_sentences,
    )

