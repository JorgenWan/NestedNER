import os
import json

def load_ner_file(file_path, digit2zero=True):
    """
    Load NER data from file_path.
    Return tokens_list, labels_list, segs_list
    """

    tokens_list = []
    labels_list = []

    tokens = []
    labels = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f.readlines():
            line = line.strip()
            if line == "":
                assert len(tokens) == len(labels)
                tokens_list.append(tokens)
                labels_list.append(labels)

                tokens = []
                labels = []
                continue

            char, label = line.split()
            if digit2zero and char.isdigit():
                char = "0"
            tokens.append(char)
            labels.append(label)

    return tokens_list, labels_list


def get_entities(tokens, labels):
    """
    Input a list of label, return a list of tuple
        (begin[int, inclusive], end[int, inclusive], type[str])
    """

    seq_len = len(labels)

    bet_list = []
    i = 0
    while True:
        if i >= seq_len:
            break
        if labels[i].startswith("B"):
            b = i
            while not labels[i].startswith("E"):
                i += 1
                assert i < seq_len, print(tokens, labels)
            e = i
            bet_list.append((b, e, labels[b][2:]))
        elif labels[i].startswith("S"):
            bet_list.append((i, i, labels[i][2:]))
        else:
            assert labels[i].startswith("O"), print(labels[i])
        i += 1

    return bet_list

def store_json_file(in_file, out_file, sent_start_id, digit2zero):
    """
    in_file: raw file, such as train.txt.clip
    out_file: json file
    """

    data = []

    # todo: seg labels are ignored
    tokens_list, labels_list = load_ner_file(in_file, digit2zero=digit2zero)

    num_sentences = len(tokens_list)
    for i in range(num_sentences):
        bet_list = get_entities(tokens_list[i], labels_list[i])
        data.append(
            {
                "sent_id": i + sent_start_id,
                "tokens": tokens_list[i],
                "entities": [
                    {
                        "type": t,
                        "start": b,
                        "end": e,
                        "tokens": tokens_list[i][b:e+1]
                    }
                    for b,e,t in bet_list
                ]
            }
        )

    with open(out_file, 'w') as f:
        json.dump(data, f, ensure_ascii=False, indent=1)

    return num_sentences


data_dir = "/NAS2020/Workspaces/NLPGroup/juncw/summer_dataset/reduce_emb_dataset/old_ner_2020_09_09/conll2003"

digit2zero = True

train_file = f"{data_dir}/bioes/train.txt"
valid_file = f"{data_dir}/bioes/valid.txt"
test_file = f"{data_dir}/bioes/test.txt"

os.makedirs(f"{data_dir}/json", exist_ok=True)
train_save_file = f"{data_dir}/json/train.json"
valid_save_file = f"{data_dir}/json/valid.json"
test_save_file = f"{data_dir}/json/test.json"

num_train_sentences = store_json_file(
    train_file, train_save_file,
    sent_start_id=0,
    digit2zero=digit2zero
)
num_valid_sentences = store_json_file(
    valid_file, valid_save_file,
    sent_start_id=num_train_sentences,
    digit2zero=digit2zero
)
num_test_sentences = store_json_file(
    test_file, test_save_file,
    sent_start_id=num_train_sentences + num_valid_sentences,
    digit2zero=digit2zero
)


