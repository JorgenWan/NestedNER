# 1. create char_dict.txt, label_dict.txt, seg_dict.txt, bichar_dict.txt, with form ``word freq``
# 2. convert data into *.pkl form by dict, with form ``[char_ids, label_ids, seg_ids, bichar_ids]``

import os
import numpy as np
import pickle as pkl
from tqdm import tqdm
from collections import OrderedDict


remove_middle = False

# data_names = ["resume", "weibo", "ontonote4", "msra", "people_daily", "synthe", "synthe2"] #
data_names = ["resume", "weibo", "ontonote4", "msra", "people_daily"]

embed_dir = "/NAS2020/Workspaces/NLPGroup/juncw/" \
                "summer_dataset/reduce_emb_dataset/pretrain_embed"

bert_dict_file = f"{embed_dir}/bert-base-chinese-vocab.txt"

for data_name in data_names:
    print(data_name)
    data_dir = f"/NAS2020/Workspaces/NLPGroup/juncw/summer_dataset/reduce_emb_dataset/{data_name}"
    train_file = f"{data_dir}/train.txt.clip"
    valid_file = f"{data_dir}/dev.txt.clip"
    test_file = f"{data_dir}/test.txt.clip"

    if not remove_middle:
        save_dir = f"{data_dir}/span_ner_bin"
    else:
        save_dir = f"{data_dir}/span_ner_no_mid_bin"
    os.makedirs(save_dir, exist_ok=True)


# 1. create char_dict.txt, label_dict.txt, seg_dict.txt, bichar_dict.txt,
# with form ``word freq``, sorted by freq in descending order
# note that char_dict.txt is bert_dict.txt, thus not sorted
# if True:
    def process_char(c, char_dict):
        """this char_dict is bert_dict"""
        if c.isdigit():
            return "0"
        if c in char_dict:
            assert c != ""
            return c
        c = c.replace("“", "\"").replace("”", "\"").replace("—", "-").replace("…", "...")
        if c in char_dict:
            assert c != ""
            return c
        c = "##"+c
        if c in char_dict:
            assert c != ""
            return c
        return "<UNK>"

    char_dict = OrderedDict()
    label_dict = dict()
    seg_dict = dict()
    bichar_dict = dict()
    entity_type_dict = dict()

    # todo: bert_dict should not have encoding="utf-8"
    # todo: because this dict has some non-utf-8 symbols
    with open(bert_dict_file, "r") as f:
        lines = f.readlines()
        for i in range(len(lines)):
            if i == 0:
                char_dict["<PAD>"] = 0
                continue
            if i == 1:
                char_dict["<UNK>"] = 0
                continue
            char = lines[i].strip()
            # print(i, lines[i])
            if char == "":
                char = "null_symbol_wjc"
            char_dict[char] = 0

    total_char = 0
    with open(train_file, "r", encoding="utf-8") as f:
        last_char = None
        for line in f.readlines():
            line = line.rstrip()
            if line == "":
                last_char = None
                continue
            total_char += 1
            char, label, seg = line.split()
            if char.isdigit():
                char = "0"
            if last_char is not None:
                bichar = last_char + char
                if bichar not in bichar_dict.keys():
                    bichar_dict[bichar] = 1
                else:
                    bichar_dict[bichar] += 1
            last_char = char

            new_char = process_char(char, char_dict)
            assert new_char != ""
            char_dict[new_char] += 1

            if not remove_middle or not label.startswith("I"):
                if label not in label_dict.keys():
                    label_dict[label] = 1
                else:
                    label_dict[label] += 1

            if label.startswith("B") or label.startswith("S"):
                entity_type = label[2:]
                if entity_type not in entity_type_dict.keys():
                    entity_type_dict[entity_type] = 1
                else:
                    entity_type_dict[entity_type] += 1

            if seg not in seg_dict.keys():
                seg_dict[seg] = 1
            else:
                seg_dict[seg] += 1
    print("Targeted char ratio: {:.2f}".format(100 - 100 * char_dict["<UNK>"] / total_char))

    def save_dict(tmp_dict, file):
        with open(file, "w") as f:
            for k, v in tmp_dict.items():
                f.write(f"{k} {v}\n")

    def sort_dict_and_save(tmp_dict, file):
        sorted_dict = list(sorted(tmp_dict.items(), key = lambda x: -x[1]))
        with open(file, "w", encoding="utf-8") as f:
            for k, v in sorted_dict:
                f.write(f"{k} {v}\n")

    save_dict(char_dict, f"{save_dir}/char_dict.txt")
    sort_dict_and_save(label_dict, f"{save_dir}/label_dict.txt")
    sort_dict_and_save(seg_dict, f"{save_dir}/seg_dict.txt")
    sort_dict_and_save(bichar_dict, f"{save_dir}/bichar_dict.txt")
    sort_dict_and_save(entity_type_dict, f"{save_dir}/entity_type_dict.txt")

# 2. convert data into *.pkl form by dict,
# with form ``List[Tuple]: [char_ids, label_ids, seg_ids, bichar_ids]``
# if True:
    def load_ordered_dict(file, res_dict):
        with open(file, "r") as f:
            for i, line in enumerate(f.readlines()):
                if i < 2:
                    continue
                k, v = line.strip().split()
                res_dict[k] = len(res_dict)
        return res_dict

    def load_dict(file, res_dict):
        with open(file, "r", encoding="utf-8") as f:
            for line in f.readlines():
                k, v = line.strip().split()
                res_dict[k] = len(res_dict)
        return res_dict

    char_dict = load_ordered_dict(f"{save_dir}/char_dict.txt", OrderedDict({"<PAD>":0, "<UNK>":1}))
    label_dict = load_dict(f"{save_dir}/label_dict.txt", {"<PAD>":0, "<BOS>":1, "<EOS>":2})
    seg_dict = load_dict(f"{save_dir}/seg_dict.txt", {})
    bichar_dict = load_dict(f"{save_dir}/bichar_dict.txt", {"<PAD>":0, "<UNK>":1})

    def char2bichar(chars):
        bichars = []
        for i in range(len(chars)):
            if i == len(chars) - 1:
                bichar = chars[i]+"-null-"
            else:
                bichar = chars[i]+chars[i+1]
            bichars.append(bichar)
        return bichars

    def g(toks, tok2id):
        ids = []
        for tok in toks:
            if tok in tok2id.keys():
                ids.append(tok2id[tok])
            else:
                ids.append(tok2id["<UNK>"])
        return ids

    def convert_to_id(in_file, save_file):

        chars_list, labels_list, segs_list, bichars_list = [], [], [], []
        char_ids_list, label_ids_list, seg_ids_list, bichar_ids_list = [], [], [], []

        chars, labels, segs = [], [], []
        total_char = 0
        total_unk = 0
        with open(in_file, "r", encoding="utf-8") as f:
            for line in f.readlines():
                line = line.rstrip()
                if line == "":
                    chars = ['[CLS]'] + chars + ['[SEP]']
                    labels = ['O'] + labels + ['O']
                    segs = ['S'] + segs + ['S']

                    chars_list.append(chars)
                    labels_list.append(labels)
                    segs_list.append(segs)

                    char_ids_list.append(g(chars, char_dict))
                    label_ids_list.append(g(labels, label_dict))
                    seg_ids_list.append(g(segs, seg_dict))

                    bichars = char2bichar(chars)
                    bichars_list.append(bichars)
                    bichar_ids_list.append(g(bichars, bichar_dict))

                    chars, labels, segs = [], [], []
                    continue

                char, label, seg = line.split()
                new_char = process_char(char, char_dict)
                if new_char == "<UNK>":
                    total_unk += 1
                
                chars.append(new_char)

                if not remove_middle or not label.startswith("I"):
                    labels.append(label)
                else:
                    labels.append("O")

                segs.append(seg)

                total_char += 1

        result = list(zip(
            chars_list, labels_list, segs_list, bichars_list,
            char_ids_list, label_ids_list, seg_ids_list, bichar_ids_list
        ))
        with open(save_file, "wb") as f:
            pkl.dump(result, f)

        print("Targeted char ratio: {:.2f}".format(100 - 100 * total_unk / total_char))

    convert_to_id(train_file, f"{save_dir}/train.pkl")
    convert_to_id(valid_file, f"{save_dir}/valid.pkl")
    convert_to_id(test_file, f"{save_dir}/test.pkl")



