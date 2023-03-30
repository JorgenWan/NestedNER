import json
from tqdm import tqdm
from transformers import BertTokenizer

def count_sent_len(datas_json_file, label, tokenizer):
    with open(datas_json_file, 'r') as f:
        datas = json.load(f)

    sent_num = len(datas)
    nested_sent_num = 0

    max_sent_len = max([len(d["tokens"]) for d in datas])
    total_sent_len = sum([len(d["tokens"]) for d in datas])
    num_ent = sum([len(d["entities"]) for d in datas])

    max_sent_bpe_num = 0

    # for d in tqdm(datas):
    #     bpe_ids = []
    #     for w in d["tokens"]:
    #         bpe_ids += tokenizer.encode(w, add_special_tokens=False)
    #     max_sent_bpe_num = max(max_sent_bpe_num, len(bpe_ids))

    total_char_len = 0
    num_word = 0
    max_char_len = 0 # in a word
    for d in datas:
        num_word += len(d["tokens"])
        for w in d["tokens"]:
            max_char_len = max(max_char_len, len(w))
            total_char_len += len(w)

    num_nested_ent = 0
    for d in datas:
        n_ent = len(d["entities"])
        flag_nested = False
        for i in range(n_ent):
            i_start, i_end = d["entities"][i]["start"], d["entities"][i]["end"]
            for j in range(n_ent):
                if j == i:
                    continue
                j_start, j_end = d["entities"][j]["start"], d["entities"][j]["end"]
                if (j_start <= i_start and j_end >= i_start or
                    j_start <= i_end and j_end >= i_end or
                    j_start >= i_start and j_end <= i_end):
                    num_nested_ent += 1
                    flag_nested = True
                    break
        if flag_nested:
            nested_sent_num += 1

    nested_sent_ratio = round(100 * nested_sent_num/sent_num, 1)
    avg_sent_len = round(total_sent_len/sent_num, 1)
    nested_ent_ratio = round(100 * num_nested_ent/num_ent, 1)

    max_ent_len = max([max([0] + [e["end"]-e["start"] + 1 for e in d["entities"]]) for d in datas])
    all_ent_len = sum([sum([e["end"]-e["start"] + 1 for e in d["entities"]]) for d in datas])
    avg_ent_len = round(all_ent_len / num_ent, 1)
    avg_char_len = round(total_char_len / num_word, 1)

    print(f"{label} "
          f"sent_num={sent_num} "
          f"nested_sent_num={nested_sent_num} "
          f"nested_sent_ratio={nested_sent_ratio} "
          f"avg_sent_len={avg_sent_len} "
          f"max_sent_len={max_sent_len} "
          f"num_ent={num_ent} "
          f"max_sent_len={max_sent_len} "
          f"num_nested_ent={num_nested_ent} "
          f"nested_ent_ratio={nested_ent_ratio} "
          f"avg_ent_len={avg_ent_len} "
          f"max_ent_len={max_ent_len} "
          f"max_char_len={max_char_len} "
          f"max_sent_bpe_num={max_sent_bpe_num} "
          f"total_sent_len={total_sent_len} "
          f"avg_char_len={avg_char_len} "
    )


if __name__ == "__main__":
    data_dir = f"/NAS2020/Workspaces/NLPGroup/juncw/database/NER/nested_ner"
    data_names = [ "ace04", "ace05", "genia", "nne"
        # "ace04",
        # "msra", "people_daily", "ecommerce",
        # "weibo", "resume", "ontonote4",  # Chinese flat NER dataset
        # "scierc", "conll04",  # entity-relation dataset
        # "ace04", "ace05", "genia", "nne", # nested NER dataset
        # "conll03",  # English flat NER dataset
    ]


    pretrained_model = {
        "ace04": "bert-base-cased",
        "ace05": "bert-base-cased",
        "genia": "dmis-lab/biobert-base-cased-v1.1",
        "nne": "bert-base-cased",
    }

    do_lower_case = {
        "ace04": False,
        "ace05": False,
        "genia": False,
        "nne": False,
    }

    for data_name in data_names:
        print(f"\n{data_name}")

        tokenizer = BertTokenizer.from_pretrained(
            pretrained_model[data_name], do_lower_case=do_lower_case[data_name]
        )

        count_sent_len(f"{data_dir}/{data_name}/train.json", "train", tokenizer)
        count_sent_len(f"{data_dir}/{data_name}/valid.json", "valid", tokenizer)
        count_sent_len(f"{data_dir}/{data_name}/test.json", "test", tokenizer)



