import json
import random


def calculate_prf(n_golds, n_preds, n_inters):
    p = (n_inters * 1.0 / n_preds) * 100 if n_preds != 0 else 0
    r = (n_inters * 1.0 / n_golds) * 100 if n_golds != 0 else 0
    f = 2.0 * p * r / (p + r) if (p + r) != 0 else 0

    p = round(p, 2)
    r = round(r, 2)
    f = round(f, 2)

    return p, r, f

def eval_dif_len_ner(pred_file, max_len):
    with open(pred_file, 'r') as f:
        data = json.load(f)

    dif_len_n_gold = [0 for _ in range(max_len)]
    dif_len_n_pred = [0 for _ in range(max_len)]
    dif_len_n_inter = [0 for _ in range(max_len)]
    n_gold = 0

    for d in data:
        gold_spans = [tuple(s) for s in d["gold_data"]["spans"]]
        pred_spans = [tuple(s) for s in d["pred_data"]["spans"]]
        inter_spans = list(set(gold_spans) & set(pred_spans))

        n_gold += len(gold_spans)

        for s in gold_spans:
            if s[1]-s[0] >= max_len:
                continue
            dif_len_n_gold[s[1]-s[0]] += 1

        for s in pred_spans:
            if s[1]-s[0] >= max_len:
                continue
            dif_len_n_pred[s[1]-s[0]] += 1

        for s in inter_spans:
            if s[1]-s[0] >= max_len:
                continue
            dif_len_n_inter[s[1]-s[0]] += 1

    dif_len_prf = []
    for i in range(max_len):
        dif_len_prf.append(
            calculate_prf(dif_len_n_gold[i], dif_len_n_pred[i], dif_len_n_inter[i])
        )

    dif_len_ratio = []
    for i in range(max_len):
        dif_len_ratio.append(
            dif_len_n_gold[i] / n_gold
        )

    return dif_len_ratio, dif_len_prf

if __name__ == "__main__":
    # "scierc", "conll04",  # entity-relation dataset
    # "ace04", "ace05", "genia", "nne",  # nested NER dataset
    # "conll03",  # English flat NER dataset
    # "weibo", "resume", "ontonote4", "msra",
    # "people_daily", "ecommerce"  # Chinese flat NER dataset

    data_dir = f"/NAS2020/Workspaces/NLPGroup/juncw/database/NER/nested_ner/__results"

    # pred_file = f"{data_dir}/ace05/span_bert_gcn/" \
    #             f"41_61 2021-11-02 22:24:29/test_41_37310.json" # ace05

    pred_file = f"{data_dir}/ace05/span_bert_gcn/" \
                f"41_82 2021-11-07 14:00:47/test_37_33670.json" # ace05

    max_len = 10

    dif_len_ratio, dif_len_prf = eval_dif_len_ner(pred_file, max_len)

    for i in range(max_len):
        ratio = round(100 * dif_len_ratio[i], 2)
        p, r, f = dif_len_prf[i]
        print(f"len = {i+1}, ratio prf: {ratio} & {p} & {r} & {f}")
