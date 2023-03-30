import json
import random


def get_nested_overlap_spans(spans):
    nested_spans, overlap_spans = [], []

    for i in range(len(spans)):
        i_start, i_end = spans[i][0], spans[i][1]
        for j in range(len(spans)):
            if j == i:
                continue
            j_start, j_end = spans[j][0], spans[j][1]

            if (j_start < i_start and j_end >= i_start and j_end < i_end) or \
               (j_start > i_start and j_start <= i_end and j_end > i_end) :
                overlap_spans.append(tuple(spans[i]))
                continue

            if (j_start <= i_start and j_end >= i_end) or \
               (j_start >= i_start and j_end <= i_end) :
                nested_spans.append(tuple(spans[i]))
                continue


    nested_spans = list(set(nested_spans))
    overlap_spans = list(set(overlap_spans))

    return nested_spans, overlap_spans

def calculate_prf(n_golds, n_preds, n_inters):
    p = (n_inters * 1.0 / n_preds) * 100 if n_preds != 0 else 0
    r = (n_inters * 1.0 / n_golds) * 100 if n_golds != 0 else 0
    f = 2.0 * p * r / (p + r) if (p + r) != 0 else 0

    p = round(p, 2)
    r = round(r, 2)
    f = round(f, 2)

    return p, r, f

# def eval_nested_overlap_ner(pred_file):
#     with open(pred_file, 'r') as f:
#         data = json.load(f)
#
#     nested_n_gold, nested_n_pred, nested_n_inter = 0, 0, 0
#     overlap_n_gold, overlap_n_pred, overlap_n_inter = 0, 0, 0
#
#     for d in data:
#         nested_spans, overlap_spans = get_nested_overlap_spans(d["gold_data"]["spans"])
#         pred_spans = [tuple(s) for s in d["pred_data"]["spans"]]
#
#         nested_spans = set(nested_spans)
#         overlap_spans = set(overlap_spans)
#         pred_spans = set(pred_spans)
#
#         nested_n_gold += len(nested_spans)
#         nested_n_pred += len(pred_spans)
#         nested_n_inter += len(nested_spans & pred_spans)
#
#         overlap_n_gold += len(overlap_spans)
#         overlap_n_pred += len(pred_spans)
#         overlap_n_inter += len(overlap_spans & pred_spans)
#
#     nested_prf = calculate_prf(nested_n_gold, nested_n_pred, nested_n_inter)
#     overlap_prf = calculate_prf(overlap_n_gold, overlap_n_pred, overlap_n_inter)
#
#     return nested_prf, overlap_prf

def eval_nested_ner(pred_file):
    with open(pred_file, 'r') as f:
        data = json.load(f)

    nested_n_gold, nested_n_pred, nested_n_inter = 0, 0, 0

    for d in data:
        nested_spans, overlap_spans = get_nested_overlap_spans(d["gold_data"]["spans"])
        pred_spans = [tuple(s) for s in d["pred_data"]["spans"]]

        nested_spans = set(nested_spans)
        pred_spans = set(pred_spans)

        nested_n_gold += len(nested_spans)
        nested_n_pred += len(pred_spans)
        nested_n_inter += len(nested_spans & pred_spans)

    nested_prf = calculate_prf(nested_n_gold, nested_n_pred, nested_n_inter)

    return nested_prf

if __name__ == "__main__":
    # "scierc", "conll04",  # entity-relation dataset
    # "ace04", "ace05", "genia", "nne",  # nested NER dataset
    # "conll03",  # English flat NER dataset
    # "weibo", "resume", "ontonote4", "msra",
    # "people_daily", "ecommerce"  # Chinese flat NER dataset

    data_dir = f"/NAS2020/Workspaces/NLPGroup/juncw/database/NER/nested_ner/__results"

    # pred_file = f"{data_dir}/ace04/span_bert_gcn/" \
    #             f"41_82 2021-11-07 14:00:47/valid_31_23994.json"  # ace04, spert
    # pred_file = f"{data_dir}/ace04/span_bert_gcn/" \
    #             f"41_15 2021-10-29 22:52:44/valid_48_37200.json"  # ace04, ours

    # pred_file = f"{data_dir}/ace05/span_bert_gcn/" \
    #             f"41_82 2021-11-07 14:00:47/valid_37_33670.json"  # ace05, spert
    # pred_file = f"{data_dir}/ace05/span_bert_gcn/" \
    #             f"41_61 2021-11-02 22:24:29/valid_36_32760.json"  # ace05, ours

    # pred_file = f"{data_dir}/genia/span_bert_gcn/" \
    #             f"41_82 2021-11-07 18:38:24/valid_21_39417.json"  # genia, spert
    pred_file = f"{data_dir}/genia/span_bert_gcn/" \
                f"41_18 2021-11-06 11:03:25/valid_8_15016.json"  # genia, spert

    # pred_file = f"{data_dir}/nne/span_bert_gcn/" \
    #             f"41_81 2021-11-07 18:45:30/valid_8_43456.json"  # nne, spert


    # nested_prf, overlap_prf = eval_nested_overlap_ner(pred_file)
    nested_prf = eval_nested_ner(pred_file)
    print(f"nested: {nested_prf}")
