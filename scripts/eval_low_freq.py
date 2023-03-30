import json
import random

def read_json_for_entity2freq(json_data_file):
    """
    json_data_file: json
    Read a .json data file.
    Return: each span is a str of tokens, seperated by ' '
    """

    entity2freq = dict()

    with open(json_data_file, 'r') as f:
        data = json.load(f)

    for inst in data:
        for e in inst["entities"]:
            entity_str = ' '.join(e["tokens"])
            if entity_str in entity2freq.keys():
                entity2freq[entity_str] += 1
            else:
                entity2freq[entity_str] = 1

    return entity2freq

def get_entity2freq_test(entity2freq_train, gold_file):
    entity2freq_test = dict() # entity freq in train set
    entity2freq_test_itself = dict() # entity freq in test set

    with open(gold_file, 'r') as f:
        data = json.load(f)

    for inst in data:
        for e in inst["entities"]:
            entity_str = ' '.join(e["tokens"])
            if entity_str in entity2freq_train.keys():
                entity2freq_test[entity_str] = entity2freq_train[entity_str]
            else:
                entity2freq_test[entity_str] = 0

            if entity_str in entity2freq_test_itself.keys():
                entity2freq_test_itself[entity_str] += 1
            else:
                entity2freq_test_itself[entity_str] = 1

    return entity2freq_test, entity2freq_test_itself

def get_low_freq_entity_set(entity2freq, freq_thresh, compare, do_lower):
    low_freq_entity_list = []
    for k, v in entity2freq.items():
        k_ = k.lower() if do_lower else k
        if compare == "=":
            if v == freq_thresh:
                low_freq_entity_list.append(k_)
        elif compare == "<=":
            if v <= freq_thresh:
                low_freq_entity_list.append(k_)
    low_freq_entity_set = set(low_freq_entity_list)
    return low_freq_entity_set

def eval_low_freq_entity(low_freq_entity_set, pred_file, entity2freq, entity2freq_itself, total_entity):
    with open(pred_file, 'r') as f:
        data = json.load(f)

    n_gold, n_inter = 0, 0
    n_graph_inter = 0
    for d in data:
        low_freq_entity = []
        for i in range(len(d["gold_data"]["spans"])):
            phrase = d["gold_data"]["phrases"][i]

            if phrase in low_freq_entity_set:
                low_freq_entity.append(d["gold_data"]["spans"][i])

                # if d["gold_data"]["spans"][i] not in d["pred_data"]["spans"]:
                #     gold_entity_type = d["gold_data"]["spans"][i][2]
                #     node_entity_types = d["gold_data"]["node_entity_types"][i]
                #     if len(node_entity_types) > 0:
                #         tmp_type = random.sample(node_entity_types, 1)[0]
                #         if gold_entity_type in node_entity_types:
                #         # if gold_entity_type == tmp_type:
                #             n_graph_inter += 1
                #             print(gold_entity_type, node_entity_types)


        low_freq_gold_entity = set([tuple(_) for _ in low_freq_entity])
        pred_entity = set([tuple(_) for _ in d["pred_data"]["spans"]])
        n_gold += len(low_freq_gold_entity)
        n_inter += len(low_freq_gold_entity & pred_entity)

    if n_gold > 0:
        recall = round(100 * n_inter / n_gold, 4)
        # graph_recall = round(100 * n_graph_inter / n_gold, 4)
    else:
        recall = 0
        # graph_recall = 0

    ratio = round(100 * n_gold / total_entity, 2)
    print(
        f"n_gold={n_gold} "
        f"n_inter={n_inter} "
        f"recall={recall} "
        f"ratio={ratio} "
        # f"graph_recall={graph_recall}"
    )


if __name__ == "__main__":
    # "scierc", "conll04",  # entity-relation dataset
    # "ace04", "ace05", "genia",  # nested NER dataset
    # "conll03",  # English flat NER dataset
    # "weibo", "resume", "ontonote4", "msra",
    # "people_daily", "ecommerce"  # Chinese flat NER dataset

    data_dir = f"/NAS2020/Workspaces/NLPGroup/juncw/database/NER/nested_ner"
    do_lower = False # for GENIA, tokenizer of BioBERT is strange (recover BPE uncased)

    # ace04
    data_name, data_label = "ace04", "test"
    pred_file = f"{data_dir}/__results/ace04/span_bert_gcn/" \
                f"41_15 2021-10-29 22:52:44/valid_48_37200.json" # ours, best
    # pred_file = f"{data_dir}/__results/ace04/span_bert_gcn/" \
    #             f"41_18 2021-11-03 20:58:34/valid_19_14725.json" # spert, best


    # ace05
    # data_name, data_label = "ace05", "test"
    # pred_file = f"{data_dir}/__results/ace05/span_bert_gcn/" \
    #             f"41_61 2021-11-02 22:24:29/test_41_37310.json"  # ace05, best
    # pred_file = f"{data_dir}/__results/ace05/span_bert_gcn/" \
    #             f"41_61 2021-11-02 22:24:29/test_18_16380.json"  # spert, best


    train_file = f"{data_dir}/{data_name}/train.json"
    gold_file = f"{data_dir}/{data_name}/{data_label}.json"

    # freq_threshs = list(range(5))
    freq_threshs = [0,1,2,3,4,5,1e6]
    compare = "<="

    entity2freq_train = read_json_for_entity2freq(train_file)

    entity2freq_eval, entity2freq_eval_itself = get_entity2freq_test(
        entity2freq_train=entity2freq_train,
        gold_file=gold_file
    )

    total_entity = sum([v for k, v in entity2freq_eval_itself.items()])

    for freq_thresh in freq_threshs:
        print(f"\n{data_name} freq_thresh {compare} {freq_thresh}")


        low_freq_entity_set = get_low_freq_entity_set(
            entity2freq=entity2freq_eval,
            freq_thresh=freq_thresh,
            compare=compare,
            do_lower=do_lower
        )

        eval_low_freq_entity(
            low_freq_entity_set=low_freq_entity_set,
            pred_file=pred_file,
            entity2freq=entity2freq_eval,
            entity2freq_itself=entity2freq_eval_itself,
            total_entity=total_entity,
        )

