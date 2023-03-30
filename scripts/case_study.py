# goal: find some sentences, in which the baseline model
#       makes the wrong prediction while our method
#       predicts correctly

import json

def load_json(file):
    with open(file, 'r') as f:
        data = json.load(f)
    return data

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
    entity2freq_test = dict()

    with open(gold_file, 'r') as f:
        data = json.load(f)

    for inst in data:
        for e in inst["entities"]:
            entity_str = ' '.join(e["tokens"])
            if entity_str in entity2freq_train.keys():
                entity2freq_test[entity_str] = entity2freq_train[entity_str]
            else:
                entity2freq_test[entity_str] = 0

    return entity2freq_test

def case_study(base_pred, ours_pred):
    assert len(base_pred) == len(ours_pred)
    # case_num = 400
    # base_pred, ours_pred = base_pred[:case_num], ours_pred[:case_num]

    def _is_predict_right(gold_data, pred_data):
        flag = True
        for span in gold_data["spans"]:
            if span not in pred_data["spans"]:
                flag = False
        return flag

    case_id = 0

    n_wrong_type_equal_highest_node = 0
    n_base_pred = 0
    n_our_pred = 0

    for i in range(len(ours_pred)):

        is_base_pred_right = _is_predict_right(
            gold_data=base_pred[i]["gold_data"],
            pred_data=base_pred[i]["pred_data"]
        )

        is_ours_pred_right = _is_predict_right(
            gold_data=ours_pred[i]["gold_data"],
            pred_data=ours_pred[i]["pred_data"]
        )

        if (not is_base_pred_right) and is_ours_pred_right \
                and len(ours_pred[i]["gold_data"]["spans"]) > 0:
        # if (not is_ours_pred_right) and is_base_pred_right \
        #     and (len(ours_pred[i]["gold_data"]["spans"]) > 0):
            bpes = base_pred[i]["bpe_strings"]

            def _f(spans):
                post_spans = []
                for j in range(len(spans)):
                    s, e = spans[j][0], spans[j][1]
                    post_spans.append(spans[j] + [' '.join(bpes[s:e+1])])
                return post_spans

            sent_bpe = " ".join(base_pred[i]["bpe_strings"][1:-1]) #.replace(" ##", "")
            sent_word = sent_bpe.replace(" ##", "")
            gold_span_phrases = ours_pred[i]["gold_data"]["phrases"]
            gold_spans = _f(spans=ours_pred[i]["gold_data"]["spans"])
            base_spans = _f(spans=base_pred[i]["pred_data"]["spans"])
            ours_spans = _f(spans=ours_pred[i]["pred_data"]["spans"])

            node_types_freqs = []
            highest_freq_types = []
            node_types = ours_pred[i]["gold_data"]["node_entity_types"]
            for j in range(len(node_types)):
                total_freq = 0
                node_types_freq = dict()
                for k in range(len(node_types[j])):
                    node_type = node_types[j][k]
                    if node_type not in node_types_freq.keys():
                        node_types_freq[node_type] = 1
                    else:
                        node_types_freq[node_type] += 1
                    total_freq += 1
                node_types_freq["all"] = total_freq
                node_types_freqs.append(node_types_freq)

                highest_freq_type = None
                highest_freq = 0
                for k, v in node_types_freq.items():
                    if k == "all":
                        continue
                    if v > highest_freq:
                        highest_freq = v
                    highest_freq_type = k
                highest_freq_types.append(highest_freq_type)

            n_base_pred += len(base_pred[i]["gold_data"]["spans"])
            n_our_pred += len(ours_pred[i]["gold_data"]["spans"])
            # b = ours_pred[i]["pred_data"]["spans"]
            for j in range(len(highest_freq_types)):
                # a = ours_pred[i]["pred_data"]["spans"][j]
                if highest_freq_types[j] == ours_pred[i]["gold_data"]["spans"][j][2]:
                    n_wrong_type_equal_highest_node += 1

            res_str = f"case id: {case_id} \n" + \
                f"sentence (bpes): {sent_bpe} \n" + \
                f"sentence (words): {sent_word} \n" + \
                f"gold span phrases: {gold_span_phrases} \n" + \
                f"gold spans: {gold_spans} \n" + \
                f"base spans: {base_spans} \n" + \
                f"ours spans: {ours_spans} \n" + \
                f"node freq: {node_types_freqs} \n" + \
                f"highest freq types: {highest_freq_types} \n"

            print(res_str)
            case_id += 1
    wrong_by_node_ratio = round(100 * n_wrong_type_equal_highest_node / n_our_pred, 2)
    print(
        f"n_base_pred: {n_base_pred} " + \
        f"n_our_pred: {n_our_pred} " + \
        f"n_wrong_type_equal_highest_node: {n_wrong_type_equal_highest_node} " + \
        f"wrong_by_node_ratio: {wrong_by_node_ratio} \n"
    )


if __name__ == "__main__":
    # todo: why there is a empty span prediction?
    data_dir = f"/NAS2020/Workspaces/NLPGroup/juncw/" \
               f"database/NER/Flat_NER/SpanNER_en"

    # GENIA
    data_name = "genia" # ace04, ace05, genia
    data_label = "valid" # valid, test

    base = "span_bert"
    base_time = "10_17 2021-04-14 17:43:05"
    base_upd = "14_103852"

    ours = "span_bert_gcn"
    ours_time = "33_1025 2021-04-13 11:49:03"
    ours_upd = "10_74180"

    # ACE2005
    # data_name = "ace05" # ace04, ace05, genia
    # data_label = "valid" # valid, test
    #
    # base = "span_bert"
    # base_time = "20_13 2021-04-14 17:43:15"
    # base_upd = "27_98523"
    #
    # ours = "span_bert_gcn"
    # ours_time = "33_114 2021-04-14 01:07:17"
    # ours_upd = "25_91225"

    train_file = f"{data_dir}/{data_name}/train.json"
    gold_file = f"{data_dir}/{data_name}/{data_label}.json"

    base_pred_file = f"{data_dir}/__results/{data_name}/{base}/" \
                         f"{base_time}/{data_label}_{base_upd}.json"
    ours_pred_file = f"{data_dir}/__results/{data_name}/{ours}/" \
                     f"{ours_time}/{data_label}_{ours_upd}.json"

    entity2freq_train = read_json_for_entity2freq(
        json_data_file=train_file
    ) # entity frequency in train data

    entity2freq_eval = get_entity2freq_test(
        entity2freq_train=entity2freq_train,
        gold_file=gold_file
    ) # the frequency of eval entity in train data

    base_pred = load_json(base_pred_file)
    ours_pred = load_json(ours_pred_file)
    case_study(
        base_pred=base_pred,
        ours_pred=ours_pred,
    )
