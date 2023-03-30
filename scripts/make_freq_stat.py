import json

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

def get_entity2freq_test(entity2freq_train, test_file):
    entity2freq_test = dict()
    n_ent = 0

    with open(test_file, 'r') as f:
        data = json.load(f)

    idx = 0
    for d in data:
        for e in d["entities"]:
            entity_str = ' '.join(e["tokens"])
            entity_str_idx = ' '.join(e["tokens"] + [str(idx)]) # distinguish same entities in different places
            if entity_str in entity2freq_train.keys():
                entity2freq_test[entity_str_idx] = entity2freq_train[entity_str]
            else:
                entity2freq_test[entity_str_idx] = 0
            idx += 1
        n_ent += len(d["entities"])

    return entity2freq_test, n_ent

def get_freq2num(entity2freq):
    freq2num = dict()
    for e, f in entity2freq.items():
        if f in freq2num.keys():
            freq2num[f] += 1
        else:
            freq2num[f] = 1
    return freq2num


if __name__ == "__main__":

    data_dir = f"/NAS2020/Workspaces/NLPGroup/juncw/database/NER/nested_ner"
    data_names = [
        # "scierc", "conll04",  # entity-relation dataset
        "ace04", "ace05", "genia", "nne", # nested NER dataset
        # "conll03",  # English flat NER dataset
        # "weibo", "resume", "ontonote4", "msra",
        # "people_daily", "ecommerce"  # Chinese flat NER dataset
    ]

    save_dir = "/NAS2020/Workspaces/NLPGroup/juncw/database/NER/nested_ner/__results/"
    save_file = f"{save_dir}/freq_stat.txt"

    f = open(save_file, "w")

    for data_name in data_names:
        entity2freq_train = read_json_for_entity2freq(f"{data_dir}/{data_name}/train.json")
        print("\n")
        for data_label in ["valid", "test"]:

            entity2freq_test, n_ent = get_entity2freq_test(
                entity2freq_train=entity2freq_train,
                test_file=f"{data_dir}/{data_name}/{data_label}.json"
            )

            freq2num = get_freq2num(entity2freq_test)

            sorted_freq = sorted(freq2num.keys())

            low_freq_ent_ratio = round(100 * (freq2num[0] + freq2num[1] + freq2num[2])/ n_ent, 2)
            # low_freq_ent_ratio = round(100 * (freq2num[0])/ n_ent, 2)

            total_freq = sum([freq2num[f]*f for f in freq2num.keys()])
            avg_ent_freq = round(total_freq/n_ent, 0)

            print(f"{data_name} {data_label} "
                  f"n_ent={n_ent} "
                  f"low_freq_ent_ratio={low_freq_ent_ratio} "
                  f"avg_ent_freq={avg_ent_freq} "
              )

            f.write("\n")
            f.write(f"{data_name} {data_label}\n")
            for k in sorted_freq:
                f.write(f"freq = {k}, n_gold = {freq2num[k]}\n")

    f.close()