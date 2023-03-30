import json

def count_sent_num(datas_json_file):
    with open(datas_json_file, 'r') as f:
        datas = json.load(f)
    print(f"{datas_json_file}: {len(datas)}")
    return datas

if __name__ == "__main__":
    data_dir = f"/NAS2020/Workspaces/NLPGroup/juncw/database/NER/Flat_NER/SpanNER_en"
    data_names = [
        # "ace04",
        "msra", "people_daily", "ecommerce",
        "weibo", "resume", "ontonote4",  # Chinese flat NER dataset
        "scierc", "conll04",  # entity-relation dataset
        "ace04", "ace05", "genia",  # nested NER dataset
        "conll03",  # English flat NER dataset
    ]

    for data_name in data_names:
        print(f"\n{data_name}")

        count_sent_num(f"{data_dir}/{data_name}/train.json")
        count_sent_num(f"{data_dir}/{data_name}/valid.json")
        count_sent_num(f"{data_dir}/{data_name}/test.json")



