import os
from tqdm import tqdm

def bioes2bmoes(from_file, to_file):
    from_f = open(from_file, "r")
    to_f = open(to_file, "w")

    for line in tqdm(from_f.readlines()):
        line = line.strip()
        if line == "":
            to_f.write(f"\n")
            continue
        line_split = line.split(" ")
        tok, label = line_split[0], line_split[1]
        if label[0] == "I":
            label = "M" + label[1:]
        to_f.write(f"{tok} {label}\n")

    from_f.close()
    to_f.close()

# for data_name in ["weibo", "resume", "ontonote4", "msra", "people_daily"]:
#     print(data_name)
#
#     from_data_dir = f"/newNAS/Workspaces/NLPGroup/juncw/summer_dataset/reduce_emb_dataset/{data_name}"
#     to_data_dir = f"/newNAS/Workspaces/NLPGroup/juncw/summer_dataset/reduce_emb_dataset/{data_name}/bmoes"
#
#     os.makedirs(to_data_dir, exist_ok=True)
#
#     bioes2bmoes(f"{from_data_dir}/train.txt.clip", f"{to_data_dir}/train.txt")
#     bioes2bmoes(f"{from_data_dir}/dev.txt.clip", f"{to_data_dir}/dev.txt")
#     bioes2bmoes(f"{from_data_dir}/test.txt.clip", f"{to_data_dir}/test.txt")


for data_name in ["conll2003"]:
    print(data_name)
    data_dir = "/NAS2020/Workspaces/NLPGroup/juncw/summer_dataset/reduce_emb_dataset/old_ner_2020_09_09/conll2003/"
    from_data_dir = f"{data_dir}/{data_name}"
    to_data_dir = f"/newNAS/Workspaces/NLPGroup/juncw/summer_dataset/reduce_emb_dataset/{data_name}/bmoes"

    os.makedirs(to_data_dir, exist_ok=True)

    bioes2bmoes(f"{from_data_dir}/train.txt.clip", f"{to_data_dir}/train.txt")
    bioes2bmoes(f"{from_data_dir}/dev.txt.clip", f"{to_data_dir}/dev.txt")
    bioes2bmoes(f"{from_data_dir}/test.txt.clip", f"{to_data_dir}/test.txt")