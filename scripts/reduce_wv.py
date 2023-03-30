from tqdm import tqdm

import json
import argparse

def load_json(file):
    with open(file, 'r') as f:
        data = json.load(f)
    return data

parser = argparse.ArgumentParser(description='Arguments for training.')

#### Train
parser.add_argument('--read_wv_path', default='glove.6B.100d.txt', type=str)
parser.add_argument('--save_wv_path', default='ACE05.glove.6B.100d.txt', type=str)
parser.add_argument('--data_dir', default='ACE05', type=str)
parser.add_argument('--cased', default=0, type=int)

args = parser.parse_args()

# ACE04
# embed_dir = "/NAS2020/Workspaces/NLPGroup/juncw/database/NER/nested_ner/data"
# args.data_dir = "/NAS2020/Workspaces/NLPGroup/juncw/database/NER/nested_ner/ace04"
# args.read_wv_path = f"{embed_dir}/glove.6B.100d.txt"
# args.save_wv_path = f"{args.data_dir}/ACE04.glove.6B.100d.txt"
# args.cased = 0

# ACE05
# embed_dir = "/NAS2020/Workspaces/NLPGroup/juncw/database/NER/nested_ner/data"
# args.data_dir = "/NAS2020/Workspaces/NLPGroup/juncw/database/NER/nested_ner/ace05"
# args.read_wv_path = f"{embed_dir}/glove.6B.100d.txt"
# args.save_wv_path = f"{args.data_dir}/ACE05.glove.6B.100d.txt"
# args.cased = 0

# GENIA
embed_dir = "/NAS2020/Workspaces/NLPGroup/juncw/database/NER/nested_ner/data"
args.data_dir = "/NAS2020/Workspaces/NLPGroup/juncw/database/NER/nested_ner/genia"
args.read_wv_path = f"{embed_dir}/pubmed_shuffle_win_2_genia.txt"
args.save_wv_path = f"{args.data_dir}/GENIA.pubmed.200d.txt"
args.cased = 1

train = load_json(f"{args.data_dir}/train.json")
valid = load_json(f"{args.data_dir}/valid.json")
test = load_json(f"{args.data_dir}/test.json")
dataset = train + valid + test

vocab = set()
for item in dataset:
    tokens = item['tokens']
    if not args.cased:
        tokens = [t.lower() for t in tokens]
    vocab.update(tokens)

with open(args.read_wv_path) as fin, \
        open(args.save_wv_path, 'w') as fout:
    for line in tqdm(fin):
        w = line.split(' ')[0]
        if w in vocab:
            fout.write(line)

print("vocab size:", len(vocab))