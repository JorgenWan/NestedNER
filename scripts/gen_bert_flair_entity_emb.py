import numpy as np
import argparse
import json
import pickle
from tqdm import tqdm

from flair.embeddings import BertEmbeddings, FlairEmbeddings, StackedEmbeddings
from flair.data import Token, Sentence


def form_sentence(tokens):
    s = Sentence()
    for w in tokens:
        s.add_token(Token(w))
    return s


def get_embs(s):
    ret = []
    for t in s:
        ret.append(t.get_embedding().cpu().numpy())
    return np.stack(ret, axis=0)


def load_json(file):
    with open(file, 'r') as f:
        data = json.load(f)
    return data


parser = argparse.ArgumentParser(description='Arguments for training.')
parser.add_argument('--data_dir', default='ACE05', type=str)
parser.add_argument('--model_name', default='bert-large-cased', type=str)
parser.add_argument('--ent_lm_emb_save_path', default='ent.lm.emb.pkl', type=str)
parser.add_argument('--cased', default=0, type=int)
parser.add_argument('--add_flair', default=0, type=int)
parser.add_argument('--flair_name', default='news', type=str)

args = parser.parse_args()

# export CUDA_VISIBLE_DEVICES=3

# ACE04
# args.data_dir = "/NAS2020/Workspaces/NLPGroup/juncw/database/NER/nested_ner/ace04"
# args.model_name = "bert-large-uncased"
# args.flair_name = "news"
# args.ent_lm_emb_save_path = f"{args.data_dir}/ACE04.ent_bert_large_uncased_flair.emb.pkl"
# args.cased = 0
# args.add_flair = 1

# args.data_dir = "/NAS2020/Workspaces/NLPGroup/juncw/database/NER/nested_ner/ace04"
# args.model_name = "bert-base-cased"
# args.flair_name = "news"
# args.ent_lm_emb_save_path = f"{args.data_dir}/ACE04.ent_bert_base_cased_flair.emb.pkl"
# args.cased = 1
# args.add_flair = 1

# ACE05
# args.data_dir = "/NAS2020/Workspaces/NLPGroup/juncw/database/NER/nested_ner/ace05"
# args.model_name = "bert-large-uncased"
# args.flair_name = "news"
# args.ent_lm_emb_save_path = f"{args.data_dir}/ACE05.ent_bert_large_uncased_flair.emb.pkl"
# args.cased = 0
# args.add_flair = 1

# args.data_dir = "/NAS2020/Workspaces/NLPGroup/juncw/database/NER/nested_ner/ace05"
# args.model_name = "bert-base-cased"
# args.flair_name = "news"
# args.ent_lm_emb_save_path = f"{args.data_dir}/ACE05.ent_bert_base_cased_flair.emb.pkl"
# args.cased = 1
# args.add_flair = 1

# GENIA
# args.data_dir = "/NAS2020/Workspaces/NLPGroup/juncw/database/NER/nested_ner/genia"
# args.model_name = "dmis-lab/biobert-large-cased-v1.1"
# args.flair_name = "pubmed"
# args.ent_lm_emb_save_path = f"{args.data_dir}/GENIA.ent_biobert_large_cased_flair.emb.pkl"
# args.cased = 1
# args.add_flair = 1

args.data_dir = "/NAS2020/Workspaces/NLPGroup/juncw/database/NER/nested_ner/genia"
args.model_name = "dmis-lab/biobert-base-cased-v1.1"
args.flair_name = "pubmed"
args.ent_lm_emb_save_path = f"{args.data_dir}/GENIA.ent_biobert_base_cased_flair.emb.pkl"
args.cased = 1
args.add_flair = 1

# NNE
# args.data_dir = "/NAS2020/Workspaces/NLPGroup/juncw/database/NER/nested_ner/nne"
# args.model_name = "bert-large-uncased"
# args.flair_name = "news"
# args.ent_lm_emb_save_path = f"{args.data_dir}/NNE.ent_bert_large_uncased_flair.emb.pkl"
# args.cased = 0
# args.add_flair = 1

# args.data_dir = "/NAS2020/Workspaces/NLPGroup/juncw/database/NER/nested_ner/nne"
# args.model_name = "bert-base-cased"
# args.flair_name = "news"
# args.ent_lm_emb_save_path = f"{args.data_dir}/NNE.ent_bert_base_cased_flair.emb.pkl"
# args.cased = 1
# args.add_flair = 1

train = load_json(f"{args.data_dir}/train.json")
valid = load_json(f"{args.data_dir}/valid.json")
test = load_json(f"{args.data_dir}/test.json")
dataset = train + valid + test

ent_set = set()
for item in tqdm(dataset):
    for e in item["entities"]:
        ent_set.add(tuple(e["tokens"]))
print("number of entities:", len(ent_set))


bert_embedding = BertEmbeddings(args.model_name, layers='-1,-2,-3,-4', use_scalar_mix=True, pooling_operation="mean")
if args.add_flair:
    flair_embedding = StackedEmbeddings([
        FlairEmbeddings(f'{args.flair_name}-forward'),
        FlairEmbeddings(f'{args.flair_name}-backward'),
    ])

if args.cased:
    bert_embedding.tokenizer.basic_tokenizer.do_lower_case = False


ent_emb_dict = {}
for item in tqdm(ent_set):
    assert type(item) is tuple
    tokens = item
    s = form_sentence(tokens)

    s.clear_embeddings()
    bert_embedding.embed(s)
    emb = get_embs(s)  # (T, 4*H)

    if args.add_flair:
        s.clear_embeddings()
        flair_embedding.embed(s)
        emb = np.concatenate([emb, get_embs(s)], axis=-1)

    ent_emb_dict[tokens] = {
        "word_emb": emb.astype('float16'),
    }

with open(args.ent_lm_emb_save_path, 'wb') as f:
    pickle.dump(ent_emb_dict, f)