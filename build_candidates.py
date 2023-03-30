import os
import math
import json
import argparse
from tqdm import tqdm
from multiprocessing import Pool
from transformers import BertTokenizer

def read_nodes(nodes_json_file):
    with open(nodes_json_file, 'r') as f:
        nodes = json.load(f)
    return nodes

def read_datas(datas_json_file):
    with open(datas_json_file, 'r') as f:
        datas = json.load(f)
    return datas

def get_ngram_sets(tokens, ngram):
    """
    Args:
        tokens: Tuple[Str]. Note: 0 is padding.
        ngram: Int
    Return:

    """

    pad = tuple([" "] * (ngram - 1))
    padded_tokens = pad + tokens + pad
    ngram_sets = set(["".join(padded_tokens[i:i + ngram]) for i in range(len(padded_tokens) - (ngram - 1))])

    # ngram_sets = set(["".join(tokens[i:i + ngram]) for i in range(len(tokens) - (ngram - 1))])

    return ngram_sets

def ngram_similarity(data_ngrams, node_ngrams):
    """
        data_ngrams: Set(Tuple(Str)). Note: 0 is padding.
        node_ngrams: Set(Tuple(Str))
    """

    # setx = data_ngrams & node_ngrams
    # setx = set(filter(data_ngrams.__contains__, node_ngrams))

    setx = data_ngrams.intersection(node_ngrams) # it's said to be faster
    len1_len2 = len(node_ngrams) + len(data_ngrams) + 1e-8 # to avoid empty sets
    sim = 1 - (len1_len2 - 2 * len(setx)) / len1_len2

    return sim

def extract_spans(encodings, max_span_size):
    """
        encodings: List[Object]. Object can be Str or Int. List[List[Obj]]
        max_span: Int
        Extract all spans for a list of tokens.
    """

    def _f(_encodes):
        _d = []
        for e in _encodes:
            _d += e
        return tuple(_d)

    spans = []
    token_count = len(encodings)
    for size in range(1, max_span_size + 1):
        spans.extend(tuple(map(
            lambda i: _f(encodings[i:i + size]),
            list(range(0, (token_count - size) + 1))
        )))

    return tuple(spans)

def raw_extract_spans(encodings, max_span_size):
    """
        encodings: List[Object]. Object can be Str or Int. List[List[Obj]]
        max_span: Int
        Extract all spans for a list of tokens.
    """

    spans = []
    token_count = len(encodings)
    for size in range(1, max_span_size + 1):
        spans.extend(tuple(map(
            lambda i: encodings[i:i + size],
            list(range(0, (token_count - size) + 1))
        )))

    return tuple(spans)

def build_batch_candidates(_input):
    args, node_ids, node_types, jnodes, jnodes_bpe, \
    batch_jdatas, batch_jdatas_bpe, \
    batch_entities, batch_entities_bpe, \
    save_cand_file, start_idx, pid = _input

    nodes_1grams = tuple([get_ngram_sets(tokens=jnode, ngram=1) for jnode in jnodes_bpe])
    nodes_2grams = tuple([get_ngram_sets(tokens=jnode, ngram=2) for jnode in jnodes_bpe])
    nodes_3grams = tuple([get_ngram_sets(tokens=jnode, ngram=3) for jnode in jnodes_bpe])
    nodes_123grams = list(zip(nodes_1grams, nodes_2grams, nodes_3grams))

    pbar = tqdm(total=len(batch_jdatas), desc=f"PID {pid}", position=(pid + 1))
    assert len(batch_jdatas) == len(batch_entities)
    zip_data = zip(batch_jdatas, batch_jdatas_bpe, batch_entities, batch_entities_bpe)
    for cur_idx, (jdata, jdata_bpe, entities, entities_bpe) in enumerate(zip_data):
        cands = []
        raw_spans = raw_extract_spans(jdata, args.max_span_size)
        spans = extract_spans(jdata_bpe, args.max_span_size)

        raw_spans_entities = raw_spans + entities
        spans_entities = spans + entities_bpe

        for i in range(len(spans_entities)):
            span = spans_entities[i]

            d1g = get_ngram_sets(tokens=span, ngram=1)
            d2g = get_ngram_sets(tokens=span, ngram=2)
            d3g = get_ngram_sets(tokens=span, ngram=3)

            # cur_span_cands = []
            # for j, (n1g, n2g, n3g) in enumerate(nodes_123grams):
            #     w = 0
            #     w1 = ngram_similarity(d1g, n1g)
            #     if w1 > 0:
            #         w += w1
            #         w2 = ngram_similarity(d2g, n2g)
            #         if w2 > 0:
            #             w += w2
            #             w += ngram_similarity(d3g, n3g)
            #     if w > 0: # extract useful nodes for faster sorting
            #         cur_span_cands.append((node_ids[j], w))

            cur_span_cands = []
            for j, (n1g, n2g, n3g) in enumerate(nodes_123grams):
                w = 0
                w1 = ngram_similarity(d1g, n1g) / len(d1g)
                if w1 > 0:
                    w += 0.5 * w1 # weighted sum of 1/2/3-gram similarity
                    w2 = ngram_similarity(d2g, n2g) / len(d2g)
                    if w2 > 0:
                        w += 1.0 * w2
                        w += 1.5 * ngram_similarity(d3g, n3g) / len(d3g)
                if w > 0:  # extract useful nodes for faster sorting
                    cur_span_cands.append((node_ids[j], w))

            # cur_span_cands = list(zip(jnodes, node_ids, node_types, weights)) # too large to sort
            cur_span_cands.sort(key=lambda x: x[1], reverse=True)
            cands.append(
                {
                    "span": raw_spans_entities[i],
                    "candidates": [
                        {
                            "node": jnodes[nid],
                            "id": nid,
                            "type": node_types[nid],
                            "weight": w
                        } for nid, w in cur_span_cands[:args.cand_size]
                    ]
                }
            )

        with open(f"{save_cand_file}_{start_idx + cur_idx}.json", 'w') as f:
            json.dump(cands, f, ensure_ascii=False, indent=1)

        pbar.update(1)


def build_candidates(args, nodes, data_file, save_cand_file, tokenizer):

    data = read_datas(data_file)
    train_size = math.ceil(args.train_ratio * len(data))
    data = data[:train_size]
    # if "train" in data_file:
    #     train_size = math.ceil(args.train_ratio * len(data))
    #     data = data[:train_size]

    # Encode tokens to ids ... j- means encodings
    # jnodes = tuple([tuple(n["tokens"]) for n in nodes])
    # jdatas = tuple([tuple(d["tokens"]) for d in data])

    # entities = tuple([
    #     tuple([
    #         tuple(e["tokens"])
    #     for e in d["entities"]])
    # for d in data])

    jnodes = tuple([tuple(n["tokens"]) for n in nodes])

    jnodes_bpe = []
    for n in tqdm(nodes):
        bpes = []
        for w in n["tokens"]:
            # tmp = tokenizer.tokenize(w)
            bpes += tokenizer.tokenize(w)
        jnodes_bpe.append(tuple(bpes))
    jnodes_bpe = tuple(jnodes_bpe)

    jdatas = tuple([tuple(d["tokens"]) for d in data])

    jdatas_bpe = []
    for d in data:
        jdatas_bpe_i = []
        for t in d["tokens"]:
            bpes = tokenizer.tokenize(t)
            jdatas_bpe_i.append(tuple(bpes))
        jdatas_bpe.append(tuple(jdatas_bpe_i))
    jdatas_bpe = tuple(jdatas_bpe)

    entities = tuple([
        tuple([
            tuple(e["tokens"])
        for e in d["entities"]])
    for d in data])

    entities_bpe = []
    for d in data:
        entities_i = []
        for e in d["entities"]:
            bpes = []
            for t in e["tokens"]:
                bpes += tokenizer.tokenize(t)
            entities_i.append(tuple(bpes))
        entities_bpe.append(tuple(entities_i))
    entities_bpe = tuple(entities_bpe)

    node_ids = tuple([n["id"] for n in nodes])
    node_types = tuple([n["type"] for n in nodes])

    pool = Pool(args.num_process)
    data_len = len(jdatas)
    step = int(data_len / args.num_process) + 1 # Note: data_len should >= args.num_process
    for pid, start_idx in enumerate(range(0, data_len, step)):
        batch_jdatas = jdatas[start_idx: start_idx + step]
        batch_jdatas_bpe = jdatas_bpe[start_idx: start_idx + step]
        batch_entities = entities[start_idx: start_idx + step]
        batch_entities_bpe = entities_bpe[start_idx: start_idx + step]
        _input = args, node_ids, node_types, jnodes, jnodes_bpe, \
                 batch_jdatas, batch_jdatas_bpe, \
                 batch_entities, batch_entities_bpe, \
                 save_cand_file, start_idx, pid
        pool.apply_async(build_batch_candidates, [_input])
    pool.close()
    pool.join()

def build_candidates_func(
    args, nodes_file, tokenizer, train_file, valid_file, test_file,
    save_cand_train_file, save_cand_valid_file, save_cand_test_file
):
    nodes = read_nodes(nodes_file)  # List[Dict]

    build_candidates(args, nodes, train_file, save_cand_train_file, tokenizer)
    build_candidates(args, nodes, valid_file, save_cand_valid_file, tokenizer)
    build_candidates(args, nodes, test_file, save_cand_test_file, tokenizer)


if __name__ == "__main__":
    data_dir = f"/NAS2020/Workspaces/NLPGroup/juncw/database/NER/nested_ner"
    data_names = ["ace04", "ace05", "genia"]

    debug = False

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--max_span_size', default=10, type=int)
    parser.add_argument('--cand_size', default=10, type=int)
    parser.add_argument('--num_process', default=48, type=int)
    parser.add_argument('--train_ratio', default=0.01, type=float)
    args, _ = parser.parse_known_args()

    if debug:
        args.train_ratio = 0.01
    else:
        args.train_ratio = 1.0
    print(args)

    pretrained_model = {
        "ace04": "bert-base-cased",
        "ace05": "bert-base-cased",
        "genia": "dmis-lab/biobert-base-cased-v1.1",
    }

    do_lower_case = {
        "ace04": False,
        "ace05": False,
        "genia": False,
    }

    for data_name in data_names:
        print(f"{data_name}")

        train_file = f"{data_dir}/{data_name}/train.json"
        valid_file = f"{data_dir}/{data_name}/valid.json"
        test_file = f"{data_dir}/{data_name}/test.json"

        if debug:
            nodes_file = f"{data_dir}/{data_name}/Graph-debug/nodes.json"
            # edges_file = f"{data_dir}/{data_name}/Graph-debug/edges.json"

            os.makedirs(f"{data_dir}/{data_name}/Graph-debug/candidates", exist_ok=True)
            save_cand_train_file = f"{data_dir}/{data_name}/Graph-debug/candidates/train"
            save_cand_valid_file = f"{data_dir}/{data_name}/Graph-debug/candidates/valid"
            save_cand_test_file = f"{data_dir}/{data_name}/Graph-debug/candidates/test"
        else:
            nodes_file = f"{data_dir}/{data_name}/Graph/nodes.json"
            # edges_file = f"{data_dir}/{data_name}/Graph/edges.json"

            os.makedirs(f"{data_dir}/{data_name}/Graph/candidates", exist_ok=True)
            save_cand_train_file = f"{data_dir}/{data_name}/Graph/candidates/train"
            save_cand_valid_file = f"{data_dir}/{data_name}/Graph/candidates/valid"
            save_cand_test_file = f"{data_dir}/{data_name}/Graph/candidates/test"

        tokenizer = BertTokenizer.from_pretrained(
            pretrained_model[data_name], do_lower_case=do_lower_case[data_name]
        )

        build_candidates_func(
            args=args, nodes_file=nodes_file, tokenizer=tokenizer,
            train_file=train_file, valid_file=valid_file, test_file=test_file,
            save_cand_train_file=save_cand_train_file,
            save_cand_valid_file=save_cand_valid_file,
            save_cand_test_file=save_cand_test_file,
        )
        print("\n")

