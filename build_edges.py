import json
from tqdm import tqdm
from transformers import BertTokenizer

def read_nodes(node_json_file):
    with open(node_json_file, 'r') as f:
        nodes = json.load(f)
    return nodes

def get_ngram_sets(tokens, ngram):
    """
    Args:
        tokens: Tuple[Str]. Note: 0 is padding.
        ngram: Int
    Return:
        Set[Str]
    """

    pad = tuple([" "] * (ngram - 1))
    padded_tokens = pad + tokens + pad
    ngram_sets = set(["".join(padded_tokens[i:i + ngram]) for i in range(len(padded_tokens) - (ngram - 1))])
    return ngram_sets

def ngram_similarity(data_ngrams, node_ngrams):
    """
        data_ngrams: Set(Str). Note: 0 is padding.
        node_ngrams: Set(Str)
    """

    # setx = data_ngrams & node_ngrams
    # setx = set(filter(data_ngrams.__contains__, node_ngrams))

    setx = data_ngrams.intersection(node_ngrams) # it's said to be faster
    len1_len2 = len(node_ngrams) + len(data_ngrams) + 1e-8 # to avoid empty sets
    sim = 1 - (len1_len2 - 2 * len(setx)) / len1_len2
    return sim

def get_nid_2_ngrams(bpe_nodes, ngram):
    nid_2_ngrams = dict()
    for node in bpe_nodes:
        nid_2_ngrams[node["id"]] = get_ngram_sets(tuple(node["bpes"]), ngram=ngram)
    return nid_2_ngrams


def build_edges(node_file, save_edge_file, tokenizer):
    """
    Build edges at BPE level.
    """
    nodes = read_nodes(node_file)  # List[Dict]

    bpe_nodes = []
    for n in tqdm(nodes):
        bpes = []
        for w in n["tokens"]:
            bpes += tokenizer.tokenize(w)
        bpe_nodes.append({"id": n["id"], "bpes": bpes})

    node_ids = [n["id"] for n in bpe_nodes]  # List[Int]
    nid_2_1grams = get_nid_2_ngrams(bpe_nodes, ngram=1)
    nid_2_2grams = get_nid_2_ngrams(bpe_nodes, ngram=2)
    nid_2_3grams = get_nid_2_ngrams(bpe_nodes, ngram=3)

    edges = []  # (id1, id2, weight)
    for i, id1 in enumerate(tqdm(node_ids)):
        for id2 in node_ids[i + 1:]:
            # average n-gram similarity
            # w = 0
            # w1 = ngram_similarity(nid_2_1grams[id1], nid_2_1grams[id2])
            # if w1 > 0:
            #     w += w1
            #     w2 = ngram_similarity(nid_2_2grams[id1], nid_2_2grams[id2])
            #     if w2 > 0:
            #         w += w2
            #         w += ngram_similarity(nid_2_3grams[id1], nid_2_3grams[id2])

            # weighted n-gram similarity
            w = 0
            w1 = ngram_similarity(nid_2_1grams[id1], nid_2_1grams[id2])
            if w1 > 0:
                w += 0.5 * w1 # weighted sum of 1/2/3-gram similarity
                w2 = ngram_similarity(nid_2_2grams[id1], nid_2_2grams[id2])
                if w2 > 0:
                    w += 1.0 * w2
                    w += 1.5 * ngram_similarity(nid_2_3grams[id1], nid_2_3grams[id2])

            if w > 0:
                edges.append(
                    {
                        "node1":
                            {
                                "id": id1,
                                "type": nodes[id1]["type"],
                                "tokens": nodes[id1]["tokens"],
                            },
                        "node2":
                            {
                                "id": id2,
                                "type": nodes[id2]["type"],
                                "tokens": nodes[id2]["tokens"],
                            },
                        "weight": w
                    }
                )

    with open(save_edge_file, 'w') as f:
        json.dump(edges, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    data_dir = f"//NAS2020/Workspaces/NLPGroup/juncw/database/NER/nested_ner"
    data_names = ["ace04", "ace05", "genia"]
    debug = True

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

        if debug:
            node_file = f"{data_dir}/{data_name}/Graph-debug/nodes.json"
            save_edge_file = f"{data_dir}/{data_name}/Graph-debug/edges.json"
        else:
            node_file = f"{data_dir}/{data_name}/Graph/nodes.json"
            save_edge_file = f"{data_dir}/{data_name}/Graph/edges.json"

        tokenizer = BertTokenizer.from_pretrained(
            pretrained_model[data_name], do_lower_case=do_lower_case[data_name]
        )

        build_edges(
            node_file=node_file, save_edge_file=save_edge_file, tokenizer=tokenizer
        )
        print("\n")
