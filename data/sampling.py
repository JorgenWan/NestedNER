import dgl
import torch
import random
import copy

from data import utils

def beam_search(nodes, cands, graph_neighbors):
    """
        A BFS-style beam search.
    """
    n_hop = len(graph_neighbors)
    cur_i = 0
    for depth in range(n_hop):
        tmp_i = len(nodes)
        for n in nodes[cur_i:]:
            nodes.extend(copy.deepcopy(cands[n][:graph_neighbors[depth]]))
        cur_i = tmp_i

def chunk_list_by_max_sum(l, max_sum):
    """
    Example:
        Input:
            l = [8, 3,2,1,5,3,6,3,1,7, 7,1]
            max_size = 8
        Output:
            [(0, 0), (1, 3), (4, 5), (6, 6), (7, 8), (9, 9), (10, 11)]
    Args:
        l: List
        max_sum: Int
    Return:
        indices: List[Tuple(Int, Int)], start and end are inclusive
    """

    indices = []
    start, end = 0, 0
    cur_sum = 0
    for i in range(len(l)):
        assert l[i] <= max_sum
        cur_sum += l[i]
        if cur_sum <= max_sum:
            end = i
        else:
            indices.append((start, end))
            start = end = i
            cur_sum = l[i]

    if end < len(l):
        indices.append((start, len(l) - 1))

    return indices

def batch_indices_by_graph_nodes(nodes_sizes, max_batch_nodes):
    """
    Function:
        Batch graphs for avoiding OOM efficient GPU usage
    Args:
        num_nodes: List[Int]
        max_batch_nodes: Int
    """

    batch_graph_indices = chunk_list_by_max_sum(
        l=nodes_sizes, max_sum=max_batch_nodes
    )
    return batch_graph_indices

def batch_indices_by_graph_edges(sgs_nodes, graph, max_batch_edges):
    """
    Function:
        Batch graphs for avoiding OOM efficient GPU usage
    Args:
        sgs_edges: List[Int]
        max_batch_edges: Int
    """

    def _cal_edges(sg_nodes):
        return graph.subgraph(sg_nodes).number_of_edges()

    edges_sizes = [_cal_edges(sg_nodes) for sg_nodes in sgs_nodes]
    batch_graph_indices = chunk_list_by_max_sum(
        l=edges_sizes, max_sum=max_batch_edges
    )
    return batch_graph_indices

def build_subgraph_for_spans_with_only_entity(
    graph, sgs_nodes, spans_weights, use_graph_weight
):
    """
        sgs_nodes: List[List[Int]]
        spans_weights: List[List[Float]]
    """

    sgs = []
    sgs_nodes_weights = list(zip(sgs_nodes, spans_weights))
    for sg_nodes, span_weights in sgs_nodes_weights:
        tmp_sg = graph.subgraph(sg_nodes)
        node_edges_src, node_edges_tgt = tmp_sg.edges()

        edges_src = node_edges_src.tolist()
        edges_tgt = node_edges_tgt.tolist()

        sg = dgl.DGLGraph()
        sg.add_nodes(len(sg_nodes))
        sg.add_edges(edges_src, edges_tgt)

        if use_graph_weight: # todo
            weights = torch.FloatTensor([w for _, w in span_weights] * 2)
            sg.add_edges(edges_src + edges_tgt, edges_tgt + edges_src, data={"w": weights})

        # sg.add_edges(span_node, span_node, data={"w": torch.FloatTensor([1])}) # todo: self-loop?
        sg = dgl.add_self_loop(sg)  # added

        sgs.append(sg)

    return sgs


def build_subgraph_for_spans(graph, sgs_nodes, spans_weights, use_graph_weight):
    """
        sgs_nodes: List[List[Int]]
        spans_weights: List[List[Float]]
    """

    sgs = []
    sgs_nodes_weights = list(zip(sgs_nodes, spans_weights))
    for sg_nodes, span_weights in sgs_nodes_weights:
        tmp_sg = graph.subgraph(sg_nodes)
        node_edges_src, node_edges_tgt = tmp_sg.edges()

        span_node = tmp_sg.number_of_nodes()
        span_edges_src = [span_node for _ in range(len(span_weights))]
        span_edges_tgt = [node for node, _ in span_weights]

        edges_src = node_edges_src.tolist() + span_edges_src + span_edges_tgt
        edges_tgt = node_edges_tgt.tolist() + span_edges_tgt + span_edges_src
        # sg = dgl.graph((edges_src, edges_tgt), num_nodes=len(sg_nodes) + 1)
        sg = dgl.DGLGraph()
        sg.add_nodes(len(sg_nodes) + 1)
        sg.add_edges(edges_src, edges_tgt)

        if use_graph_weight: # todo
            weights = torch.FloatTensor([w for _, w in span_weights] * 2)
            sg.add_edges(edges_src + edges_tgt, edges_tgt + edges_src, data={"w": weights})

        # sg.add_edges(span_node, span_node, data={"w": torch.FloatTensor([1])}) # todo: self-loop?
        sg = dgl.add_self_loop(sg)  # added
        sgs.append(sg)

    return sgs

def pad_char_ids(raw_char_ids, pad_id=0, max_n_char=128, max_n_word=512):
    char_ids = []
    for char_ids_i in raw_char_ids:
        n_pad = max_n_char - len(char_ids_i)
        if n_pad > 0:
            char_ids.append(char_ids_i + ([pad_id] * n_pad))
        elif n_pad < 0:
            raise Exception(
                '(Chars-1) Text too long (%d / %d):' % (len(char_ids_i), n_pad)
            )
    n_pad = max_n_word - len(char_ids)
    if n_pad > 0:
        char_ids += [[pad_id] * max_n_char] * n_pad
    elif n_pad < 0:
        raise Exception(
            '(Chars-2) Text too long (%d / %d):' % (len(char_ids), n_pad)
        )
    char_ids = torch.tensor(char_ids, dtype=torch.long)
    return char_ids

def pad_word_ids(raw_word_ids, pad_id=0, max_n_word=512):
    word_ids = raw_word_ids
    n_pad = max_n_word - len(raw_word_ids)
    if n_pad > 0:
        word_ids = word_ids + ([pad_id] * n_pad)
    elif n_pad < 0:
        raise Exception(
            '(Words) Text too long (%d / %d):' % (len(word_ids), n_pad)
        )
    word_ids = torch.tensor(word_ids, dtype=torch.long)
    return word_ids

def create_sample(
    cfg, doc, span_cands, entity_cands, graph,
    node2type_idx, node_idx2bpe_idx, sample_type
):
    """
    Args:
        doc: document
        max_span_size: Int
        cands: something like
                {
                    span: [
                        {
                            "node": jnodes[nid],
                            "id": nid,
                            "type": node_types[nid],
                            "weight": w
                        }
                        for nid, w in cur_span_cands[:args.cand_size]
                    ]
                }
        graph: dgl.Graph
        neg_entity_count: Int
    Return:
        1. something like Span-based NER method
        2. something about the span-level graph
    """

    max_n_word = cfg.max_n_word
    max_n_char = cfg.max_word_len

    # char_ids
    char_ids = None
    if cfg.use_char_encoder:
        char_ids = pad_char_ids(
            doc.char_ids, pad_id=cfg.char2idx['[PAD]'],
            max_n_char=max_n_char, max_n_word=max_n_word
        )

    # word_ids
    word_ids = None
    if cfg.use_word_encoder or cfg.use_char_encoder:
        word_ids = pad_word_ids(
            doc.word_ids, pad_id=cfg.word2idx['[PAD]'], max_n_word=max_n_word
        )
    # if cfg.use_word_encoder:
    #     word_ids = pad_word_ids(
    #         doc.word_ids, pad_id=cfg.word2idx['[PAD]'], max_n_word=max_n_word
    #     )

    # lm_spans
    lm_spans = [token.num_bpe for token in doc.tokens] # +1 for [CLS] and [SEP]
    # lm_spans = None # not for [CLS] and [SEP]

    # word_strs for lm_embed
    word_strs = tuple([token.phrase for token in doc.tokens])

    # positive entities
    pos_entity_spans, pos_entity_types, pos_entity_masks, \
    pos_entity_sizes, pos_entity_phrases = [], [], [], [], []
    for e in doc.entities:
        span = (e.start, e.end)
        # print("kkk", span)
        # size2 = e.end - e.start + 1
        # span2 = doc.tokens[e.start:e.start + size2].span
        # print("lll", span2)
        # assert span2 == span, print(span, span2, size2)
        pos_entity_spans.append(span)
        pos_entity_types.append(e.entity_type_id)
        pos_entity_masks.append(utils.create_entity_mask(*span, max_n_word))
        pos_entity_sizes.append(len(e.tokens))
        pos_entity_phrases.append(e.phrase)

    if cfg.data_name != "nne":
        # negative entities
        neg_entity_spans, neg_entity_sizes, neg_entity_phrases = [], [], []
        for size in range(1, cfg.max_span_size + 1):
            for i in range(0, (len(doc.tokens) - size) + 1):
                # span = doc.tokens[i:i + size].span
                span = (i, i + size - 1)
                if span not in pos_entity_spans:
                    neg_entity_spans.append(span)
                    neg_entity_sizes.append(size)
                    neg_entity_phrases.append(" ".join([t.phrase for t in doc.tokens[i:i + size]]))

        # sample negative entities
        if sample_type == "train":
            neg_entity_samples = random.sample(
                list(zip(neg_entity_spans, neg_entity_sizes, neg_entity_phrases)),
                min(len(neg_entity_spans), cfg.neg_entity_count)
            )
            neg_entity_spans, neg_entity_sizes, neg_entity_phrases = zip(*neg_entity_samples) \
                if neg_entity_samples else ([], [], [])
    elif cfg.data_name == "nne":
        short_len = 3

        # negative entities
        neg_entity_spans_short, neg_entity_sizes_short, neg_entity_phrases_short = [], [], []
        for size in range(1, short_len + 1):
            for i in range(0, (len(doc.tokens) - size) + 1):
                # span = doc.tokens[i:i + size].span
                span = (i, i + size - 1)
                if span not in pos_entity_spans:
                    neg_entity_spans_short.append(span)
                    neg_entity_sizes_short.append(size)
                    neg_entity_phrases_short.append(" ".join([t.phrase for t in doc.tokens[i:i + size]]))

        neg_entity_spans, neg_entity_sizes, neg_entity_phrases = [], [], []
        for size in range(short_len + 1, cfg.max_span_size + 1):
            for i in range(0, (len(doc.tokens) - size) + 1):
                # span = doc.tokens[i:i + size].span
                span = (i, i + size - 1)
                if span not in pos_entity_spans:
                    neg_entity_spans.append(span)
                    neg_entity_sizes.append(size)
                    neg_entity_phrases.append(" ".join([t.phrase for t in doc.tokens[i:i + size]]))

        # sample negative entities
        if sample_type == "train":
            neg_entity_samples = random.sample(
                list(zip(neg_entity_spans, neg_entity_sizes, neg_entity_phrases)),
                min(len(neg_entity_spans), cfg.neg_entity_count)
            )
            neg_entity_spans, neg_entity_sizes, neg_entity_phrases = zip(*neg_entity_samples) \
                if neg_entity_samples else ([], [], [])

        neg_entity_spans = neg_entity_spans + neg_entity_spans_short
        neg_entity_sizes = neg_entity_sizes + neg_entity_sizes_short
        neg_entity_phrases = neg_entity_phrases + neg_entity_phrases_short

    neg_entity_masks = [utils.create_entity_mask(*span, max_n_word) for span in neg_entity_spans]
    neg_entity_types = [cfg.entity_type2idx['None']] * len(neg_entity_spans)

    # merge
    entity_types = pos_entity_types + neg_entity_types
    entity_masks = pos_entity_masks + neg_entity_masks
    entity_sizes = pos_entity_sizes + list(neg_entity_sizes)
    entity_spans = pos_entity_spans + list(neg_entity_spans)

    assert len(entity_masks) == len(entity_sizes) == len(entity_types)

    # create tensors
    # token indices
    bpe_ids = torch.zeros(cfg.max_n_bpe, dtype=torch.long)
    bpe_ids[:len(doc.bpe_ids)] = torch.tensor(doc.bpe_ids, dtype=torch.long)
    # bpe_ids = None # todo: no LM fine-tunning now

    # masking of tokens
    lm_attn_masks = torch.zeros(cfg.max_n_bpe, dtype=torch.bool)
    lm_attn_masks[:len(doc.bpe_ids)] = torch.ones(len(doc.bpe_ids), dtype=torch.bool)
    # lm_attn_masks = None # todo: no LM fine-tunning now

    # also create samples_masks:
    # tensors to mask entity/relation samples of batch
    # since samples are stacked into batches, "padding" entities/relations possibly must be created
    # these are later masked during loss computation
    if entity_masks:
        entity_types = torch.tensor(entity_types, dtype=torch.long)
        entity_masks = torch.stack(entity_masks)
        entity_sizes = torch.tensor(entity_sizes, dtype=torch.long)
        entity_spans = torch.tensor(entity_spans, dtype=torch.long)
        entity_sample_masks = torch.ones([entity_masks.shape[0]], dtype=torch.bool)
    else:
        # corner case handling (no pos/neg entities)
        entity_types = torch.zeros([1], dtype=torch.long)
        entity_masks = torch.zeros([1, max_n_word], dtype=torch.bool)
        entity_sizes = torch.zeros([1], dtype=torch.long)
        entity_spans = torch.zeros([1, 2], dtype=torch.long)
        entity_sample_masks = torch.zeros([1], dtype=torch.bool)

    if cfg.use_gcn:
        ##############################################################################
        # subgraph for each sampled span or entity
        # subgraph nodes: BFS-style beam search
        # span weights to positive entities: (the first order)
        sgs_nodes = [] # tobe: List[List[Int]]
        sgs_node_types = [] # tobe: List[List[Int]]
        spans_weights = [] # tobe: List[List[Float]]

        sgs_nodes_char_ids = [] # tobe: List[List[Tuple[List[Int]]]], n_span, n_node,
        # n_word, n_char
        sgs_nodes_word_ids = [] # tobe: List[List[List[[Int]]], n_span, n_node, n_word
        # sgs_nodes_lm_spans = [] # tobe: List[List[Tuple[Int]]]
        sgs_nodes_lm_spans = None # tobe: List[List[Tuple[Int]]]
        sgs_nodes_word_strs = [] # tobe: List[List[Tuple[Str]]], n_span, n_node, n_word
        sgs_nodes_size = [] # tobe: List[List[Int]]

        for sp in (pos_entity_phrases + list(neg_entity_phrases)):
            _span_cands = span_cands[sp][:cfg.n_neighbor]

            sg_nodes = [n["id"] for n in _span_cands]
            beam_search(
                nodes=sg_nodes,
                cands=entity_cands,
                graph_neighbors=cfg.graph_neighbors,
            )
            sg_nodes = tuple(set(sg_nodes))

            _span_weights = []
            for n in _span_cands:
                _span_weights.append((sg_nodes.index(n["id"]), n["weight"]))
            spans_weights.append(tuple(_span_weights))
            sgs_nodes.append(sg_nodes)

            sgs_node_types.append([node2type_idx[n] for n in sg_nodes])
            sgs_nodes_char_ids.append([list(cfg.node2char_ids[n]) for n in sg_nodes])
            sgs_nodes_word_ids.append([list(cfg.node2word_ids[n]) for n in sg_nodes])
            # sgs_nodes_lm_spans.append([cfg.node2lm_spans[n] for n in sg_nodes])
            if cfg.use_lm_embed:
                sgs_nodes_word_strs.append([cfg.node2word_strs[n] for n in sg_nodes])
            if cfg.use_size:
                sgs_nodes_size.append([cfg.node2size[n] for n in sg_nodes])

        sgs = [] # tobe: List[Graph]
        batch_indices = batch_indices_by_graph_edges(sgs_nodes, graph, cfg.max_batch_edges)
        # batch_indices = [(0, len(sgs_nodes)-1)]
        for b, e in batch_indices:
            sgs_in_be = build_subgraph_for_spans_with_only_entity(
                graph, sgs_nodes[b: e + 1], spans_weights[b: e + 1], cfg.use_graph_weight
            )
            sgs.append((b, e, dgl.batch(sgs_in_be)))

        gold_sgs_node_types = copy.deepcopy(sgs_node_types[:len(pos_entity_phrases)])

        n_span = len(sgs_nodes)
        max_n_nodes = 0
        max_n_bpes = 0
        sgs_nodes_bpes = [] # tobe: List[List[List[Int]]]
        sgs_nodes_bpes_len = [] # tobe: List[List[Int]]
        sgs_nodes_bpes_tensor = None
        # for i in range(n_span):
        #     len_sgs_nodes_i = len(sgs_nodes[i])
        #     max_n_nodes = max(max_n_nodes, len_sgs_nodes_i)
        #     bpes_idx = []
        #     bpes_len = []
        #     for j in range(len_sgs_nodes_i):
        #         node_idx = sgs_nodes[i][j]
        #         max_n_bpes = max(max_n_bpes, len(node_idx2bpe_idx[node_idx]))
        #         bpes_idx.append(node_idx2bpe_idx[node_idx])
        #         bpes_len.append(len(node_idx2bpe_idx[node_idx]))
        #     sgs_nodes_bpes.append(bpes_idx)
        #     sgs_nodes_bpes_len.append(bpes_len)
        #
        # sgs_nodes_bpes_tensor = torch.zeros(
        #     [n_span, max_n_nodes, max_n_bpes], dtype=torch.long
        # )
        # for i in range(len(sgs_nodes_bpes)):
        #     for j in range(len(sgs_nodes_bpes[i])):
        #         len_ij = len(sgs_nodes_bpes[i][j])
        #         sgs_nodes_bpes_tensor[i, j, :len_ij] = torch.LongTensor(sgs_nodes_bpes[i][j])

        # token_strings = [t.phrase for t in doc.tokens]
        sgs_nodes = tuple(sgs_nodes)
        bpe_strings = tuple(doc.bpes)

        sgs_nodes_char_ids = [
            [pad_char_ids(node, pad_id=cfg.char2idx['[PAD]'],
                          max_n_char=max_n_char, max_n_word=cfg.max_ent_n_word)
            for node in span]
        for span in sgs_nodes_char_ids] # n_span, n_node, n_word, n_char

        sgs_nodes_word_ids = [
            [pad_word_ids(node, pad_id=cfg.word2idx['[PAD]'],
                          max_n_word=cfg.max_ent_n_word) for node in span]
        for span in sgs_nodes_word_ids ] # n_span, n_node, n_word

        ##############################################################################
    else:
        sgs = None
        sgs_nodes = None
        sgs_node_types = None
        bpe_strings = None
        gold_sgs_node_types = None
        sgs_nodes_bpes = None
        sgs_nodes_bpes_tensor = None
        sgs_nodes_bpes_len = None
        sgs_nodes_char_ids = None
        sgs_nodes_word_ids = None
        sgs_nodes_lm_spans = None
        sgs_nodes_word_strs = None
        sgs_nodes_size = None

    return dict(
        char_ids=char_ids, word_ids=word_ids,
        lm_spans=lm_spans, word_strs=word_strs,
        bpe_ids=bpe_ids, lm_attn_masks=lm_attn_masks,
        entity_masks=entity_masks, entity_sizes=entity_sizes,
        entity_types=entity_types, entity_sample_masks=entity_sample_masks,
        entity_spans=entity_spans,
        sgs=sgs, sgs_nodes=sgs_nodes, sgs_node_types=sgs_node_types,
        bpe_strings=bpe_strings, gold_sgs_node_types=gold_sgs_node_types,
        sgs_nodes_bpes=sgs_nodes_bpes, sgs_nodes_bpes_tensor=sgs_nodes_bpes_tensor,
        sgs_nodes_bpes_len=sgs_nodes_bpes_len,
        sgs_nodes_char_ids=sgs_nodes_char_ids,
        sgs_nodes_word_ids=sgs_nodes_word_ids,
        sgs_nodes_lm_spans=sgs_nodes_lm_spans,
        sgs_nodes_word_strs=sgs_nodes_word_strs,
        sgs_nodes_size=sgs_nodes_size,
    )

def create_third_sample(
    cfg, doc, node_str2idx, idx2entity_type
):
    """
        For building node embeds.
    """

    max_n_word = cfg.max_n_word
    max_n_char = 128

    # positive entities
    entity_spans, entity_types, entity_masks, entity_sizes, entity_ids = [], [], [], [], []
    for e in doc.entities:
        entity_spans.append(e.span)
        entity_types.append(e.entity_type)
        entity_masks.append(utils.create_entity_mask(*e.span, max_n_word))
        entity_sizes.append(len(e.tokens))
        entity_ids.append(node_str2idx[e.phrase + " " + idx2entity_type[e.entity_type]])

    # create tensors
    # token indices
    bpe_ids = torch.zeros(max_n_word, dtype=torch.long)
    bpe_ids[:len(doc.bpe_ids)] = torch.tensor(doc.bpe_ids, dtype=torch.long)

    # masking of tokens
    lm_attn_masks = torch.ones(max_n_word, dtype=torch.bool)

    # also create samples_masks:
    # tensors to mask entity/relation samples of batch
    # since samples are stacked into batches, "padding" entities/relations possibly must be created
    # these are later masked during loss computation
    if entity_masks:
        entity_types = torch.tensor(entity_types, dtype=torch.long)
        entity_masks = torch.stack(entity_masks)
        entity_sizes = torch.tensor(entity_sizes, dtype=torch.long)
        entity_sample_masks = torch.ones([entity_masks.shape[0]], dtype=torch.bool)
    else:
        # corner case handling (no pos/neg entities)
        entity_types = torch.zeros([1], dtype=torch.long)
        entity_masks = torch.zeros([1, max_n_word], dtype=torch.bool)
        entity_sizes = torch.zeros([1], dtype=torch.long)
        entity_sample_masks = torch.zeros([1], dtype=torch.bool)

    return dict(
        bpe_ids=bpe_ids, lm_attn_masks=lm_attn_masks,
        entity_masks=entity_masks, entity_sizes=entity_sizes,
        entity_types=entity_types, entity_sample_masks=entity_sample_masks,
        entity_ids=entity_ids
    )


def collate_fn_padding(batch):
    new_batch = dict()
    keys = batch[0].keys()

    for key in keys:
        samples = [s[key] for s in batch]
        if key in [
            "lm_spans", "word_strs", \
            "sgs", "sgs_nodes", "entity_ids", "bpe_strings", \
            "sgs_node_types", "gold_sgs_node_types", "sgs_nodes_bpes", \
            "sgs_nodes_bpes_tensor", "sgs_nodes_bpes_len", \
            "sgs_nodes_char_ids", "sgs_nodes_word_ids", \
            "sgs_nodes_lm_spans", "sgs_nodes_word_strs",
            "sgs_nodes_size",
        ]:
            new_batch[key] = samples
        elif hasattr(batch[0][key], "shape"):
            if not batch[0][key].shape:
                new_batch[key] = torch.stack(samples)
            else:
                new_batch[key] = utils.padded_stack(samples)
        else:
            new_batch[key] = samples

    return new_batch
