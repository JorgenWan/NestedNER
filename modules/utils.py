import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import numpy as np

from collections import defaultdict

def if_nan_exists(tensor):
    x = torch.isnan(tensor).sum()
    if x > 0:
        return True
    return False

uni_range = 1.0

def Embedding(num_embeds, embed_dim, pad_idx):
    e = nn.Embedding(num_embeds, embed_dim, padding_idx=pad_idx)
    scale = np.sqrt(3.0 / embed_dim)
    nn.init.uniform_(e.weight, -scale, scale)
    nn.init.constant_(e.weight[pad_idx], 0)

    return e

def Linear(input_dim, output_dim, bias=False):
    l = nn.Linear(input_dim, output_dim, bias=bias)
    scale = np.sqrt(3.0 / (input_dim+output_dim))
    l.weight.data.uniform_(-scale, scale)
    if bias:
        l.bias.data.uniform_(-scale, scale)
    return l

def Cross_Entropy(outputs, labels, sample_masks, is_logits=True, pad_idx=None):
    """
    Input:
         outputs: bsz × ? × output
         labels: bsz × ?
         ? means this dim may not exist
    """
    if is_logits:
        log_probs = F.log_softmax(outputs, dim=-1)
    else:
        log_probs = torch.log(outputs + 1e-13)
    log_probs = log_probs.view(-1, log_probs.size(-1))

    nll_loss = F.nll_loss(
        log_probs, labels.view(-1),
        ignore_index=pad_idx, reduction='none'
    )
    nll_loss = (nll_loss * sample_masks).sum()
    # nll_loss = F.nll_loss(
    #     log_probs, labels.view(-1),
    #     ignore_index=pad_idx, reduction='mean'
    # )

    return nll_loss

def Weighted_Cross_Entropy(outputs, labels, is_logits=True, pad_idx=None, O_idx=None, weight_ratio=None):
    """
    Input:
         outputs: bsz × ? × output
         labels: bsz × ?
         ? means this dim may not exist
    """

    if is_logits:
        log_probs = F.log_softmax(outputs, dim=-1)
    else:
        log_probs = torch.log(outputs)
    log_probs = log_probs.view(-1, log_probs.size(-1))

    nll_loss = F.nll_loss(
        log_probs, labels.view(-1),
        ignore_index=pad_idx, reduction='none'
    )

    mask1 = labels.ne(O_idx)
    mask2 = labels.ne(pad_idx)
    weight_before_norm = mask2 * (mask1 + (1 - weight_ratio) / (2 * weight_ratio - 1))

    normalizer = weight_before_norm.sum(dim=-1) / (mask2.sum(dim=-1) + 1e-13) + 1e-13
    weight = (weight_before_norm.t() / normalizer).t()
    nll_loss = nll_loss * weight.view(-1)
    nll_loss = nll_loss.sum()

    return nll_loss

def Weighted_Logit(logits, labels, pad_idx=None, O_idx=None, weight_ratio=None):
    batch_size, seq_len, label_size = logits.size()

    # todo: here we assume that <PAD> <BOS> <EOS> all before O in label dict
    new_labels = (labels > O_idx) * labels
    new_labels_for_entity = F.one_hot(new_labels, num_classes=label_size).float()
    new_labels_for_entity[:, :, 0] = 0

    new_labels_for_others = ((labels <= O_idx).float() * float(1 / label_size) ).unsqueeze(-1).expand(logits.size())

    new_labels_ = new_labels_for_entity + new_labels_for_others

    weight_before_norm = new_labels_ + (1 - weight_ratio) / (2 * weight_ratio - 1)
    normalizer = weight_before_norm.sum(dim=-1) / (float(label_size) * new_labels_.sum(dim=-1) + 1e-13) + 1e-13
    weight = weight_before_norm / normalizer.view(batch_size, seq_len, 1)

    wlogits = logits * weight
    return wlogits

def Layer_Norm(normalized_shape, eps=1e-5, elementwise_affine=True):
    return nn.LayerNorm(normalized_shape, eps, elementwise_affine)

def make_positions(tensor, pad_idx, left_pad):
    batch_size, seq_len = tensor.size()

    max_length = pad_idx + 1 + seq_len
    if not hasattr(make_positions, 'range_buf'):
        make_positions.range_buf = tensor.new()
    make_positions.range_buf = make_positions.range_buf.type_as(tensor)
    if make_positions.range_buf.numel() < max_length:
        torch.arange(pad_idx + 1, max_length, out=make_positions.range_buf)

    mask = tensor.ne(pad_idx)
    positions = make_positions.range_buf[:seq_len].expand_as(tensor)
    if left_pad:
        positions = positions - mask.size(1) + mask.long().sum(dim=1).unsqueeze(1)
    positions = tensor.clone().masked_scatter_(mask, positions[mask])

    return positions


INCREMENTAL_STATE_INSTANCE_ID = defaultdict(lambda: 0)

def _get_full_incremental_state_key(module_instance, key):
    """
    assign a unique ID to each module instance, so that
    incremental state is not shared across module instances
    """

    module_name = module_instance.__class__.__name__
    if not hasattr(module_instance, '_instance_id'):
        INCREMENTAL_STATE_INSTANCE_ID[module_name] += 1
        module_instance._instance_id = INCREMENTAL_STATE_INSTANCE_ID[module_name]

    full_key = f"{module_name}.{module_instance._instance_id}.{key}"

    return full_key


def get_incremental_state(module, incremental_state, key):
    """Helper for getting incremental state for an nn.Module."""

    full_key = _get_full_incremental_state_key(module, key)
    if incremental_state is None or full_key not in incremental_state:
        return None

    return incremental_state[full_key]

def set_incremental_state(module, incremental_state, key, value):
    """Helper for setting incremental state for an nn.Module."""
    if incremental_state is not None:
        full_key = _get_full_incremental_state_key(module, key)
        incremental_state[full_key] = value


def log_sum_exp(scores):
    """
    Calculate the log_sum_exp with trick.
    This is to avoid underflow or overflow.
    :param scores: bsz × from_label × to_label
    :return: bsz × to_label
    """
    batch_size, from_label_size, to_label_size = scores.size()

    max_scores, _ = torch.max(scores, 1)
    expanded_max_scores = max_scores.view(batch_size, 1, to_label_size)\
                          .expand(batch_size, from_label_size, to_label_size)
    x = torch.exp(scores - expanded_max_scores)
    x = torch.sum(x, 1)
    x = torch.log(x)
    x += max_scores

    return x

def combine_bidirection(outs, num_layers, batch_size):
    """
    # num_layer*2 bsz hid -> num_layer bsz hid*2
    """
    if outs is None:
        return None

    out = outs.view(num_layers, 2, batch_size, -1).transpose(1, 2) \
        .contiguous().view(num_layers, batch_size, -1)
    return out

def pack_pad_forward(rnn, x, lengths, need_hidden_state=False, need_output=False):

    sorted_len, perm_idx = lengths.sort(0, descending=True)
    _, recover_idx = perm_idx.sort(0, descending=False)
    sorted_x = x[perm_idx]

    packed_x = pack_padded_sequence(sorted_x, sorted_len, batch_first=True)
    output, hidden = rnn(packed_x, None)

    assert need_hidden_state ^ need_output
    if need_hidden_state:
        hidden_state = hidden[0].transpose(1, 0).contiguous().view(x.size(0), -1)
        return hidden_state[recover_idx]
    if need_output:
        output, _ = pad_packed_sequence(output, batch_first=True)
        return output[recover_idx]