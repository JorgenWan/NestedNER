import torch
import numpy as np
import networkx as nx

def get_token(h: torch.tensor, x: torch.tensor, token: int):
    """ Get specific token embedding (e.g. [CLS]) """
    emb_size = h.shape[-1]

    token_h = h.view(-1, emb_size)
    flat = x.contiguous().view(-1)

    # get contextualized embedding of given token
    token_h = token_h[flat == token, :]

    return token_h

def create_entity_mask(start, end, context_size):
    mask = torch.zeros(context_size, dtype=torch.bool)
    mask[start:end + 1] = 1 # default: 'end' is inclusive

    return mask

def extend_tensor(tensor, extended_shape, fill=0):
    tensor_shape = tensor.shape

    extended_tensor = torch.zeros(extended_shape, dtype=tensor.dtype).to(tensor.device)
    extended_tensor = extended_tensor.fill_(fill)

    if len(tensor_shape) == 1:
        extended_tensor[:tensor_shape[0]] = tensor
    elif len(tensor_shape) == 2:
        extended_tensor[:tensor_shape[0], :tensor_shape[1]] = tensor
    elif len(tensor_shape) == 3:
        extended_tensor[:tensor_shape[0], :tensor_shape[1], :tensor_shape[2]] = tensor
    elif len(tensor_shape) == 4:
        extended_tensor[:tensor_shape[0], :tensor_shape[1], :tensor_shape[2], :tensor_shape[3]] = tensor

    return extended_tensor


def padded_stack(tensors, pad_idx=0):
    dim_count = len(tensors[0].shape)

    max_shape = [max([t.shape[d] for t in tensors]) for d in range(dim_count)]
    padded_tensors = []

    for t in tensors:
        e = extend_tensor(t, max_shape, fill=pad_idx)
        padded_tensors.append(e)

    stacked = torch.stack(padded_tensors)
    return stacked

def collate_fn_padding(batch, pad_idx):
    padded_batch = dict()
    keys = batch[0].keys()

    for key in keys:
        samples = [s[key] for s in batch]

        if not batch[0][key].shape:
            padded_batch[key] = torch.stack(samples)
        else:
            padded_batch[key] = padded_stack([s[key] for s in batch], pad_idx=pad_idx)

    return padded_batch

def get_A_hat(G):
    """
    G: networkx.Graph
    """
    A = nx.to_numpy_matrix(G, weight="weight")
    A = A + np.eye(G.number_of_nodes())
    Degrees = []
    for d in G.degree(weight=None):
        if d[1] == 0:
            Degrees.append(0)
        else:
            Degrees.append(d[1] ** (-0.5))
    D = np.diag(Degrees)
    A_hat = D @ A @ D
    return A_hat