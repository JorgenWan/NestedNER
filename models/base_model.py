import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_, clip_grad_value_

from models.utils import calculate_ner_match, calculate_prf

from modules.utils import Embedding

class Base_Model(nn.Module):

    @classmethod
    def build_model(cls, cfg):
        return cls(cfg)

    def __init__(self, cfg):
        super().__init__()

        self.char_embed_dim = cfg.char_embed_dim
        self.input_dim = self.char_embed_dim

        self.dropout = cfg.dropout
        self.sentence_average = cfg.sentence_average

        self.clip_norm = cfg.clip_norm
        self.clip_value = cfg.clip_value

    def forward(self, sample):
        """ to be overrided """
        raise NotImplementedError

    def _forward_logits(self, sample):
        """ to be overrided """
        raise NotImplementedError

    def evaluate(self, sample, idx2label):
        """ to be overrided """
        raise NotImplementedError

    def aggregate_log_outputs(self, log_outputs):
        """ to be overrided """
        raise NotImplementedError

    def clip_gradient(self):
        if self.clip_norm > 0:
            clip_grad_norm_(self.parameters(), self.clip_norm)
        if self.clip_value > 0:
            clip_grad_value_(self.parameters(), self.clip_value)

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument('--num_layers', type=int)
        parser.add_argument('--dropout', type=float) 
