import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from criterions.base_criterion import Base_Criterion
from . import register_criterion

@register_criterion("cross_entropy")
class Cross_Entropy(Base_Criterion):

    def __init__(self, cfg):
        super().__init__()

        self.pad_idx = cfg.src_dict.pad_idx
        self.sentence_average = cfg.sentence_average

    def forward(self, model, sample):

        net_output = model(
            src_tokens=sample["src_tokens"],
            src_lengths=sample["src_lengths"],
            prev_tgt_tokens=sample["prev_tgt_tokens"]
        )
        log_probs = F.log_softmax(net_output["tgt_logits"], dim=-1)
        log_probs = log_probs.view(-1, log_probs.size(-1))

        nll_loss = F.nll_loss(
            log_probs, sample["tgt_tokens"].view(-1),
            ignore_index=self.pad_idx, reduction='sum'
        )

        loss = nll_loss
        sample_size = sample["num_sentences"] if self.sentence_average else sample["num_tokens"]
        log_output = {
            "loss": loss.data.item(),
            "nll_loss": nll_loss.data.item(),
            "sample_size": sample_size,
            "num_sentences": sample["num_sentences"],
            "num_tokens": sample["num_tokens"]
        }

        return loss, sample_size, log_output

    def aggregate_log_outputs(self, log_outputs):
        """Aggregate log outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in log_outputs)
        nll_loss_sum = sum(log.get('nll_loss', 0) for log in log_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in log_outputs)
        num_sentences = sum(log.get('num_sentences', 0) for log in log_outputs)
        num_tokens = sum(log.get('num_tokens', 0) for log in log_outputs)
        aggregate_output = {
            "loss": loss_sum,
            "nll_loss": nll_loss_sum,
            "sample_size": sample_size,
            "num_sentences": num_sentences,
            "num_tokens": num_tokens
        }
        return aggregate_output
