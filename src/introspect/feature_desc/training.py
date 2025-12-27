# given logits, input ids and a "start" index, compute the next-token prediction loss
import torch
import torch.nn as nn
from functools import partial
import torch.nn.functional as F
from transformers import Trainer, TrainingArguments
from transformers.modeling_utils import unwrap_model
from pathlib import Path
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from ..utils import patch_embeds, make_trainer_base, base_generate
from .dataset import collate_fn


HERE = Path(__file__).resolve().parent / "artifacts"


def compute_loss(model, inputs, feature_id, return_outputs=False, **kwargs):
    toks = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    feature_v = inputs["feature_v"]
    labels = inputs["labels"]
    desc_pos = inputs["desc_pos"]

    # Inject the feature vector into the input embeddings
    mask = toks == feature_id
    feature_idx = torch.nonzero(mask, as_tuple=False)

    base = unwrap_model(model)
    handle = base.get_input_embeddings().register_forward_hook(
        partial(patch_embeds, feature_idx=feature_idx, feature_vector=feature_v)
    )
    logits = model(input_ids=toks, attention_mask=attention_mask).logits
    handle.remove()

    pos = (
        torch.arange(labels.shape[1], device=labels.device)
        .unsqueeze(0)
        .expand(*labels.shape)
    )
    ignore = pos < desc_pos.unsqueeze(1)
    labels[ignore] = -100  # default torch ignore index value
    loss = F.cross_entropy(
        logits[:, :-1, :].contiguous().permute(0, 2, 1),
        labels[:, 1:],
        reduction="mean",
        ignore_index=-100,
    )
    if return_outputs:
        return loss, {
            "logits": logits.argmax(dim=-1),
            "labels": labels,
            "desc_pos": desc_pos,
        }
    return loss


def generate(model, tokenizer, prompt, feature_v, max_new_tokens=10):
    return base_generate(model, tokenizer, prompt, feature_v, max_new_tokens)


def make_trainer(cfg, model, tokenizer, train_dataset, val_dataset, test_dataset):
    return make_trainer_base(
        cfg=cfg,
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        compute_loss_fn=compute_loss,
        collate_fn=collate_fn,
        feature_token_id=tokenizer.convert_tokens_to_ids("[FEATURE]"),
        output_dir_base=HERE,
    )
