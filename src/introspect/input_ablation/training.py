import torch
import torch.nn.functional as F
from pathlib import Path

from ..utils import make_trainer_base
from .dataset import attrib_collate_fn


HERE = Path(__file__).resolve().parent / "artifacts"


def compute_loss(model, inputs, feature_id, return_outputs=False, **kwargs):
    toks = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    labels = inputs["labels"].clone()
    model_start_pos = inputs["model_start_pos"]

    logits = model(input_ids=toks, attention_mask=attention_mask).logits

    pos = (
        torch.arange(labels.shape[1], device=labels.device)
        .unsqueeze(0)
        .expand_as(labels)
    )
    labels[pos <= model_start_pos.unsqueeze(1)] = -100
    labels[attention_mask == 0] = -100

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
            "model_start_pos": model_start_pos,
        }
    return loss


def make_trainer(cfg, model, tokenizer, train_dataset, val_dataset, test_dataset):
    return make_trainer_base(
        cfg=cfg,
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        compute_loss_fn=compute_loss,
        collate_fn=attrib_collate_fn,
        feature_token_id=tokenizer.eos_token_id,
        output_dir_base=HERE,
    )
