"""Common utility functions for the introspect package."""
import torch
import numpy as np
import random
import torch.utils.data as data
from functools import partial
from pathlib import Path
from transformers import Trainer, TrainingArguments, DynamicCache


def set_seed(seed):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def to_dataloader(subset, collate_fn, batch_size, num_workers=0, **kwargs):
    return data.DataLoader(
        subset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate_fn,
        **kwargs,
    )


def patch_embeds(module, inp, out, feature_idx, feature_vector):
    """Patch embeddings at `feature_idx` with `feature_vector`.

    Args:
        feature_idx: Index of the feature tokens has dimension (batch_size x 2)
            feature_idx[:, 0] is batch index (arange(batch_size))
            feautre_idx[:, 1] is token index that we are patching in feature_vector
        feature_vector: dimension (batch_size, d_model)
    """
    out[feature_idx[:, 0], feature_idx[:, 1], :] = feature_vector[feature_idx[:, 0]].to(
        out.dtype
    )
    return out


def make_trainer_base(
    cfg,
    model,
    tokenizer,
    train_dataset,
    val_dataset,
    compute_loss_fn,
    collate_fn,
    feature_token_id,
    output_dir_base,
):
    """Create a generic Trainer with shared configuration.

    Args:
        cfg: Configuration object with training parameters
        model: The model to train
        tokenizer: The tokenizer
        train_dataset: Training dataset
        val_dataset: Validation dataset
        compute_loss_fn: Function to compute loss (will be partially applied)
        collate_fn: Collate function for data loader
        feature_token_id: Token ID for [FEATURE] token
        output_dir_base: Base directory for output (should be the artifacts directory)
    """
    fsdp_config = getattr(cfg, "fsdp", None)

    training_args = {
        "output_dir": output_dir_base / cfg.experiment.name / "checkpoints",
        "eval_strategy": "steps",
        "prediction_loss_only": True,
        "eval_steps": cfg.eval_steps,
        "logging_steps": cfg.logging_steps,
        "save_steps": cfg.save_steps,
        "num_train_epochs": cfg.epochs,
        "per_device_train_batch_size": cfg.batch_size,
        "per_device_eval_batch_size": cfg.batch_size,
        "report_to": "wandb",
        "save_strategy": "steps",
        "save_total_limit": cfg.save_total_limit,
        "run_name": cfg.experiment.name,
        "bf16": True,
    }

    if fsdp_config is not None and isinstance(fsdp_config, str):
        # consistently using "full_shard", need to adjust for more fine-grained
        # control over hyperparameters...this works for now
        training_args["fsdp"] = fsdp_config

    args = TrainingArguments(**training_args)

    # monkeypatch compute_loss function in Trainer class
    t = Trainer(
        args=args,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=partial(collate_fn, model, tokenizer),
    )
    t.compute_loss = partial(
        compute_loss_fn,
        feature_id=feature_token_id,
    )
    return t


@torch.inference_mode()
def base_generate(model, tokenizer, prompt, feature_vector, max_new_tokens=10):
    """Generate text with feature vector patching.

    Args:
        model: The model to use for generation
        tokenizer: The tokenizer
        prompt: Input prompt string (should contain [FEATURE] token)
        feature_vector: Feature vector to inject at [FEATURE] token positions
        max_new_tokens: Maximum number of new tokens to generate
    """
    feature_id = tokenizer.convert_tokens_to_ids("[FEATURE]")
    toks = tokenizer(prompt, return_tensors="pt").input_ids
    feature_idx = torch.nonzero(toks == feature_id, as_tuple=False)
    embeds = model.get_input_embeddings()(toks.to(model.device))
    embeds[feature_idx[:, 0], feature_idx[:, 1], :] = feature_vector.to(
        embeds.dtype
    ).to(model.device)
    cache = DynamicCache(config=model.config)
    cache = model(
        inputs_embeds=embeds, past_key_values=cache, use_cache=True, return_dict=True
    ).past_key_values
    generated_ids = model.generate(
        input_ids=toks,
        cache_position=torch.tensor([toks.shape[1]], device=model.device),
        max_new_tokens=max_new_tokens,
        past_key_values=cache,
        use_cache=True,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    return tokenizer.decode(generated_ids[0], skip_special_tokens=True)
