"""Activation experiment.

Handles build, train, and eval actions for activation experiments.
"""
import os
import re
import tqdm
import torch
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformer_lens import HookedTransformer

from .activation_patch.preparation import (
    make_train_data,
    process_true_false,
    HERE,
    filter_train_data,
    load,
)
from .activation_patch.training import make_trainer, generate
from .utils import set_seed


def build_action(cfg):
    """Build activation data."""
    model = HookedTransformer.from_pretrained(cfg.hf_name)
    tokenizer = AutoTokenizer.from_pretrained(cfg.hf_name)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    if not os.path.exists(HERE / "true_false_prompts.csv"):
        process_true_false()
        make_train_data(model, tokenizer, cfg)
    if not os.path.exists(HERE / "train_data.pkl"):
        make_train_data(model, tokenizer, cfg)
    filter_train_data(cfg)


def train_action(cfg):
    """Train activation model."""
    model, tokenizer, train, val, test = load(cfg)
    trainer = make_trainer(cfg, model, tokenizer, train, val, test)
    trainer.train()


def eval_action(cfg):
    """Evaluate activation model."""
    _, _, _, _, test = load(cfg)
    config = PeftConfig.from_pretrained(cfg.checkpoint_dir)
    model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
    tokenizer.add_special_tokens(
        {"additional_special_tokens": ["[FEATURE]", "[LAYER]", "[PATCHED_OUTPUT]"]}
    )
    model.resize_token_embeddings(len(tokenizer))
    lora_model = PeftModel.from_pretrained(model, cfg.checkpoint_dir)
    lora_model.eval()

    for prompt_item in tqdm.tqdm(test, desc="Evaluating"):
        prompt, patching_vector, _ = prompt_item
        # Remove the answer
        answer_prompt = "The output would be:"
        answer_idx = prompt.index("The output would be:")
        skeleton = prompt[: answer_idx + len(answer_prompt)]

        # Convert patching_vector to tensor if it's not already
        if not isinstance(patching_vector, torch.Tensor):
            patching_vector = torch.tensor(patching_vector, dtype=torch.float32)
        generated = generate(lora_model, tokenizer, skeleton, patching_vector)
        print(f"Generated: {generated[answer_idx + len(answer_prompt):]}")
        print(f"Ground truth: {prompt[answer_idx + len(answer_prompt):]}")
        print("-" * 100)


def run(action, cfg):
    """Run the specified action for the activation experiment."""
    os.environ["WANDB_TAGS"] = ",".join(cfg.experiment.tags)
    set_seed(cfg.seed)

    if action == "build":
        build_action(cfg)
    elif action == "train":
        train_action(cfg)
    elif action == "eval":
        eval_action(cfg)
    else:
        raise ValueError(f"Invalid action: {action}")
