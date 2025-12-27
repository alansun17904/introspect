"""Feature description experiment.

Handles build, train, and eval actions for feature description experiments.
"""
import os
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from .feature_desc.preparation import build, load
from .feature_desc.training import make_trainer, generate
from .utils import set_seed


def build_action(cfg):
    build(cfg)


def train_action(cfg):
    model, tokenizer, train, val, test = load(cfg)
    trainer = make_trainer(cfg, model, tokenizer, train, val, test)
    trainer.train()


def eval_action(cfg):
    _, _, _, _, test = load(cfg)
    config = PeftConfig.from_pretrained(cfg.checkpoint_dir)
    model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
    tokenizer.add_special_tokens(
        {"additional_special_tokens": ["[FEATURE]", "[LAYER]"]}
    )
    model.resize_token_embeddings(len(tokenizer))
    lora_model = PeftModel.from_pretrained(model, cfg.checkpoint_dir)
    lora_model.eval()
    for prompt in test:
        generated = generate(lora_model, tokenizer, prompt[3], prompt[1])
        print(f"Generated: {generated}")
        print(f"Ground truth: {prompt[0]}")
        print("-" * 100)


def run(action, cfg):
    os.environ["WANDB_TAGS"] = ",".join(cfg.experiment.tags)
    if action == "build":
        build_action(cfg)
    elif action == "train":
        train_action(cfg)
    elif action == "eval":
        eval_action(cfg)
    else:
        raise ValueError(f"Invalid action: {action}")
