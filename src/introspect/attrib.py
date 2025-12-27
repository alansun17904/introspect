"""Attribution experiment.

Handles build, train, and eval actions for attribution experiments.
"""
import os
import pandas as pd
import tqdm
from torch.utils.data import random_split
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, TaskType, get_peft_model, PeftConfig, PeftModel
from omegaconf import OmegaConf

from .input_ablation.preparation import generate_outputs_df, HERE, filter_outputs
from .input_ablation.dataset import AttribDataset
from .input_ablation.training import make_trainer
from .utils import set_seed


def build_action(cfg):
    """Build attribution data."""
    hf_name = cfg.hf_name

    
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(hf_name)
    tokenizer = AutoTokenizer.from_pretrained(hf_name)
    
    # Set tokenizer settings
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Generate outputs
    if not os.path.exists(HERE / "input_ablation_data.csv"):
        df = generate_outputs_df(model, tokenizer, max_tokens=cfg.max_tokens, batch_size=cfg.batch_size, max_samples=cfg.train_data_size)
        output_file = HERE / "input_ablation_data.csv"
        df.to_csv(output_file, index=False)
    else:
        df = pd.read_csv(HERE / "input_ablation_data.csv")
    df = filter_outputs(df)
    df.to_csv(HERE / "filtered_ablation_data.csv", index=False)

def train_action(cfg):
    """Train attribution model."""
    hf_name = cfg.hf_name
    tokenizer = AutoTokenizer.from_pretrained(hf_name)
    model = AutoModelForCausalLM.from_pretrained(hf_name)
    if cfg.peft is not None:
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            **OmegaConf.to_container(cfg.peft_config, resolve=True),
        )
        model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = AttribDataset(cfg)
    val_split = getattr(cfg, "val_split", 0.1)
    test_split = getattr(cfg, "test_split", 0.1)
    train, val, test = random_split(
        dataset, [1 - val_split - test_split, val_split, test_split]
    )
    trainer = make_trainer(cfg, model, tokenizer, train, val, test)
    trainer.train()


def eval_action(cfg):
    """Evaluate attribution model."""
    dataset = AttribDataset(cfg)
    val_split = getattr(cfg, "val_split", 0.1)
    test_split = getattr(cfg, "test_split", 0.1)
    _, _, test = random_split(
        dataset, [1 - val_split - test_split, val_split, test_split]
    )
    dataset = test
    config = PeftConfig.from_pretrained(cfg.checkpoint_dir)
    model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    lora_model = PeftModel.from_pretrained(model, cfg.checkpoint_dir)
    lora_model.eval()

    for chat in tqdm.tqdm(dataset, desc="Evaluating"):
        prompt = tokenizer.apply_chat_template(
            [chat[0]], tokenize=False, add_generation_prompt=True
        )
        generated = tokenizer.decode(
            lora_model.generate(
                **tokenizer(prompt, return_tensors="pt").to(lora_model.device),
                max_new_tokens=cfg.max_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )[0],
            skip_special_tokens=False,
        )
        print(f"Generated: {generated}")
        print(f"Ground truth: {chat[1]['content']}")
        print("-" * 100)


def run(action, cfg):
    """Run the specified action for the attribution experiment."""
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
