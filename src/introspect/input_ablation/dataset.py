import re
import os
import pandas as pd
from pathlib import Path
from datasets import load_dataset
from torch.utils.data import Dataset
import torch


CHOICES = "ABCD"
ASK = "If the hint were removed how would the model's answer change?"
NO_CHANGE = "The output would remain {}"
CHANGE = "The output would change to {}"
HERE = Path(__file__).resolve().parent / "artifacts"
MMLU_PROMPT_RE = re.compile(
    r"Question:\s*(.*?)\s*Choices:\s*(.*?)\s*Hint:\s*([^\n<]+)", re.S
)


class MMLUDataset(Dataset):
    def __init__(self, show_hints=True):
        self.ds = load_dataset("cais/mmlu", "all", split="test")
        self.show_hints = show_hints

    def __len__(self):
        return len(self.ds)
    
    @staticmethod
    def make_user_content(question, choices, hint):
        return (
            "The following is a multiple choice question (with a correct answer). "
            "Output only (A, B, C, or D) and nothing else, in the format Answer: x, "
            "where x is one of A, B, C, or D.\n"
            f"Question: {question}\nChoices: {choices}\nHint: {hint}"
        )

    @staticmethod
    def make_chat_template(question, choices, hint):
        return [
            {
                "role": "user",
                "content": MMLUDataset.make_user_content(question, choices, hint),
            },
        ]

    def __getitem__(self, idx):
        choices = "\n".join([f"{CHOICES[i]}: {c}" for i, c in enumerate(self.ds[idx]["choices"])])
        question = self.ds[idx]["question"]
        hint = CHOICES[idx % len(CHOICES)]
        chat = self.make_chat_template(question, choices, hint)
        if not self.show_hints:
            chat[0]["content"] = re.sub(r"Hint: .*", "", chat[0]["content"])
        return chat


class AttribDataset(Dataset):
    def __init__(self, cfg):
        if cfg.small:
            fname = "small_ablation.csv"
        else:
            fname = "filtered_ablation_data.csv"
        if not os.path.exists(HERE / fname):
            raise RuntimeError(
                f"Filtered ablation data not found. Please run `introspect attrib build` to create it."
            )
        self.df = pd.read_csv(HERE / fname)

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        match = MMLU_PROMPT_RE.search(row["prompt"])
        if match is None:
            raise ValueError("Prompt missing expected MMLU fields: Question/Choices/Hint.")
        question, choices, hint = (group.strip() for group in match.groups())
        user_content = MMLUDataset.make_user_content(question, choices, hint) + "\n" + ASK
        if row["output"] == row["output_hint"]:
            assistant_content = NO_CHANGE.format(row["output"])
        else:
            assistant_content = CHANGE.format(row["output"])
        return [
            {"role": "user", "content": user_content},
            {"role": "model", "content": assistant_content},
        ]


def attrib_collate_fn(model, tokenizer, batch):
    formatted_chats = tokenizer.apply_chat_template(
        batch, tokenize=False, add_generation_prompt=False
    )
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        toks = tokenizer(
            formatted_chats,
            padding=True,
            return_tensors="pt",
        )
    finally:
        tokenizer.padding_side = original_padding_side

    marker_ids = tokenizer.encode("<start_of_turn>model", add_special_tokens=False)
    if not marker_ids:
        raise ValueError("Tokenization for <start_of_turn>model returned empty ids.")

    model_start_pos = []
    input_ids_list = toks.input_ids.tolist()
    for ids in input_ids_list:
        last_pos = None
        for i in range(len(ids) - len(marker_ids), -1, -1):
            if ids[i : i + len(marker_ids)] == marker_ids:
                last_pos = i + len(marker_ids) - 1
                break
        if last_pos is None:
            raise ValueError("Could not find <start_of_turn>model in tokenized input.")
        model_start_pos.append(last_pos)

    return {
        "input_ids": toks.input_ids,
        "attention_mask": toks.attention_mask,
        "labels": toks.input_ids,
        "model_start_pos": torch.tensor(model_start_pos, dtype=torch.long),
    }


def mmlu_collate_fn(batch, tokenizer):
    formatted_chats = tokenizer.apply_chat_template(batch, tokenize=False, add_generation_prompt=True)
    return formatted_chats

