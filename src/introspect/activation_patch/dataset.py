import os
import json
import random
from pathlib import Path
import pickle
from functools import partial
import torch

from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np


HERE = Path(__file__).resolve().parent / "artifacts"
PROMPT_TEMPLATES = [
    "Input text: [Input Text]. If we steer layer [LAYER] towards [FEATURE], at tokens [TOKENS], how does affect the generated continuation? The output would be: [PATCHED_OUTPUT]",
    "If [FEATURE] at layer [LAYER] is added to tokens [TOKENS], when processing the text [Input Text], how would the output change? The output would be: [PATCHED_OUTPUT]",
    "Given the text [Input Text], what would be the effect on the output if [FEATURE] at layer [LAYER] is added to tokens [TOKENS]? The output would be: [PATCHED_OUTPUT]",
]


class ActivationPatchDataset(Dataset):
    def __init__(self, cfg, tokenizer):
        self.cfg = cfg
        if not os.path.exists(HERE / "filtered_train_data.pkl"):
            raise RuntimeError(
                "Filtered train data not found. Please run `introspect activation_patch build` to create it."
            )
        self.df = pickle.load(open(HERE / "filtered_train_data.pkl", "rb"))
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        template = random.choice(PROMPT_TEMPLATES)
        row = self.df.iloc[idx]
        plen = len(
            self.tokenizer.encode(row["patched_output"], add_special_tokens=False)
        )
        prompt = template.replace("[LAYER]", str(row["segment"]))
        prompt = prompt.replace("[Input Text]", row["input_text"])
        prompt = prompt.replace("[TOKENS]", row["subject"])
        prompt = prompt.replace("[PATCHED_OUTPUT]", row["patched_output"])
        return prompt, row["patching_vector"], plen

    def __len__(self):
        return len(self.df)


def collate_fn(model, tokenizer, batch):
    prompt, patching_vector, plen = zip(*batch)
    patching_vector = torch.stack([torch.tensor(pv) for pv in patching_vector], dim=0)
    toks = tokenizer(
        prompt,
        padding=True,
        max_length=128,  # should be enough for most prompts
        truncation=True,
        return_tensors="pt",
    )
    plen = torch.tensor(plen, dtype=torch.long)

    return {
        "input_ids": toks.input_ids,
        "patching_vector": patching_vector,
        "attention_mask": toks.attention_mask,
        "labels": toks.input_ids,
        "patched_output_pos": toks.input_ids.shape[1]
        - plen,  # position of [PATCHED_OUTPUT] token
    }
