import tqdm
import torch
import random
import pandas as pd
from pathlib import Path
from functools import partial
import torch.utils.data as data


PROMPT_TEMPLATES = [
    "At layer [LAYER], [FEATURE] encodes ",
    "[FEATURE] activates at layer [LAYER] for ",
    "We can describe [FEATURE] at layer [LAYER] as encoding ",
    "[FEATURE] activates at layer [LAYER] for inputs with the following features: ",
]

HERE = Path(__file__).resolve().parent / "artifacts"


def collate_fn(model, tokenizer, batch):
    prompt, feature_v, plen, _ = zip(*batch)
    plen = torch.tensor(plen, dtype=torch.long)
    feature_v = torch.stack(list(feature_v), dim=0)
    toks = tokenizer(
        prompt,
        padding=True,
        max_length=50,  # by looking at data, most length <= 50
        truncation=True,
        return_tensors="pt",
    )

    return {
        "input_ids": toks.input_ids,
        "feature_v": feature_v,
        "feature_id": tokenizer.convert_tokens_to_ids("[FEATURE]"),
        "pad_token_id": tokenizer.convert_tokens_to_ids(tokenizer.pad_token),
        "attention_mask": toks.attention_mask,
        "labels": toks.input_ids,
        "desc_pos": toks.input_ids.shape[1] - plen,
    }


class FeatureDescriptionDataset(data.Dataset):
    def __init__(self, cfg, tokenizer):
        self.explanations = pd.read_csv(HERE / "layer_explanations.csv")
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.single_layer = cfg.single_layer
        if cfg.single_layer:
            print(f"Loading single layer {cfg.layer_id}")
            self.explanations = self.explanations[
                self.explanations["layer"] == cfg.layer_id
            ]
            self.encoder_mtx = torch.load(HERE / f"layer_{cfg.layer_id}_encoder.pth")
        else:
            self.encoder_mtx = [
                torch.load(HERE / f"layer_{layer}_encoder.pth")
                for layer in tqdm.tqdm(
                    range(cfg.sae.layers), desc="Loading encoder matrices"
                )
            ]

    def __len__(self):
        return len(self.explanations)

    def __getitem__(self, idx):
        layer, desc, index = self.explanations.iloc[idx]
        prompt = random.choice(PROMPT_TEMPLATES)
        plen = len(self.tokenizer.encode(desc, add_special_tokens=False))
        skeleton = prompt.replace("[LAYER]", str(layer))
        prompt = skeleton + desc
        if self.single_layer:
            feature_v = self.encoder_mtx[:, index]
        else:
            feature_v = self.encoder_mtx[layer][:, index]
        return (prompt, feature_v, plen, skeleton)
