"""Generate feature descriptions data.

- To generate the training data, we use Llama-Scope to obtain residual stream
    SAE features for **Llama-3.1-8B**. We use these for the direction {v}. 
- Ground-truth explanations for each feature are given by Neuropedia.
- Training data comes in the pairs:
    Input: ( {v} , {l} ) => Output: ({e})

Requires 
```bash
pip install awscli
```
"""
import re
import os
import gzip
import json
import tqdm
import subprocess
from omegaconf import OmegaConf
import torch
import pandas as pd
from pathlib import Path
from sae_lens import SAE
from torch.utils.data import random_split
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Any, Dict, Iterator, List, Optional, Tuple
from peft import get_peft_model, TaskType, LoraConfig

from .dataset import FeatureDescriptionDataset


HERE = Path(__file__).resolve().parent
ARTIFACTS_DIR = HERE / "artifacts"


def download_data(model: str = "llama3.1-8b", sae: str = "llamascope-res-32k"):
    command = (
        f"aws s3 sync s3://neuronpedia-datasets/v1/{model} {ARTIFACTS_DIR} "
        f"--no-sign-request "
        f"--exclude '*' "
        f"--include '*-{sae}/*explanations*' "
        f"--include '*-{sae}/*model.jsonl' "
        f"--include '*-{sae}/*release.jsonl' "
        f"--include '*-{sae}/*source.jsonl' "
        f"--include '*-{sae}/*sourceset.jsonl'"
    )
    print(command)
    subprocess.run(command, shell=True)


def iter_jsonl(path: Path, *, limit: Optional[int] = None) -> Iterator[Dict[str, Any]]:
    """Stream records from a `.jsonl` or `.jsonl.gz` file."""
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                return
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def get_layer_explanations(
    *, model: str = "llamascope-res-32k", layer: int = 0
) -> List[Dict[str, Any]]:
    model_dir = ARTIFACTS_DIR / f"{layer}-{model}"
    explanations_path = model_dir / "explanations"
    exps = []
    for fname in tqdm.tqdm(
        os.listdir(explanations_path), desc=f"Layer {layer} explanations"
    ):
        if fname.endswith(".jsonl.gz"):
            exps.extend(
                [
                    # remove quotes/whitespace from the description
                    (
                        layer,
                        re.sub(
                            r'^\s*["\']?(.*?)["\']?\s*$',
                            r"\1",
                            v["description"].strip(),
                        ),
                        v["index"],
                    )
                    for v in list(iter_jsonl(explanations_path / fname))
                ]
            )
    return exps


def load_source_metadata(
    *, model: str = "llamascope-res-32k", layer: int = 0
) -> Dict[str, Any]:
    """Load the per-layer source metadata
    - `saelensRelease`: the SAE-Lens release name (e.g. "llama_scope_lxr_32x")
    - `saelensSaeId`: the SAE id within that release (e.g. "l0r_32x")
    """
    model_dir = ARTIFACTS_DIR / f"{layer}-{model}"
    source_path = model_dir / "source.jsonl"
    return next(iter_jsonl(source_path, limit=1))


def get_sae_release_and_id(
    *, model: str = "llamascope-res-32k", layer: int = 0
) -> Tuple[str, str]:
    meta = load_source_metadata(model=model, layer=layer)
    release = meta.get("saelensRelease")
    sae_id = meta.get("saelensSaeId")
    return release, sae_id


def load_sae_encoder_matrix(
    *,
    model: str = "llamascope-res-32k",
    layer: int = 0,
    device: str = "cpu",
) -> "Any":
    """Return the encoder matrix for a given layer (residual stream). This is
    because we are answering questions like "what does [v] mean at layer [l]?"
    """
    release, sae_id = get_sae_release_and_id(model=model, layer=layer)
    sae = SAE.from_pretrained(release, sae_id, device=device)
    return sae.W_enc


def load(cfg):
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
    tokenizer.add_special_tokens(
        {"additional_special_tokens": ["[FEATURE]", "[LAYER]"]}
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.resize_token_embeddings(len(tokenizer))
    dataset = FeatureDescriptionDataset(cfg, tokenizer)
    train, val, test = random_split(
        dataset, [1 - cfg.val_split - cfg.test_split, cfg.val_split, cfg.test_split]
    )
    return (model, tokenizer, train, val, test)


def build(cfg):
    target_model = cfg.target_model
    model = cfg.sae.model
    layers = cfg.sae.layers
    print(f"Building data for {target_model} with {model} and {layers} layers")
    # Load SAE encoder matrices for residual stream of each layer
    download_data(model=target_model, sae=model)
    for layer in tqdm.tqdm(range(layers), desc="Building data"):
        W_enc = load_sae_encoder_matrix(model=model, layer=layer, device="cpu")
        torch.save(W_enc, ARTIFACTS_DIR / f"layer_{layer}_encoder.pth")
    # Load explanations for each layer
    all_exps = []
    for layer in tqdm.tqdm(range(layers), desc="Building data"):
        exps = get_layer_explanations(model=model, layer=layer)
        all_exps.extend(exps)
    df = pd.DataFrame(all_exps, columns=["layer", "description", "index"])
    df.to_csv(ARTIFACTS_DIR / "layer_explanations.csv", index=False)
