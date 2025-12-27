import os
import re
import json
import pickle
import tqdm
import torch
from functools import partial
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, Subset
from transformer_lens import HookedTransformer
from transformer_lens.utils import get_act_name
from transformers import AutoTokenizer, AutoModelForCausalLM
from omegaconf import OmegaConf
from peft import get_peft_model, TaskType, LoraConfig


HERE = Path(__file__).resolve().parent / "artifacts"


PROMPT_PREFIX_TEMPLATE = "Answer with only {} or {}: "


class ActivationPatchDataset(Dataset):
    def __init__(self):
        if not os.path.exists(HERE / "true_false_prompts.csv"):
            process_true_false()
        self.true_false_prompts = pd.read_csv(HERE / "true_false_prompts.csv")

    def __len__(self):
        return len(self.true_false_prompts)

    def __getitem__(self, idx):
        return self.true_false_prompts.iloc[idx]


def collate_fn(tokenizer, batch):
    """Assume that the tokenizer padding side is left"""
    templates, subjects, c_subjects, targets, c_targets = zip(*batch)
    templates = [
        PROMPT_PREFIX_TEMPLATE.format(tar, c_tar) + t
        for t, s, c_s, tar, c_tar in zip(
            templates, subjects, c_subjects, targets, c_targets
        )
    ]
    suff_char_index = [len(v.split("{}")[1]) for v in templates]
    clean_templates = list(map(lambda x: x[0].format(x[1]), zip(templates, subjects)))
    corr_templates = list(map(lambda x: x[0].format(x[1]), zip(templates, c_subjects)))
    all_templates = clean_templates + corr_templates
    toks = tokenizer(
        all_templates, padding=True, return_tensors="pt", return_offsets_mapping=True
    )
    template_start_idx = torch.tensor(
        [
            len(all_templates[i]) - suff_char_index[i % len(suff_char_index)]
            for i in range(len(all_templates))
        ]
    )
    template_start_idx = template_start_idx.unsqueeze(1)
    indices = torch.nonzero(
        (toks.offset_mapping[:, :, 0] <= template_start_idx)
        & (template_start_idx < toks.offset_mapping[:, :, 1])
    )[:, 1]
    return {
        "clean_input_text": all_templates[: len(clean_templates)],
        "clean_subject": subjects[: len(clean_templates)],
        "clean_input_ids": toks.input_ids[: len(clean_templates)],
        "clean_input_mask": toks.attention_mask[: len(clean_templates)],
        "clean_last_subject_idx": indices[: len(clean_templates)],
        "corr_input_ids": toks.input_ids[len(clean_templates) :],
        "corr_input_mask": toks.attention_mask[len(clean_templates) :],
        "corr_last_subject_idx": indices[len(clean_templates) :],
        "clean_targets": targets,
        "corr_targets": c_targets,
    }


def process_true_false() -> pd.DataFrame:
    def first_template_match(template, prompts):
        """Find the first counterfactual prompt that matches the template
        of the 'factual' prompt template"""
        rgx = template.format("(.*)")
        for p in prompts:
            if ptrn := re.fullmatch(rgx, p):
                return p, ptrn.group(1)
        return None, None

    counterfact = json.load(open(HERE / "counterfact.json"))
    true_false_prompts = []
    for cf in counterfact:
        row = dict()
        cf_rr = cf["requested_rewrite"]
        row["prompt"] = cf_rr["prompt"]
        row["subject"] = cf_rr["subject"]
        _, row["counter_subject"] = first_template_match(
            row["prompt"], cf["attribute_prompts"]
        )
        if row["counter_subject"] is None:
            continue
        row["target"] = cf_rr["target_true"]["str"]
        row["counter_target"] = cf_rr["target_new"]["str"]
        true_false_prompts.append(row)
    return pd.DataFrame(true_false_prompts).to_csv(
        HERE / "true_false_prompts.csv", index=False
    )


def patch_rs(act, hook, tok_idx, new_tok_val):
    if act.shape[1] <= 1:
        return act  # bootleg way to only patch KV-cache during generation
    act[torch.arange(act.shape[0], device=act.device), tok_idx, :] = new_tok_val
    return act


def generate(model: HookedTransformer, input_ids, **kwargs) -> str:
    nt = model.generate(
        input_ids.to(model.cfg.device), max_new_tokens=4, padding_side="left", **kwargs
    )
    return nt


@torch.inference_mode
def make_train_data(model, tokenizer, cfg):
    """Assumes that cfg has attribute `layer_segments`: list of layer indices (non-inclusive) in
    increasing order. We patch all of the layers in each segment.
    """
    ap = ActivationPatchDataset()
    dl = DataLoader(
        ap,
        batch_size=cfg.make_train_data.batch_size,
        collate_fn=partial(collate_fn, tokenizer),
    )
    train_data, segments = [], [
        (([0] + cfg.layer_segments)[i], cfg.layer_segments[i])
        for i in range(len(cfg.layer_segments))
    ]
    for batch in tqdm.tqdm(dl, desc="Making train data"):
        for layer_segment in segments:
            row = dict()
            row["clean_output"] = tokenizer.batch_decode(
                generate(model, batch["clean_input_ids"], verbose=False),
                eos_token_id=tokenizer.eos_token_id,
                skip_special_tokens=True,
            )
            # Get patching vector across layer segments
            row["segment"] = layer_segment[1]
            row["input_text"] = batch["clean_input_text"]
            row["subject"] = batch["clean_subject"]
            row["targets"] = batch["clean_targets"]
            row["counter_targets"] = batch["corr_targets"]
            _, ac = model.run_with_cache(batch["corr_input_ids"])
            resid = ac.accumulated_resid()
            i = (
                batch["corr_last_subject_idx"]
                .view(1, -1, 1, 1)
                .expand(resid.shape[0], -1, 1, resid.shape[-1])
            )
            # Average patching vector across layer segments
            v = torch.mean(
                resid.gather(dim=2, index=i.to(resid.device)).squeeze(2)[
                    layer_segment[0] : layer_segment[1]
                ],
                dim=0,
            )
            row["patching_vector"] = v.detach().cpu().numpy()
            # Get target output
            with model.hooks(
                fwd_hooks=[
                    (
                        get_act_name("resid_post", layer),
                        partial(
                            patch_rs,
                            tok_idx=batch["clean_last_subject_idx"],
                            new_tok_val=v,
                        ),
                    )
                    for layer in range(layer_segment[0], layer_segment[1])
                ]
            ):
                row["patched_output"] = tokenizer.batch_decode(
                    generate(model, batch["clean_input_ids"], verbose=False),
                    eos_token_id=tokenizer.eos_token_id,
                    skip_special_tokens=True,
                )
            train_data.append(row)
    # store the training data in a pickle file
    with open(HERE / "train_data.pkl", "wb") as f:
        pickle.dump(train_data, f)


def filter_train_data(cfg):
    """Remove instructions from each prompt to reduce no. tokens during training;
    remove completions that do not contain the right answer; crop completions to
    include only the right answer
    """
    train_data = pickle.load(open(HERE / "train_data.pkl", "rb"))

    # Flatten batch data into individual rows
    flattened_rows = []
    for row in train_data:
        batch_size = len(row["input_text"])
        # patching_vector has shape (batch_size, feature_dim)
        patching_vectors = row["patching_vector"]

        for i in range(batch_size):
            flattened_row = {
                "segment": row["segment"],
                "input_text": row["input_text"][i],
                "subject": row["subject"][i],
                "target": row["targets"][i],
                "counter_target": row["counter_targets"][i],
                "clean_output": row["clean_output"][i],
                "patched_output": row["patched_output"][i],
                "patching_vector": patching_vectors[
                    i
                ],  # Extract single vector from batch
            }
            flattened_rows.append(flattened_row)

    # Convert to DataFrame
    df = pd.DataFrame(flattened_rows)

    def remove_instructions(text):
        # Remove the first line of the text (instructions)
        return "".join(text.split(":")[1:])

    # Filter to only include responses where the model correctly does the completion
    # Check if target_output contains the target value
    def is_correct_completion(target_output, actual_output):
        # Check if target appears in the output (case-insensitive, as a word boundary)
        actual_lower = actual_output.lower().strip()
        target_lower = target_output.lower().strip()
        # Check if target is in output (as whole word or substring)
        return target_lower in actual_lower

    df["is_correct"] = df.apply(
        lambda row: is_correct_completion(
            row["input_text"] + " " + row["target"], row["clean_output"]
        ),
        axis=1,
    )

    df["input_text"] = df["input_text"].apply(remove_instructions)
    df["patched_output"] = df["patched_output"].apply(remove_instructions)
    df["clean_output"] = df["clean_output"].apply(remove_instructions)

    df = df[df["is_correct"]].copy()
    df = df.drop(columns=["is_correct"])  # only keep correct completions

    pickle.dump(df, open(HERE / "filtered_train_data.pkl", "wb"))
    return df


def load(cfg):
    """Load model, tokenizer, and datasets for training."""
    from .dataset import ActivationPatchDataset

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
        {"additional_special_tokens": ["[FEATURE]", "[LAYER]", "[PATCHED_OUTPUT]"]}
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.resize_token_embeddings(len(tokenizer))
    dataset = ActivationPatchDataset(cfg, tokenizer)

    # Do not randomly shuffle, since this results in leakage across layers
    train_size = int(len(dataset) * (1 - cfg.val_split - cfg.test_split))
    val_size = int(len(dataset) * cfg.val_split)
    train, val, test = (
        Subset(dataset, range(train_size)),
        Subset(dataset, range(train_size, train_size + val_size)),
        Subset(dataset, range(train_size + val_size, len(dataset))),
    )

    return (model, tokenizer, train, val, test)
