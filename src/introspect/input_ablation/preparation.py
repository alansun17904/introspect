import re
import tqdm
import pandas as pd
from pathlib import Path
from transformers import pipeline
from functools import partial
from torch.utils.data import DataLoader

from .dataset import MMLUDataset, mmlu_collate_fn

HERE = Path(__file__).resolve().parent / "artifacts"
HERE.mkdir(parents=True, exist_ok=True)


def generate_outputs_df(model, tokenizer, max_tokens=4, batch_size=16, max_samples=None):
    dataset = MMLUDataset(show_hints=False)
    dataset_hints = MMLUDataset(show_hints=True)
    dl = DataLoader(dataset, batch_size=batch_size, collate_fn=partial(mmlu_collate_fn, tokenizer=tokenizer))
    dl_hints = DataLoader(dataset_hints, batch_size=batch_size, collate_fn=partial(mmlu_collate_fn, tokenizer=tokenizer))

    gen = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=max_tokens, do_sample=False, return_full_text=False)

    prompts, outputs, outputs_hint = [], [], []
    total_samples = 0
    
    for batch_no_hint, batch_with_hint in tqdm.tqdm(zip(dl, dl_hints), desc="Generating outputs"):
        prompts.extend(batch_with_hint)

        results_no_hint = gen(batch_no_hint)
        results_with_hint = gen(batch_with_hint)

        outputs.extend([result[0]["generated_text"] for result in results_no_hint])
        outputs_hint.extend([result[0]["generated_text"] for result in results_with_hint])
        
        total_samples += len(batch_no_hint)
        
        if max_samples is not None and total_samples >= max_samples:
            break
    
    # Create DataFrame
    df = pd.DataFrame({
        'prompt': prompts,
        'output': outputs,
        'output_hint': outputs_hint
    })
    
    return df


def filter_outputs(df):
    def extract_answer(out):
        match = re.compile(r"^(?:Answer:\s*)?(.+?)\s*$").match(out)
        if match:
            return match.group(1)
        return None
    df["output"] = df["output"].apply(extract_answer)
    df["output_hint"] = df["output_hint"].apply(extract_answer)
    # drop rows where the output doesn't follow format 
    df = df[df["output"].notna()]
    df = df[df["output_hint"].notna()]
    return df