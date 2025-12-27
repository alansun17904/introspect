# Replication of 'Training Language Models to Explain Their Own Computations'

This is my "minimal replication" of [Li et al. 2025](https://arxiv.org/pdf/2511.08579). My training runs and their metadata can be found on [WandB](https://wandb.ai/asun17904/introspect?nw=nwuserasun17904). The trained models for each experiment are uploaded on HuggingFace:
- [gemma-feature-desc](https://huggingface.co/asun17904/gemma-feature-desc)
- [gemma-activation-patch](https://huggingface.co/asun17904/gemma-activation-patch)
- [gemma-input-ablation](https://huggingface.co/asun17904/gemma-input-ablation)

these correspond to the feature discovery, activation patching, and input ablation experiments, respectively. Throughout, I use `gemma-2-2b` and `gemma-2-2b-it` for both the explained and explainer model (smaller than the models used in the paper, but works pretty well). 

My implementation is quite general and it can take apply to any model, you can change this simply by specifying a different model name in each of the configuration files. One except to this is the "input_ablation" experiments which require hyperspecific user-model-assistant roles. I think this can be trivially extended, but ran out of time to add this. 

## Installation
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Running Experiments
Every experiment has three actions: `build`, `train`, and `eval`. They need to be run in this order. You can call these 
```
introspect {feature, attrib, activation} action --config CONFIG 
```
The config file specifies experiment hyperparameters. The configs I used can be found in `configs/`. Each of my experiments use the same configuration file for all actions. 
