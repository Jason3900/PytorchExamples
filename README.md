# Pytorch Examples

## Introduction
We provide a simple pytorch-based bert classification example with well-formed structure. You can simply build and run your fine-tuning task with tiny modification of the code.

## Requirements
```bash
conda create -n torch_env python=3.9 pandas tqdm scikit-learn -y
conda activate torch_env
conda install pytorch cudatoolkit=11.3.1 -y
pip install transformers wandb
```

## Train
1. Download transformers' pretrained model files (pytorch_model.bin, config.json, vocab.txt ...) and put them in one dir, eg. pretrained
2. Customize a dataset in src/datasets.py. We provide IMDB and SNLI dataset as demos. Basically, for sent /sent-pair classification task, the only thing you need to do is to inherit SeqCLSDataset class and implement read_line / read_example according to your data format.
3. Create labelspace file containing all labels, sep by line break
4. Edit scripts/train.sh
5. (optional) --use_wandb and set wandb_key to enable logging with wandb.ai
6. Run it!

```bash
bash scripts/train.sh
```
