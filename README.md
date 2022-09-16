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
2. Customize a dataset in `src/datasets.py`. We provide [IMDB](https://www.kaggle.com/datasets/atulanandjha/imdb-50k-movie-reviews-test-your-bert?select=train.csv) and [SNLI](https://nlp.stanford.edu/projects/snli/snli_1.0.zip) dataset as demos. Basically, for sent /sent-pair classification task, the only thing you need to do is to inherit `SeqCLSDataset` class and implement `read_line` / `read_example` according to your data format.
3. Create labelspace file containing all labels, sep by line break
4. Edit scripts/train.sh
5. (optional) --use_wandb and set wandb_key to enable logging with wandb.ai
6. Activate conda env and Run it! 

    ```bash
    bash scripts/train.sh
    ```

## Debug
[Fairseq](https://github.com/facebookresearch/fairseq) wraps a multiprocessing-supported pdb. You can use `from debugger.mp_pdb import pdb;pdb.set_trace` in our code to debug in real time. See common usage at https://docs.python.org/3/library/pdb.html
