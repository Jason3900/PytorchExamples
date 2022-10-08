# Pytorch Examples

## Introduction

We provide a simple pytorch-based bert classification example with well-formed structure. You can simply build and run your fine-tuning task with tiny modification of the code.

## Requirements

1. install Open MPI and NCCL to use horovod with GPU
   - follow the Open MPI installation guide [here](https://www.open-mpi.org/faq/?category=building#easy-build)
   - follow the NCCL2 installation guide [here](https://docs.nvidia.com/deeplearning/nccl/install-guide/index.html)
     - I personally recommend installing NCCL via tar file follow Sec. 3.3, as it's more flexible.

2. install basic dependencies
   ```bash
   conda create -n torch_env python=3.9 pandas tqdm scikit-learn numpy -y
   conda activate torch_env
   # install pytorch via pip instead of conda is a simpler way to use horovod. But you have to make sure g++-5 or above is installed.
   pip install torch --extra-index-url https://download.pytorch.org/whl/cu113
   pip install transformers wandb
   ```

3. install horovod with GPU support, see [horovod on GPU](https://horovod.readthedocs.io/en/stable/gpus_include.html) for more instructions.
   ```bash
   # specify nccl location if you build it with tar file
   HOROVOD_NCCL_HOME=/usr/local/nccl_2.9.9-1+cuda11.3_x86_64 \
   # build horovod with pytorch and nccl support
   HOROVOD_WITH_PYTORCH=1 HOROVOD_GPU_OPERATIONS=NCCL \
   pip install horovod[pytorch] --no-cache-dir 2>&1 | tee install_horovod.log
   ```

## Train

1. Download transformers' pretrained model files (pytorch_model.bin, config.json, vocab.txt ...) and put them in one dir, eg. pretrained
2. Customize a dataset in `src/datasets.py`. We provide [IMDB](https://www.kaggle.com/datasets/atulanandjha/imdb-50k-movie-reviews-test-your-bert?select=train.csv) and [SNLI](https://nlp.stanford.edu/projects/snli/snli_1.0.zip) dataset as demos. Basically, for sent /sent-pair classification task, the only thing you need to do is to inherit `SeqCLSDataset` class and implement `read_line` / `read_example` according to your data format.
3. Create labelspace file containing all labels, sep by line break
4. edit ./hostfile to specify which machines to be used for distributed training
5. Edit scripts/train.sh
6. (optional) --use-wandb and set wandb-key to enable logging with wandb.ai 
7. Activate conda env and Run it! 

    ```bash
    bash scripts/train_hvd.sh
    ```

## Debug
[Fairseq](https://github.com/facebookresearch/fairseq) wraps a multiprocessing-supported pdb. You can use `from debugger.mp_pdb import pdb;pdb.set_trace` in our code to debug in real time. See common usage at https://docs.python.org/3/library/pdb.html
