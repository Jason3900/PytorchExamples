# -*- coding:UTF-8 -*-
from argparse import ArgumentParser


def get_args(mode="train"):
    parser = ArgumentParser()
    assert mode in {"train", "inference"}
    get_common_args(parser)
    if mode == "train":
        get_train_args(parser)
    elif mode == "inference":
        get_inference_args(parser)
    else:
        raise NotImplementedError
    args = parser.parse_args()
    return args


def get_common_args(parser):
    parser.add_argument("--task-name", type=str, help="task name for logging")
    parser.add_argument("--timestamp", type=str)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--label-path", type=str, required=True, help="labelspace, each line contains one unique label")
    parser.add_argument("--pretrained", type=str, required=True, help="pretrained models on huggginface transformers' model hub")
    parser.add_argument("--pooler-type", type=str, required=True, help="choose mean pooler or original cls pooler")
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--max-seq-len", type=int, default=128)
    parser.add_argument("--pre-tokenized", action="store_true", help="if true, input text will be splitted by space first, then tokenize(slower speed)")
    parser.add_argument("--save-cache", action="store_true", help="save/overwrite data cache")
    parser.add_argument("--use-cache", action="store_true", help="load cache (.pkl)")
    parser.add_argument("--log-interval", default=10, type=int)

def get_train_args(parser):
    parser.add_argument("--horovod", action="store_true", help="use horovod for distributed training")
    parser.add_argument("--train-path", type=str, required=True)
    parser.add_argument("--valid-path", type=str)
    parser.add_argument("--ckpt-path", type=str, required=True, help="dir to save ckpts")
    parser.add_argument("--max-epochs", type=int, required=True, help="max epoches to train")
    parser.add_argument("--patience", type=int, required=True, help="if no improve counts exceeds patience, trigger early stop")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--no-scale-lr", action="store_true", help="don't scale up lr by world_size")
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--shuffle", action="store_true", help="shuffle train data")
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--wandb-key", type=str)
    

def get_inference_args(parser):
    pass
