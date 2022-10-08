# -*- coding:UTF-8 -*-
from src.dataloaders import init_dataloader
from src.datasets import SNLIDataset, IMDBDataset, LabelMap
from src.model import BertCLSModel
from src.utils import init_distributed, pin_gpu_to_process, wrap_hvd_optimizer, broadcast_model_params
import horovod.torch as hvd
import torch
from transformers import AutoTokenizer
from transformers.optimization import get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from datetime import datetime
import os
import random
import wandb
import logging


class Trainer:
    def __init__(self, args):
        self.timestamp = args.timestamp
        self.fix_seed(args.seed)
        self.device = None
        self.is_distributed = False
        self.setup_device(use_horovod=args.horovod)
        self.label_map = LabelMap(args.label_path)
        self.tokenizer = AutoTokenizer.from_pretrained(args.pretrained)
        self.model = BertCLSModel(pretrained_path=args.pretrained,
                                  num_labels=self.label_map.num_labels,
                                  dropout_prob=args.dropout,
                                  pooler_type=args.pooler_type)

        if torch.cuda.is_available():
            self.model.cuda()
        self.batch_size = args.batch_size
        self.shuffle_train_data = args.shuffle
        self.max_seq_len = args.max_seq_len
        self.train_loader, self.valid_loader = self.prepare_data(train_path=args.train_path,
                                                                 valid_path=args.valid_path,
                                                                 pre_tokenized=args.pre_tokenized,
                                                                 save_cache=args.save_cache,
                                                                 use_cache=args.use_cache)

        self.ckpt_path = os.path.join(args.ckpt_path, self.timestamp)
        if not os.path.exists(self.ckpt_path):
            os.makedirs(self.ckpt_path)
        self.max_epochs = args.max_epochs
        self.patience = args.patience
        self.log_interval = args.log_interval
        self.lr = args.lr
        self.scale_lr = not args.no_scale_lr
        self.weight_decay = args.weight_decay
        self.optimizer = self.init_optimizer()
        self.lr_scheduler = self.init_scheduler()
        self.best_metric = 0.0
        self.best_epoch = 0
        self.use_wandb = args.use_wandb
        if self.use_wandb:
            os.environ["WANDB_API_KEY"] = args.wandb_key
            wandb.init(project=args.task_name, group="torch")

    def setup_device(self, use_horovod: bool):
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        if torch.cuda.is_available() and use_horovod:
            if torch.cuda.is_available() and use_horovod:
                self.is_distributed = True
            logging.info("use horovod for distributed training")
            init_distributed()
            pin_gpu_to_process()
        else:
            self.is_distributed = False

    def fix_seed(self, seed):
        torch.manual_seed(seed)
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        random.seed(seed)

    def prepare_data(self, train_path: str, valid_path: str, pre_tokenized: bool, save_cache: bool, use_cache: bool):
        train_set = IMDBDataset(data_type="train", path=train_path, label_map=self.label_map,
                                tokenizer=self.tokenizer, pre_tokenize=pre_tokenized,
                                max_seq_len=self.max_seq_len, save_cache=save_cache, use_cache=use_cache)

        train_loader = init_dataloader(dataset=train_set,
                                       shuffle=True,
                                       batch_size=self.batch_size,
                                       input_pad_id=self.tokenizer.pad_token_id,
                                       is_distributed=self.is_distributed
                                       )
        valid_loader = None
        if valid_path is not None:
            valid_set = IMDBDataset(data_type="valid", path=valid_path, label_map=self.label_map,
                                    tokenizer=self.tokenizer, pre_tokenize=pre_tokenized,
                                    max_seq_len=self.max_seq_len, save_cache=save_cache, use_cache=use_cache)

            valid_loader = init_dataloader(dataset=valid_set,
                                           batch_size=self.batch_size,
                                           input_pad_id=self.tokenizer.pad_token_id,
                                           shuffle=False)
        return train_loader, valid_loader

    def init_optimizer(self):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        # effective batch size is scaled by the num of hvd process
        if self.is_distributed and self.scale_lr:
            world_size = hvd.size()
            self.lr *= world_size
            logging.info(f"scale lr to {self.lr} with hvd_size: {world_size}")
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.lr)
        if self.is_distributed:
            optimizer = wrap_hvd_optimizer(optimizer, self.model.named_parameters())
            logging.info("wrap optimizer with horovod")
        return optimizer

    def init_scheduler(self):
        total_train_steps = len(self.train_loader) * self.max_epochs
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer=self.optimizer,
            num_warmup_steps=total_train_steps * 0.1,
            num_training_steps=total_train_steps,
        )
        logging.info("setup lr_scheduler")
        return lr_scheduler

    def eval(self):
        total_preds = []
        total_golds = []
        total_loss = 0
        with torch.no_grad():
            for batch in self.valid_loader:
                if torch.cuda.is_available():
                    self.batch2cuda(batch, ignore_keys=["uuid"])
                loss, logits = self.model(batch)
                batch_golds = batch["label"].cpu().tolist()
                batch_preds = torch.argmax(logits.detach().cpu(), dim=-1).tolist()
                total_golds.extend(batch_golds)
                total_preds.extend(batch_preds)
                total_loss += loss
        acc = accuracy_score(y_true=total_golds, y_pred=total_preds)
        epoch_loss = total_loss / len(self.valid_loader)
        return acc, epoch_loss

    @staticmethod
    def batch2cuda(batch, ignore_keys=["uuid"]):
        for k, v in batch.items():
            if k in ignore_keys:
                continue
            batch[k] = v.cuda()

    def save_ckpt(self):
        torch.save(self.model.state_dict(), os.path.join(self.ckpt_path, "best.ckpt"))
        logging.info(f"successfully saved ckpt to {self.ckpt_path}")

    def train(self):
        if self.is_distributed:
            broadcast_model_params(model=self.model, optimizer=self.optimizer)
            logging.info("broadcast model params")
        current_step = 0
        no_improve = 0
        for epoch in range(self.max_epochs):
            num_steps_one_epoch = len(self.train_loader)
            pbar = tqdm(total=num_steps_one_epoch)
            self.model.train()
            for batch in self.train_loader:
                if torch.cuda.is_available():
                    self.batch2cuda(batch, ignore_keys=["uuid"])
                self.optimizer.zero_grad()
                loss, logits = self.model(batch)
                loss.backward()
                self.optimizer.step()
                self.lr_scheduler.step()
                current_step += 1
                loss = loss.detach().item()
                current_lr = self.lr_scheduler.get_last_lr()[0]
                if current_step % self.log_interval == 0 or current_step % num_steps_one_epoch == 0:
                    if self.use_wandb:
                        wandb.log({"loss": loss, "lr": current_lr})
                    logging.info(f"step {current_step}, current loss: {loss}, current_lr: {current_lr}")
                    if self.is_distributed:
                        if hvd.rank() == 0:
                            pbar.update(self.log_interval)
                    else:
                        pbar.update(self.log_interval)
                
            pbar.close()

            self.model.eval()
            valid_acc, valid_loss = self.eval()
            if self.use_wandb:
                wandb.log({"valid_loss": loss, "valid_acc": valid_acc})
            if self.best_metric < valid_acc:
                self.best_metric = valid_acc

                # save ckpt on rank 0 only
                if self.is_distributed and hvd.rank() != 0:
                    continue
                else:
                    self.save_ckpt()

                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= self.patience:
                    logging.info(f"no improve within {no_improve} epochs, early stop")
                    break

            logging.info(f"epoch {epoch} finished. valid acc: {valid_acc}, valid loss: {valid_loss}")
            logging.info(f"current best acc {self.best_metric}, in epoch {self.best_epoch}")
