# -*- coding:UTF-8 -*-
from src.dataloaders import init_dataloader
from src.datasets import SNLIDataset, IMDBDataset, LabelMap
from src.model import BertCLSModel
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
        timestamp = datetime.today().strftime('%Y-%m-%d-%H-%M')
        self.fix_seed(args.seed)
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
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
                                                                 pre_tokenized =args.pre_tokenized,
                                                                 save_cache=args.save_cache,
                                                                 use_cache=args.use_cache)
        
        self.ckpt_path = os.path.join(args.ckpt_path, timestamp)
        if not os.path.exists(self.ckpt_path):
            os.makedirs(self.ckpt_path)
        self.max_epochs = args.max_epochs
        self.patience = args.patience
        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.optimizer = self.init_optimzer()
        self.lr_scheduler = self.init_scheduler()
        self.best_metric = 0.0
        self.best_epoch = 0
        self.use_wandb = args.use_wandb
        if self.use_wandb:
            os.environ["WANDB_API_KEY"] = args.wandb_key
            wandb.init(project=args.task_name)

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
                                       batch_size=self.batch_size,
                                       input_pad_id = self.tokenizer.pad_token_id,
                                       shuffle=self.shuffle_train_data)
        valid_loader = None
        if valid_path is not None:
            valid_set = IMDBDataset(data_type="valid", path=valid_path, label_map=self.label_map, 
                        tokenizer=self.tokenizer, pre_tokenize=pre_tokenized,
                        max_seq_len=self.max_seq_len, save_cache=save_cache, use_cache=use_cache)

            valid_loader = init_dataloader(dataset=valid_set,
                                        batch_size=self.batch_size,
                                        input_pad_id = self.tokenizer.pad_token_id,
                                        shuffle=False)
        return train_loader, valid_loader

    def init_optimzer(self):
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
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.lr)
        return optimizer

    def init_scheduler(self):
        total_train_steps = len(self.train_loader) * self.max_epochs
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer=self.optimizer,
            num_warmup_steps=total_train_steps * 0.1,
            num_training_steps=total_train_steps,
        )
        return lr_scheduler

    def eval(self):
        total_preds = []
        total_golds = []
        total_loss = 0
        with torch.no_grad():
            for batch in tqdm(self.valid_loader):
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

    def train(self):

        current_step = 0
        no_improve = 0
        for epoch in range(self.max_epochs):
            self.model.train()
            for batch in tqdm(self.train_loader):
                if torch.cuda.is_available():
                    self.batch2cuda(batch, ignore_keys=["uuid"])
                loss, logits = self.model(batch)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.lr_scheduler.step()
                current_step += 1
                loss = loss.detach().item()
                current_lr = self.lr_scheduler.get_last_lr()[0]
                if current_step % 10 == 0:
                    if self.use_wandb:
                        wandb.log({"loss": loss, "lr": current_lr})
                    logging.info(f"step {current_step}, current loss: {loss}, current_lr: {current_lr}")

            self.model.eval()
            valid_acc, valid_loss = self.eval()
            if self.use_wandb:
                wandb.log({"valid_loss": loss, "valid_acc": valid_acc})
            if self.best_metric < valid_acc:
                self.best_metric = valid_acc
                logging.info(f"successfully saved ckpt to {self.ckpt_path}")
                torch.save(self.model.state_dict(), os.path.join(self.ckpt_path, "best.ckpt"))
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= self.patience:
                    logging.info(f"no improve within {no_improve} epochs, early stop")
                    break
            
            logging.info(f"epoch {epoch} finished. valid acc: {valid_acc}, valid loss: {valid_loss}")
            logging.info(f"current best acc {self.best_metric}, in epoch {self.best_epoch}")
        
        