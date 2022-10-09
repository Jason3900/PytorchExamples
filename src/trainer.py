# -*- coding:UTF-8 -*-
from src.dataloaders import init_dataloader
from src.datasets import SNLIDataset, IMDBDataset, LabelMap
from src.model import BertCLSModel
from src.utils import \
    init_distributed, pin_gpu_to_process, wrap_hvd_optimizer, get_world_size, \
    broadcast_model_params, setup_gradient_scaler, metric_average, LogMessage
import horovod.torch as hvd
import torch
from transformers import AutoTokenizer
from transformers.optimization import get_linear_schedule_with_warmup
from filelock import FileLock
from tqdm import tqdm
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
        self.use_amp = args.amp
        self.use_adasum = args.adasum
        assert not (self.use_amp and self.use_adasum), "amp and adasum cannot be used together"
        self.setup_device(use_horovod=args.horovod)  # also setup is_distributed and device attr
        self.log_info_message = LogMessage(self.is_distributed)
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
                                                                 num_workers=args.num_data_workers,
                                                                 save_cache=args.save_cache,
                                                                 use_cache=args.use_cache)

        self.ckpt_path = os.path.join(args.ckpt_path, self.timestamp)
        with FileLock(os.path.expanduser("~/.horovod_lock")):
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

    def prepare_data(self,
                     train_path: str,
                     valid_path: str,
                     pre_tokenized: bool,
                     num_workers: int,
                     save_cache: bool,
                     use_cache: bool):
        train_set = IMDBDataset(data_type="train", path=train_path, label_map=self.label_map,
                                tokenizer=self.tokenizer, pre_tokenize=pre_tokenized,
                                max_seq_len=self.max_seq_len, save_cache=save_cache, use_cache=use_cache)

        train_loader = init_dataloader(dataset=train_set,
                                       shuffle=True,
                                       batch_size=self.batch_size,
                                       input_pad_id=self.tokenizer.pad_token_id,
                                       num_workers=num_workers,
                                       is_distributed=self.is_distributed
                                       )
        valid_loader = None
        if valid_path is not None:
            valid_set = IMDBDataset(data_type="valid", path=valid_path, label_map=self.label_map,
                                    tokenizer=self.tokenizer, pre_tokenize=pre_tokenized,
                                    max_seq_len=self.max_seq_len, save_cache=save_cache, use_cache=use_cache)

            valid_loader = init_dataloader(dataset=valid_set,
                                           shuffle=False,
                                           batch_size=self.batch_size,
                                           input_pad_id=self.tokenizer.pad_token_id,
                                           num_workers=0,  # disable multiprocessing on valid set data loading
                                           is_distributed=self.is_distributed
                                           )
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
        if self.is_distributed:
            # If using GPU Adasum allreduce, scale learning rate by local_size.
            if self.use_adasum:
                scale_factor = hvd.local_size()
            elif self.scale_lr:
                scale_factor = get_world_size()
            else:
                scale_factor = 1

            self.lr *= scale_factor
            self.log_info_message(f"scale lr to {self.lr} with scale factor: {scale_factor}")
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.lr)
        if self.is_distributed:
            optimizer = wrap_hvd_optimizer(optimizer, self.model.named_parameters(), self.use_adasum)
            self.log_info_message("wrap optimizer with horovod")
        return optimizer

    def init_scheduler(self):
        total_train_steps = len(self.train_loader) * self.max_epochs
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer=self.optimizer,
            num_warmup_steps=total_train_steps * 0.1,
            num_training_steps=total_train_steps,
        )
        self.log_info_message("setup lr_scheduler")
        return lr_scheduler

    def eval(self):
        total_preds = []
        total_golds = []
        acc = 0.
        total_loss = 0.
        with torch.no_grad():
            for batch in self.valid_loader:
                if torch.cuda.is_available():
                    self.batch2cuda(batch, ignore_keys=["uuid"])
                loss, logits = self.model(batch)
                batch_golds = batch["label"].cpu()
                batch_preds = torch.argmax(logits.cpu(), dim=-1)
                acc += batch_preds.eq(batch_golds).float().sum()
                total_golds.extend(batch_golds.tolist())
                total_preds.extend(batch_preds.tolist())
                total_loss += loss.item()
        # use sampler to determine the number of examples in this worker's partition.
        acc /= len(self.valid_loader.sampler)
        total_loss /= len(self.valid_loader.sampler)
        # use allreduce to get average metrics
        if self.is_distributed:
            total_loss = metric_average(total_loss, 'avg_loss')
            acc = metric_average(acc, 'avg_accuracy')
        return acc, total_loss

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
            logging.info(f"broadcast model params to rank {hvd.rank()}")
        if self.use_amp:
            grad_scaler = setup_gradient_scaler()

        current_step = 0
        no_improve = 0
        for epoch in range(self.max_epochs):
            num_steps_one_epoch = len(self.train_loader)
            pbar = tqdm(total=num_steps_one_epoch)
            # In distributed mode, calling the set_epoch() method at the beginning of each epoch 
            # before creating the DataLoader iterator is necessary
            # to make shuffling work properly across multiple epochs.
            # Otherwise, the same ordering will be always used.
            if self.is_distributed:
                self.train_loader.sampler.set_epoch(epoch)
            self.model.train()
            for batch in self.train_loader:
                if torch.cuda.is_available():
                    self.batch2cuda(batch, ignore_keys=["uuid"])
                self.optimizer.zero_grad()
                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        loss, logits = self.model(batch)
                else:
                    loss, logits = self.model(batch)
                # Each parameter’s gradient (.grad attribute) should be unscaled
                # before the optimizer updates the parameters,
                # so the scale factor does not interfere with the learning rate.
                if self.use_amp:
                    grad_scaler.scale(loss).backward()
                    if self.is_distributed:
                        self.optimizer.synchronize()
                    # In-place unscaling of all gradients before weights update
                    grad_scaler.unscale_(self.optimizer)
                    if self.is_distributed:
                        with self.optimizer.skip_synchronize():
                            grad_scaler.step(self.optimizer)
                    else:
                        grad_scaler.step(self.optimizer)
                    # Update scaler in case of overflow/underflow
                    grad_scaler.update()
                else:
                    loss.backward()
                    self.optimizer.step()
                self.lr_scheduler.step()
                current_step += 1
                loss = loss.detach().item()
                current_lr = self.lr_scheduler.get_last_lr()[0]
                if current_step % self.log_interval == 0 or current_step % num_steps_one_epoch == 0:

                    update_steps = self.log_interval
                    if current_step % num_steps_one_epoch == 0:
                        update_steps = current_step % self.log_interval
                        if update_steps == 0:
                            update_steps = self.log_interval

                    if self.use_wandb:
                        wandb.log({"loss": loss, "lr": current_lr})

                    if (self.is_distributed and hvd.rank() == 0) or not self.is_distributed:
                        logging.info(f"step {current_step}, current loss: {loss}, current_lr: {current_lr}")
                        pbar.update(update_steps)

            pbar.close()
            self.log_info_message(f"epoch {epoch} training finished.")
            self.model.eval()
            valid_acc, valid_loss = self.eval()
            if self.use_wandb:
                wandb.log({"valid_loss": loss, "valid_acc": valid_acc})
            if self.best_metric < valid_acc:
                self.best_metric = valid_acc
                self.best_epoch = epoch

                # save ckpt on rank 0 only
                if self.is_distributed and hvd.rank() != 0:
                    continue
                else:
                    self.save_ckpt()

                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= self.patience:

                    self.log_info_message(f"no improve within {no_improve} epochs, early stop")
                    break

            self.log_info_message(f"epoch {epoch} finished. \
                            valid acc: {valid_acc}, valid loss: {valid_loss}")
            self.log_info_message(f"current best acc {self.best_metric}, in epoch {self.best_epoch}")
