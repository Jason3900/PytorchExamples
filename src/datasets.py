# -*- coding:UTF-8 -*-
from collections import OrderedDict
from torch.utils.data import Dataset
import pickle as pkl
import os
import json
import logging
import random
import numpy as np
import pyarrow as pa
import pandas as pd
from tqdm import tqdm

def print_examples(data, k=3):
    for example in random.sample(data, k=k):
        logging.info(example)

def process_pre_tokenized(text: list, tokenizer):
    tokens = []
    for word in text:
        tokens.extend(tokenizer.tokenize(word))
    return tokens

class LabelMap:
    def __init__(self, label_path):
        self.id2label = self.build_id2label(label_path)
        self.label2id = OrderedDict([(label,idx) for idx, label in enumerate(self.id2label)])
        self.num_labels = len(self.id2label)

    @staticmethod
    def build_id2label(path):
        labels = []
        with open(path, "r", encoding="utf8") as fr:
            for line in fr:
                line = line.strip()
                if not line:
                    continue
                labels.append(line)
        return labels
            

class SeqCLSDataset(Dataset):
    def __init__(self, data_type: str, path: str, label_map, tokenizer, pre_tokenize, max_seq_len, save_cache=True, use_cache=True):
        super().__init__()
        self.data_type = data_type
        self.pre_tokenize = pre_tokenize
        self.max_seq_len = max_seq_len
        self.use_cache = use_cache
        self.save_cache = save_cache
        self.num_samples = 0
        self.label_map = label_map
        self.tokenizer = tokenizer
        self.data = self.gather_data(path)
        print_examples(self.data, k=5)

    @staticmethod
    def read_line(path):
        with open(path, "r", encoding="utf8") as fr:
            for idx, line in enumerate(fr):
                line = line.strip()
                if not line:
                    continue
                yield idx, line

    @staticmethod
    def read_sample(line):
        """
        implement it by yourself
        return uuid: str, text_a: str, text_b: str, label: int
        """
        raise NotImplementedError

    @staticmethod
    def py2zero_copy_objs(data_i):
        for k, v in data_i.items():
            if k == "uuid":
                value = pa.scalar(v)
            elif k == "label":
                value = np.int32(v)
            else:
                value = np.array(v, dtype=np.int32)
            data_i[k] = value
        return data_i

    def gather_data(self, path):
        pkl_path = path+".pkl"
        if self.use_cache and os.path.exists(pkl_path):
            with open(pkl_path, "rb") as fr:
                data = pkl.load(fr)
            self.num_samples = len(data)
            logging.info(f"successfully load {pkl_path}")
            logging.info(f"num samples: {self.num_samples}")
        else:
            max_len_in_data = 0
            data = []
            for idx, line in self.read_line(path):
                try:
                    uuid, text_a, text_b, label = self.read_sample(line)
                    if not uuid:
                        uuid = f"{self.data_type}-{idx}"
                    inputs = self.build_inputs(text_a, text_b)
                    if max_len_in_data < len(inputs["input_ids"]):
                        max_len_in_data = len(inputs["input_ids"])
                    label_id = self.label_map.label2id[label]
                except KeyboardInterrupt:
                    raise KeyboardInterrupt
                except Exception as e:
                    logging.warning(f"{path}, {idx} process error, continue")
                    continue
                self.num_samples += 1
                data_i = {"uuid": uuid, **inputs, "label": label_id}
                # convert to numpy or pyarrow objects to avoid copy-on-write behavior
                # when use multiprocessing data loading strategy
                # see more details on https://github.com/pytorch/pytorch/issues/13246#issuecomment-905703662
                data_i = self.py2zero_copy_objs(data_i)
                data.append(data_i)
            logging.info(f"num samples: {self.num_samples}, max len in dataset: {max_len_in_data}")
        if self.save_cache:
            if not os.path.exists(pkl_path):
                with FileLock(os.path.expanduser("~/.horovod_lock")):
                    with open(pkl_path, "wb") as fw:
                        fw.write(pkl.dumps(data))
            else:
                logging.info("try to save dataset but pkl exists! ignore...")
        return data


    def truncate_tokens(self, tokens_a, tokens_b):
        # -3 for [CLS] [SEP] [SEP]
        if len(tokens_b) == 0 and len(tokens_a) <= self.max_seq_len - 2:
            truncate = False
        elif len(tokens_b) and len(tokens_a) + len(tokens_b) <= self.max_seq_len - 3:
            truncate = False
        else:
            truncate = True

        if truncate:
            if not len(tokens_b):
                tokens_a = tokens_a[:self.max_seq_len - 2]
            else:
                half_max_len = self.max_seq_len // 2 - 2
                if (self.max_seq_len - len(tokens_a) - 3 < len(tokens_b) // 2) or \
                   (self.max_seq_len - len(tokens_b) - 3 < len(tokens_a) // 2):
                    tokens_a = tokens_a[:half_max_len]
                    tokens_b = tokens_b[:half_max_len]
                else:
                    tokens_a = tokens_a[:self.max_seq_len - 3] 
                    tokens_b = tokens_b[:self.max_seq_len - len(tokens_a)]
        return tokens_a, tokens_b


    def build_inputs(self, text_a: str, text_b: str):
        if self.pre_tokenize:
            tokens_a = process_pre_tokenized(text_a.split(" "), self.tokenizer)
            tokens_b = process_pre_tokenized(text_b.split(" "), self.tokenizer) if text_b else []
        else:
            tokens_a = self.tokenizer.tokenize(text_a)
            tokens_b = self.tokenizer.tokenize(text_b) if text_b else []

        tokens_a, tokens_b = self.truncate_tokens(tokens_a, tokens_b)
        input_tokens = [self.tokenizer.cls_token] + \
            tokens_a + [self.tokenizer.sep_token]
        token_type_ids = [0 for _ in range(len(input_tokens))]
        if text_b:
            input_tokens += tokens_b + [self.tokenizer.sep_token]
        input_ids = self.tokenizer.convert_tokens_to_ids(input_tokens)
        token_type_ids = token_type_ids + \
            [1 for _ in range(len(input_ids) - len(token_type_ids))]
        attn_mask = [1 for _ in range(len(input_ids))]
        return {"input_ids": input_ids, "token_type_ids": token_type_ids, "attention_mask": attn_mask}

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return self.num_samples


class SNLIDataset(SeqCLSDataset):
    def __init__(self, **args):
        super().__init__(**args)

    @staticmethod
    def read_sample(line):
        data_i = json.loads(line)
        text_a = data_i["sentence1"].strip()
        text_b = data_i["sentence2"].strip()
        label = data_i["gold_label"].strip()
        return None, text_a, text_b, label

class IMDBDataset(SeqCLSDataset):
    # https://www.kaggle.com/datasets/atulanandjha/imdb-50k-movie-reviews-test-your-bert?select=train.csv
    def __init__(self, **args):
        super().__init__(**args)

    @staticmethod
    def read_line(path):
        df = pd.read_csv(path)
        for idx, row in tqdm(df.iterrows()):
            yield idx, (row["text"], row["sentiment"])

    @staticmethod
    def read_sample(items):
        text_a, label = items
        return None, text_a, "", label