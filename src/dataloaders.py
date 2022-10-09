# -*- coding:UTF-8 -*-
import torch
from torch.utils.data import DataLoader, DistributedSampler


class BatchCollate:
    def __init__(self, input_pad_id):
        self.input_pad_id = input_pad_id

    def pad_to_max_len(self, input_seq, max_len, pad_value=0):
        pad_len = max_len - len(input_seq)
        pad_piece = [pad_value for _ in range(pad_len)]
        return input_seq + pad_piece

    def pad_instance(self, instance, max_len):
        """
        padding each tensor to max len
        """
        instance["input_ids"] = self.pad_to_max_len(
            instance["input_ids"], max_len, pad_value=self.input_pad_id)
        instance["token_type_ids"] = self.pad_to_max_len(
            instance["token_type_ids"], max_len, pad_value=instance["token_type_ids"][-1])
        instance["attention_mask"] = self.pad_to_max_len(
            instance["attention_mask"], max_len, pad_value=0)

        return instance

    def __call__(self, batch):

        max_len = max([len(i["input_ids"]) for i in batch])

        for item in batch:
            item = self.pad_instance(item, max_len)

        keys = item.keys()

        batch_dict = dict()
        for key in keys:
            if key != "uuid":
                # pytorch: It is generally not recommended to return CUDA tensors in multi-process loading
                # because of many subtleties in using CUDA and sharing CUDA tensors in multiprocessing
                # Instead, we recommend using automatic memory pinning (i.e., setting pin_memory=True),
                # which enables fast data transfer to CUDA-enabled GPUs.
                value = torch.tensor(
                    [item[key] for item in batch], dtype=torch.long)
            else:
                value = [item[key] for item in batch]
            batch_dict[key] = value
        return batch_dict


def init_dataloader(dataset, shuffle: bool, batch_size: int, input_pad_id: int, num_workers=0, is_distributed=False):
    collate_fn = BatchCollate(input_pad_id=input_pad_id)
    sampler = init_sampler(dataset=dataset,
                           shuffle=shuffle,
                           is_distributed=is_distributed)
    if is_distributed:
        # sampler option is mutually exclusive with shuffle
        shuffle = None

    data_loader = DataLoader(
        dataset=dataset,
        sampler=sampler,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,  # set pin memory to True to enable faster data transfer
        collate_fn=collate_fn
    )
    return data_loader


def init_sampler(dataset, shuffle: bool, is_distributed: bool):
    if is_distributed:
        import horovod.torch as hvd
        sampler = DistributedSampler(dataset=dataset,
                                     shuffle=shuffle,
                                     num_replicas=hvd.size(),
                                     rank=hvd.rank(),
                                     drop_last=True)
    else:
        sampler = None
    return sampler
