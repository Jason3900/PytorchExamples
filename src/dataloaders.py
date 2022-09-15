# -*- coding:UTF-8 -*-
import torch
from torch.utils.data import DataLoader


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
                value = torch.tensor(
                    [item[key] for item in batch], dtype=torch.long)
            else:
                value = [item[key] for item in batch]
            batch_dict[key] = value
        return batch_dict


def init_dataloader(dataset, batch_size: int, input_pad_id: int, shuffle: bool):
    collate_fn = BatchCollate(input_pad_id=input_pad_id)
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn
    )
    return data_loader
