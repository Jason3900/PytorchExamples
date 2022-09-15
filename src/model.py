# -*- coding:UTF-8 -*-
import torch.nn as nn
from transformers import AutoModel, AutoConfig


class BertCLSModel(nn.Module):
    def __init__(self, pretrained_path, num_labels, dropout_prob=0.0, pooler_type="cls"):
        super().__init__()
        self.num_labels = num_labels
        self.encoder = AutoModel.from_pretrained(pretrained_path)
        self.encoder_config = AutoConfig.from_pretrained(pretrained_path)
        self.pooler = BertPooler(self.encoder_config.hidden_size, pooler_type)
        self.classifier = nn.Linear(
            self.encoder_config.hidden_size, num_labels)
        self.dropout = nn.Dropout(dropout_prob) if dropout_prob else None
        self.loss_fn = nn.CrossEntropyLoss()
        self._init_weights(self.classifier)

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, batch):
        last_hidden_states, _ = self.encoder(input_ids=batch["input_ids"],
                                             token_type_ids=batch["token_type_ids"],
                                             attention_mask=batch["attention_mask"],
                                             return_dict=False)
        pooled_output = self.pooler(last_hidden_states)
        if self.dropout is not None:
            pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if batch["label"] is not None:
            loss = self.loss_fn(
                logits.view(-1, self.num_labels), batch["label"].view(-1))
        return loss, logits


class BertPooler(nn.Module):
    def __init__(self, hidden_size, mode="cls"):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()
        self.mode = mode

    def forward(self, hidden_states):
        if self.mode == "cls":
            hidden_states = hidden_states[:, 0]
        elif self.mode == "mean":
            hidden_states = hidden_states.mean(dim=1)
        else:
            raise NotImplementedError
        pooled_output = self.dense(hidden_states)
        pooled_output = self.activation(pooled_output)
        return pooled_output
