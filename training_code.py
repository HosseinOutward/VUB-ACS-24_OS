import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import pandas as pd
import numpy as np
from Utilities import LogDataset
import matplotlib.pyplot as plt
import os
import pytorch_lightning as pl
from torch.optim import Adam
from transformers import BertConfig, BertModel


class CustomBERT(nn.Module):
    def __init__(self, class_elements=7, logit_elements=2, hidden=10, token_count=400, class_count=399):
        super(CustomBERT, self).__init__()
        self.device = 'cpu'
        self.class_elements = class_elements
        self.logit_elements = logit_elements
        self.all_elements = class_elements + logit_elements
        self.hidden = hidden
        self.token_size = token_count
        self.class_count = class_count
        self.bert_model = BertModel(BertConfig(
            hidden_size=hidden * self.all_elements,
            num_hidden_layers=10,
            num_attention_heads=self.all_elements,
            intermediate_size=1000,
            type_vocab_size=1,
            max_position_embeddings=token_count,
        )).to(float)

        self.mask_emb = nn.Parameter(torch.randn(self.all_elements * hidden)).to(float)

        self.embeddings = [nn.Embedding(class_count + 1, hidden, padding_idx=class_count).to(float) for _ in
                           range(class_elements)]
        self.linear_vec = [nn.Linear(1, hidden).to(float) for _ in range(logit_elements)]

        self.position_embeddings = nn.Embedding(token_count, self.all_elements * hidden)

        self.position_ids = torch.arange(token_count).to(int)

        self.class_projector_head = [nn.Linear(hidden, class_count).to(float) for i in range(self.class_elements)]
        self.logit_projector_head = [nn.Linear(hidden, 1).to(float) for i in range(self.logit_elements)]

    def to(self, *args, **kwargs):
        self.device = args[0]
        self.bert_model = self.bert_model.to(*args, **kwargs)
        self.mask_emb = self.mask_emb.to(*args, **kwargs)
        self.embeddings = [embedding.to(*args, **kwargs) for embedding in self.embeddings]
        self.linear_vec = [linear.to(*args, **kwargs) for linear in self.linear_vec]
        self.position_embeddings = self.position_embeddings.to(*args, **kwargs)
        self.position_ids = self.position_ids.to(*args, **kwargs).to(int)
        self.class_projector_head = [head.to(*args, **kwargs) for head in self.class_projector_head]
        self.logit_projector_head = [head.to(*args, **kwargs) for head in self.logit_projector_head]
        return super().to(*args, **kwargs)

    def forward(self, x_class, x_value, mask_idx):
        x_class = [self.embeddings[i](x_class[:, :, i]) for i in range(self.class_elements)]
        x_value = [self.linear_vec[i](x_value[:, :, i].unsqueeze(2)) for i in range(self.logit_elements)]
        x = torch.cat(x_class + x_value, dim=2)
        x[:, mask_idx] = self.mask_emb
        x_class, x_value = None, None

        position_embeddings = self.position_ids.unsqueeze(0).expand_as(x[:, :, 0])
        position_embeddings = self.position_embeddings(position_embeddings)
        x = x + position_embeddings
        position_embeddings = None

        x = self.bert_model(inputs_embeds=x, attention_mask=torch.ones(x.shape[:-1]).to(float).to(self.device))

        x = x.last_hidden_state
        x = torch.split(x, self.hidden, dim=2)
        x_class, x_value = x[:self.class_elements], x[self.class_elements:]
        x = None

        x_class = torch.stack(
            [F.softmax(self.class_projector_head[i](x_class[i]), 2) for i in range(self.class_elements)]).transpose(1,
                                                                                                                    0).transpose(
            1, 2)
        x_value = torch.stack(
            [F.sigmoid(self.logit_projector_head[i](x_value[i])) for i in range(self.logit_elements)]).transpose(1,
                                                                                                                 0).transpose(
            1, 2)
        return x_class, x_value


class LitModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = CustomBERT()
        self.criterion_class = nn.CrossEntropyLoss()
        self.criterion_scaler = nn.MSELoss()
        self.eye_matrix = torch.tensor(np.eye(int(399))).to(float).to(self.device)
        self.loss_history = [[], []]

    def to(self, *args, **kwargs):
        self.model = self.model.to(*args, **kwargs)
        self.eye_matrix = self.eye_matrix.to(*args, **kwargs)
        return super().to(*args, **kwargs)

    def forward(self, x):
        return self.model(x)

    def to_loss_ready(self, data_class, data_value):
        return (torch.stack(
                            [self.eye_matrix[data_class[:, :, i]] for i in range(7)]
                        ).transpose(0, 1).transpose(1, 2),
                data_value.unsqueeze(3))

    def training_step(self, batch, batch_idx):
        _, _, (data_class, data_logit), col = batch
        idx = data_class.shape[1] // 2

        outputs_class, outputs_logit = self.model(data_class.to(self.device), data_logit.to(self.device), idx)

        data_class, data_logit = self.to_loss_ready(data_class, data_logit)

        loss_class = self.criterion_class(outputs_class, data_class)
        loss_logit = self.criterion_scaler(outputs_logit, data_logit)

        ll = 2
        if batch_idx % ll == 0:
            self.loss_history[0].append(loss_class.item() / ll)
            self.loss_history[1].append(loss_logit.item() / ll)
        else:
            self.loss_history[0][-1] += loss_class.item() / ll
            self.loss_history[1][-1] += loss_logit.item() / ll

        return loss_class  # + loss_logit

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=0.1)
        return optimizer


def get_dataset(dataset, limit=float('inf')):
    temp = [i * 400 * 2 + 100 for i in range(len(dataset) // 900)]
    np.random.shuffle(temp)
    train_idx, test_idx = temp[:int(len(temp) * 0.8)], temp[int(len(temp) * 0.8):]
    train_idx = np.array([i + j for i in train_idx for j in range(100)])
    test_idx = np.array([i + j for i in test_idx for j in range(100)])
    np.random.shuffle(train_idx)
    np.random.shuffle(test_idx)

    train_dataset = torch.utils.data.Subset(dataset, train_idx[:min(limit, len(train_idx))])
    test_dataset = torch.utils.data.Subset(dataset, test_idx[:min(limit//5, len(test_idx))])

    return train_dataset, test_dataset, train_idx, test_idx
