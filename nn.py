import torch
from torch import nn
import numpy as np


class DistMultNN(nn.Module):

    def __init__(self, dataset, batch_size=32, learning_rate=1e-4):
        with open("config.json") as f:
            ds = json.load(f)
        if dataset not in ds:
            raise ValueError("invalid dataset name")
        self.NO_ENTITIES = ds[dataset]['no_entities']
        self.NO_RELATIONSHIPS = ds[dataset]['no_relationships']
        self.ENCODING_DIM = 10
        self.entities_embedding = nn.init.xavier_uniform_(torch.zeros((self.NO_ENTITIES, self.ENCODING_DIM)))
        self.relation_embedding = nn.init.xavier_uniform_(torch.zeros((self.NO_RELATIONSHIPS, self.ENCODING_DIM)))
        self.LEARNING_RATE = learning_rate
        self.BATCH_SIZE = batch_size
        self.W = torch.rand(self.ENCODING_DIM, self.ENCODING_DIM)  # W is symmetric
        self.W = (self.W + self.W.t()) / 2
        self.b = torch.rand(self.ENCODING_DIM, 1)
        self.lambda_ = torch.rand(1, 2)
        r = torch.randn(self.ENCODING_DIM, self.ENCODING_DIM)

    def forward(self, sample):
        """
        Deal with a single train item -> batch
        samples ((e1,r,e2), label)
        """
        entity_1 = self.entities_embedding[sample[0]]
        entity_2 = self.entities_embedding[sample[2]]
        r = self.relation_embedding[sample[1]]

        y_1 = torch.nn.ReLU(self.w.mm(entity_1))
        y_2 = torch.nn.ReLU(self.w.mm(entity_2))
        r_out = y_1.t().mm(r).mm(y_2)

        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.parameters(), lr=0.001)
        model = DistMultNN().to(rank)
        ddp_model = DDP(self, device_ids=[rank])
        optimizer.zero_grad()
        outputs = ddp_model(torch.randn(20, 10))
        labels = torch.randn(20, 5).to(rank)
        loss_fn(outputs, labels).backward()
        optimizer.step()
        loss.backward()

    def _circular_correl(self, e1, e2):
        return torch.irfft(torch.conj(torch.rfft(e1, 1)) * torch.rfft(e2, 1), 1).real

    def holograhpic_embedding(self, e1, r, e2):
        """
        from one hot encoding of e1, r, e2 to
        holograhpic_embedding. Eq. 9
        """
        return r.t().mm(self._circular_correl(e1, e2))

    def H(self, h, x):
        """
        H is the harmony score of h and an relation
        embedding x. Equation (1) in paper

        `x` is (e_l, r, e_r)
        `h` is a reference point somehow??
        """
        diff = h - x
        return h.t().mm(self.W).mm(h) + self.b.t().mm(h) - self.lambda_.mm(diff.t().mm(diff))

    def muy(self, x):
        """
        eq. 3
        x is an embedding of a relation
        """
        V = self.W - self.lambda_ * torch.eye(self.ENCODING_DIM)
        # todo recalculate V after every step
        return -0.5 * V.inverse() * self.b + self.lambda_ * x

    def loss(self, e1, r, e2, embed_func):
        """
        Eq. 13
        """
        x = embed_func(e1, r, e2)
        self.H(self.muy(x), embed_func(e1, r, e2))
