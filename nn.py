import torch
from torch import nn
import numpy as np


class DistMultNN(nn.Module):

    def __init__(self, no_entities, no_relationships, encoding_dim=10, lambda_=1.):
        super().__init__()
        self.seed = 42
        np.random.seed(self.seed)
        self.NO_ENTITIES = no_entities
        self.NO_RELATIONSHIPS = no_relationships
        self.ENCODING_DIM = encoding_dim
        self.entities_embedding = nn.Embedding(no_entities, encoding_dim)
        self.relation_embedding = nn.Embedding(no_relationships, encoding_dim)
        self.W = torch.rand(self.ENCODING_DIM, self.ENCODING_DIM,
                            requires_grad=True)  # W is symmetric, todo: requireGrad?
        self.W = (self.W + self.W.t()) / 2
        self.V = self.W.detach() - lambda_ * torch.eye(self.ENCODING_DIM)
        self.lambda_ = lambda_
        self.b = torch.rand(self.ENCODING_DIM, requires_grad=True)
        self.rnn = torch.nn.RNN(input_size=encoding_dim, hidden_size=1, num_layers=1, nonlinearity='relu')
        self.loss_func = torch.nn.LogSoftmax(dim=1)

    def loss(self, y_pred):
        """
        compute the batch loss
        """
        softmax = -1 * self.loss_func(y_pred)
        result = torch.mean(softmax[:, 0])
        return result

    def forward(self, samples):
        """
        Deal with a single train item -> batch
        samples [(e1,r,e2), ...]
        """
        batch_size = samples.shape[0]
        e1s = torch.gather(samples, 1, torch.tensor([[0]*batch_size]).view(-1,1))
        rs = torch.gather(samples, 1, torch.tensor([[1]*batch_size]).view(-1,1))
        e2s = torch.gather(samples, 1, torch.tensor([[2]*batch_size]).view(-1,1))
        entity_1 = self.entities_embedding(e1s)
        entity_2 = self.entities_embedding(e2s)
        r = self.relation_embedding(rs)
        x = self.harmonic_holograhpic_embedding(entity_1, r, entity_2)
        batch_result = self.score(x)
        return batch_result

    def _circular_correl(self, e1, e2):
        ccorel = torch.irfft(torch.conj(torch.rfft(e1, 1, onesided=False)) * torch.rfft(e2, 1, onesided=False),
                           1, onesided=False)
        return ccorel.transpose(1,2)

    def harmonic_holograhpic_embedding(self, e1, r, e2):
        """
        from one hot encoding of e1, r, e2 to
        harmonic holograhpic_embedding. Eq. 11
        * is element wise multiplication
        """
        return r * self._circular_correl(e1, e2)

    def H(self, h, x):
        """
        H is the harmony score of h and an relation
        embedding x. Equation (1) in paper

        `x` is the embedding of (e_l, r, e_r)
        `h` is a column vector (x, 1)
        """
        # h = next(self.rnn.parameters())  # hidden weight, row vector
        diff = h - x
        result = h.t(1,2).mm(self.W).mm(h) + h.mm(self.b) \
                 - self.lambda_ * diff.norm() ** 2  # this line is a hack for the dot product
        assert result.shape == torch.Size([1, 1])
        result = result.item()
        return result

    def muy(self, x):
        """
        eq. 3
        x is an embedding of a relation
        """
        V = self.W - (self.lambda_ * torch.eye(self.ENCODING_DIM))
        V = V.inverse()
        r = torch.einsum('...ij,jk', V, -0.5*self.b + self.lambda_ * x)
        assert r.shape == (V.shape[0], V.shape[1])
        for i in range(V.shape[0]):
            assert V[i].mm(-0.5*self.b + self.lambda_ * x).allclose(r[i])
        return r


    def score(self, x):
        """
        embed_func is harmonic_holograhpic_embedding
        Eq. 13
        """
        return self.H(self.muy(x), x)
