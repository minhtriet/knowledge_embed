import torch
from torch import nn
import json
from data import RelationDataset
from torch.utils.data import DataLoader
import torch.optim as optim


class FooNet(nn.Module):
    def __init__(self, no_entities, no_relationships, encoding_dim=10, lambda_=1.):
        super().__init__()
        torch.manual_seed(42)
        self.NO_ENTITIES = no_entities
        self.NO_RELATIONSHIPS = no_relationships
        self.ENCODING_DIM = encoding_dim
        self.entities_embedding = nn.Embedding(no_entities, encoding_dim)
        self.relation_embedding = nn.Embedding(no_relationships, encoding_dim)
        self.W = torch.nn.Parameter(torch.rand(self.ENCODING_DIM, self.ENCODING_DIM))
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
        e1s, rs, e2s = samples[:,:,0], samples[:,:,1], samples[:,:,2]
        assert e1s.shape == (samples.shape[0], samples.shape[1])
        entity_1 = self.entities_embedding(e1s)
        entity_2 = self.entities_embedding(e2s)
        r = self.relation_embedding(rs)
        x = self.harmonic_holograhpic_embedding(entity_1, r, entity_2)
        batch_result = self.score(x)
        return batch_result

    def harmonic_holograhpic_embedding(self, e1, r, e2):
        return r * self._circular_correl(e1, e2)

    def _circular_correl(self, e1, e2):
        ccorel = torch.irfft(torch.conj(torch.rfft(e1, 1, onesided=False)) * torch.rfft(e2, 1, onesided=False),
                             1, onesided=False)
        return ccorel

    def _2d_norm_batch(self, x):
        """
        norm based on last dimension
        """
        result = torch.einsum('...k,...k', x, x)
        return result

    def H(self, h, x):
        diff = h - x
        Wsym = self.W + self.W.t()
        hW = torch.einsum('...ij,jk', h, Wsym)
        result = torch.einsum('ijk,ikj->ij', hW, h.transpose(1, 2)) + torch.einsum('...k,k', h,
                                                                                   self.b) - self.lambda_ * self._2d_norm_batch(
            diff)
        assert result.shape == torch.Size([x.shape[0], x.shape[1]])
        return result

    def muy(self, x):
        Wsym = self.W + self.W.t()
        V = Wsym - (self.lambda_ * torch.eye(self.ENCODING_DIM))
        V = V.inverse()
        r = torch.einsum('ijk,kt', -0.5 * self.b + self.lambda_ * x, V)
        assert r.shape == x.shape
        # for i in range(x.shape[0]):
        #     assert (-0.5 * self.b + self.lambda_ * x[i]).mm(V).allclose(r[i],atol=3e-7)
        return r

    def score(self, x):
        return self.H(self.muy(x), x)


dataset_name = "FB15K"
NO_NEGSAMPLES = 10
with open("config.json") as f:
    ds = json.load(f)
dataset = RelationDataset("data/fb15k", no_negsamples=NO_NEGSAMPLES)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)  # only to load train data

NO_ENTITIES = ds[dataset_name]['no_entities']
NO_RELATIONSHIPS = ds[dataset_name]['no_relationships']
BATCH_SIZE = 8
ENCODING_DIM = 10
nn = FooNet(NO_ENTITIES, NO_RELATIONSHIPS)
optimizer = optim.SGD(nn.parameters(), lr=0.00001)
nn.train()
min_loss = float('inf')
for i_batch, sample_batched in enumerate(dataloader):
    optimizer.zero_grad()
    with torch.set_grad_enabled(True):
        y_pred = nn.forward(sample_batched)
        loss = nn.loss(y_pred)
        if (i_batch > 100000) and (min_loss > loss.item()):
            torch.save(nn.state_dict(), "best_model")
            min_loss = loss.item()
        loss.backward()
    if i_batch % 20 == 0:
        print(i_batch, loss.item())
    optimizer.step()


