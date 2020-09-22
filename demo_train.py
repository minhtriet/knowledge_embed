import torch
import json
from data import RelationDataset
from torch.utils.data import DataLoader
import torch.optim as optim
from harmon_net import HarmonNet

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
nn = HarmonNet(NO_ENTITIES, NO_RELATIONSHIPS)
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
