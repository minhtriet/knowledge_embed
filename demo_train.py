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
train_dataset = RelationDataset("data/fb15k", no_negsamples=NO_NEGSAMPLES, slice="train")
train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)  # only to load train data

NO_ENTITIES = ds[dataset_name]['no_entities']
NO_RELATIONSHIPS = ds[dataset_name]['no_relationships']
BATCH_SIZE = 8
ENCODING_DIM = 10
nn = HarmonNet(NO_ENTITIES, NO_RELATIONSHIPS)
optimizer = optim.SGD(nn.parameters(), lr=0.00001)
nn.train()
min_loss = float('inf')

val_dataset = RelationDataset("data/fb15k", no_negsamples=NO_NEGSAMPLES, slice="dev")
val_dataloader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False)

for i_batch, sample_batched in enumerate(train_dataloader):
    optimizer.zero_grad()
    with torch.set_grad_enabled(True):
        y_pred = nn.forward(sample_batched)
        loss = nn.loss(y_pred)
        if (i_batch > 100000) and (min_loss > loss.item()):
            torch.save(nn.state_dict(), "best_model")
            min_loss = loss.item()
        loss.backward()
        optimizer.step()
    if i_batch % 40 == 0:
        with torch.no_grad():
            nn.eval()
            for i, val_batch in enumerate(val_dataloader):
                y_pred = nn.forward(val_batch)
                loss += nn.loss(y_pred)
            print("val loss", loss)