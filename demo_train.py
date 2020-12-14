import torch
import json
from data import RelationDataset
from torch.utils.data import DataLoader
import torch.optim as optim
from harmon_net import HarmonNet
from torch.utils.tensorboard import SummaryWriter
import logging


device = 'cuda' if torch.cuda.is_available() else 'cpu'
logging.basicConfig(format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s', level=logging.DEBUG)
dataset_name = "FB15K"
NO_NEGSAMPLES = 10
BATCH_SIZE = 32
ENCODING_DIM = 20
with open("config.json") as f:
    ds = json.load(f)
train_dataset = RelationDataset(f"data/{dataset_name}", no_negsamples=NO_NEGSAMPLES, slice="train")
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)  # only to load train data

NO_ENTITIES = ds[dataset_name]['no_entities']
NO_RELATIONSHIPS = ds[dataset_name]['no_relationships']
nn = HarmonNet(NO_ENTITIES, NO_RELATIONSHIPS, device=device).to(device)
optimizer = optim.Adam(nn.parameters(), lr=0.001)
min_loss = float('inf')
writer = SummaryWriter()
val_dataset = RelationDataset(f"data/{dataset_name}", no_negsamples=NO_NEGSAMPLES, slice="dev")
val_dataloader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False)


for epoch in range(8):
    for i_batch, sample_batched in enumerate(train_dataloader):
        nn.train()
        optimizer.zero_grad()
        with torch.set_grad_enabled(True):
            y_pred = nn.forward(sample_batched.to(device))
            writer.add_graph(nn, sample_batched)
            loss = nn.loss(y_pred)
            if (i_batch > 100000) and (min_loss > loss.item()):
                torch.save(nn.state_dict(), f"best_model_{ENCODING_DIM}")
                min_loss = loss.item()
            temp1=nn.entities_embedding.weight.clone()
            temp2=nn.W.clone()
            loss.backward()
            optimizer.step()
        if i_batch % 4096 == 0:
            with torch.no_grad():
                nn.eval()
                for i, val_batch in enumerate(val_dataloader):
                    y_pred = nn.forward(val_batch)
                    val_loss = nn.loss(y_pred)
                    logging.debug(val_loss)
                    writer.add_scalar('val loss', val_loss, i_batch)
writer.close()