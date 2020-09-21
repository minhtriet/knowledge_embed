from nn import DistMultNN
import torch.optim as optim
import json
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from data import RelationDataset
writer = SummaryWriter('runs/')
dataset_name ="FB15K"
NO_NEGSAMPLES = 10

with open("config.json") as f:
    ds = json.load(f)
NO_ENTITIES = ds[dataset_name]['no_entities']
NO_RELATIONSHIPS = ds[dataset_name]['no_relationships']
BATCH_SIZE = 8
ENCODING_DIM = 10
model = DistMultNN(NO_ENTITIES, NO_RELATIONSHIPS, ENCODING_DIM)
optimizer = optim.SGD(model.parameters(), lr=0.0001)
dataset = RelationDataset("data/fb15k", no_negsamples=NO_NEGSAMPLES)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)  # only to load train data


for i_batch, sample_batched in enumerate(dataloader):
    # Forward pass: Compute predicted y by passing x to the model
    optimizer.zero_grad()
    writer.add_graph(model, sample_batched)
    writer.close()
    y_pred = model(sample_batched)
    # Compute and print loss
    loss = model.loss(y_pred)
    # if i_batch % 20 == 0:
    #     print(i_batch, loss.item())
    # Zero gradients, perform a backward pass, and update the weights.
    loss.backward()
    optimizer.step()
    break
