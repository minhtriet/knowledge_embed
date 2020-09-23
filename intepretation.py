import torch
from harmon_net import HarmonNet
import json
from data import RelationDataset
from torch.utils.data import DataLoader

dataset_name = "FB15K"
with open("config.json") as f:
    ds = json.load(f)

NO_ENTITIES = ds[dataset_name]['no_entities']
NO_RELATIONSHIPS = ds[dataset_name]['no_relationships']


dataset = RelationDataset("data/fb15k")
dataset.test_data

model = HarmonNet(NO_ENTITIES, NO_RELATIONSHIPS)
model.load_state_dict(torch.load("best_model"))
model.eval()