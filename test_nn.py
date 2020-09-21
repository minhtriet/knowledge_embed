from nn import DistMultNN
import torch

def test_event_embedding():
    x = DistMultNN("fb15k")
    x.holograhpic_embedding(torch.tensor([1,0,0,0], dtype=torch.float).reshape((4,1)),
                            torch.tensor([0,1,0], dtype=torch.float).reshape((3,1)),
                            torch.tensor([0,1,0,0], dtype=torch.float).reshape((4,1)))
