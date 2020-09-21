import data
from torch import Size

no_negsamples = 10
ds = data.RelationDataset("data/FB15K", no_negsamples=no_negsamples)

def testCorrectIO():
    inp = ds.__getitem__(1)
    assert inp.shape == Size([3 + 2*no_negsamples])