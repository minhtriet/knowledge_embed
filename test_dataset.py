import data
from torch import Size
import pytest
no_negsamples = 10


testdata = [
    ("train", 0, (0,0,1)), ("train", -1, (8867, 565, 4237)), ("train", 49, (96, 41, 24)),
#    ("val", 0, (5167, 52, 1427)), ("val", -1, (1142, 166, 2156)), ("val", 49, (9071, 43, 101)),
#    ("test", 0, (453, 37, 1347)), ("test", -1, (1814, 11, 480)), ("test", 49, (764, 46, 779)),
]


@pytest.mark.parametrize("slice, index, expect", testdata)
def testTrainData(slice, index, expect):
    ds = data.RelationDataset("data/FB15K", no_negsamples=no_negsamples)
    assert (ds.train_data[0][index], ds.train_data[1][index], ds.train_data[2][index]) == expect
