import data
import pytest

no_negsamples = 10

testdata = [
    ("train", 0, (0, 0, 1)), ("train", -1, (8867, 565, 4237)), ("train", 49, (96, 41, 24)),
    ("dev", 0, (5167, 52, 1427)), ("dev", -1, (1142, 166, 2156)), ("dev", 49, (9071, 43, 101)),
    ("test", 0, (453, 37, 1347)), ("test", -1, (764, 46, 779)), ("test", 49, (1814, 11, 480)),
]


@pytest.mark.parametrize("slice, index, expect", testdata)
def test_train_data(slice, index, expect):
    ds = data.RelationDataset("data/FB15K", no_negsamples=no_negsamples, slice=slice)
    assert (ds.data[0][index], ds.data[1][index], ds.data[2][index]) == expect
