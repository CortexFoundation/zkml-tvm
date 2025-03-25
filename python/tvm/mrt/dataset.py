from __future__ import annotations

import typing

from os import path

import tvm

from .types import *

class Dataset:
    def next(self) -> typing.Optional[DataLabelT]:
        """ get next data, None if end. """
        raise RuntimeError("Base Dataset Error: next")

    def reset(self):
        """ reset dataset internal reader status. """
        raise RuntimeError("Base Dataset Error: reset")

    def resize(self, batch_size: int) -> Dataset:
        raise RuntimeError("Base Dataset Error: batch resize")

    def __len__(self):
        raise RuntimeError("Base Dataset Error: __len__")

    def label(self, index):
        return index

    def labels(self, *indexes):
        return [ self.label(i) for i in indexes ]

class ImageNet(Dataset):
    category_name = "imagenet_category.json"

    def __init__(self):
        base_dir = path.join(path.dirname(__file__), "datasets")

        with open(path.join(base_dir, self.category_name)) as f:
            self.synset = eval(f.read())

    def label(self, index):
        return self.synset.get(index, "unknown category")

class Coco(Dataset):
    category_name = "coco_category.json"

    def __init__(self):
        base_dir = path.join(path.dirname(__file__), "datasets")

        with open(path.join(base_dir, self.category_name)) as f:
            self.synset = eval(f.read())

    def label(self, index):
        return self.synset.get(index, "unknown category")

class Voc(Dataset):
    category_name = "voc_category.json"

    def __init__(self):
        base_dir = path.join(path.dirname(__file__), "datasets")

        with open(path.join(base_dir, self.category_name)) as f:
            self.synset = eval(f.read())

    def label(self, index):
        return self.synset.get(index, "unknown category")

class Cifar10(Dataset):
    category_name = "cifar10_category.json"

    def __init__(self):
        base_dir = path.join(path.dirname(__file__), "datasets")

        with open(path.join(base_dir, self.category_name)) as f:
            self.synset = eval(f.read())

    def label(self, index):
        return self.synset.get(index, "unknown category")

class Cifar100(Dataset):
    category_name = "cifar100_category.json"

    def __init__(self):
        base_dir = path.join(path.dirname(__file__), "datasets")

        with open(path.join(base_dir, self.category_name)) as f:
            self.synset = eval(f.read())

    def label(self, index):
        return self.synset.get(index, "unknown category")

class QuickDraw(Dataset):
    category_name = "quickdraw_category.json"

    def __init__(self):
        base_dir = path.join(path.dirname(__file__), "datasets")

        with open(path.join(base_dir, self.category_name)) as f:
            self.synset = eval(f.read())

    def label(self, index):
        return self.synset.get(index, "unknown category")

class Mnist(Dataset):
    category_name = "mnist_category.json"

    def __init__(self):
        base_dir = path.join(path.dirname(__file__), "datasets")

        with open(path.join(base_dir, self.category_name)) as f:
            self.synset = eval(f.read())

    def label(self, index):
        return self.synset.get(index, "unknown category")

class Trec(Dataset):
    category_name = "trec_category.json"

    def __init__(self):
        base_dir = path.join(path.dirname(__file__), "datasets")

        with open(path.join(base_dir, self.category_name)) as f:
            self.synset = eval(f.read())

    def label(self, index):
        return self.synset.get(index, "unknown category")

class Country211(Dataset):
    category_name = "category.json"

    def __init__(self):
        base_dir = path.join(path.dirname(__file__), "datasets")

        with open(path.join(base_dir, self.category_name)) as f:
            self.synset = eval(f.read())

    def label(self, index):
        return self.synset.get(index, "unknown category")

class Dtd(Dataset):
    category_name = "category.json"

    def __init__(self):
        base_dir = path.join(path.dirname(__file__), "datasets")

        with open(path.join(base_dir, self.category_name)) as f:
            self.synset = eval(f.read())

    def label(self, index):
        return self.synset.get(index, "unknown category")

class Emnist(Dataset):
    category_name = "category.json"

    def __init__(self):
        base_dir = path.join(path.dirname(__file__), "datasets")

        with open(path.join(base_dir, self.category_name)) as f:
            self.synset = eval(f.read())

    def label(self, index):
        return self.synset.get(index, "unknown category")

class FashionMNIST(Dataset):
    category_name = "category.json"

    def __init__(self):
        base_dir = path.join(path.dirname(__file__), "datasets")

        with open(path.join(base_dir, self.category_name)) as f:
            self.synset = eval(f.read())

    def label(self, index):
        return self.synset.get(index, "unknown category")

class Fgvcaircraft(Dataset):
    category_name = "category.json"

    def __init__(self):
        base_dir = path.join(path.dirname(__file__), "datasets")

        with open(path.join(base_dir, self.category_name)) as f:
            self.synset = eval(f.read())

    def label(self, index):
        return self.synset.get(index, "unknown category")

class Flowers102(Dataset):
    category_name = "category.json"

    def __init__(self):
        base_dir = path.join(path.dirname(__file__), "datasets")

        with open(path.join(base_dir, self.category_name)) as f:
            self.synset = eval(f.read())

    def label(self, index):
        return self.synset.get(index, "unknown category")

class Food101(Dataset):
    category_name = "category.json"

    def __init__(self):
        base_dir = path.join(path.dirname(__file__), "datasets")

        with open(path.join(base_dir, self.category_name)) as f:
            self.synset = eval(f.read())

    def label(self, index):
        return self.synset.get(index, "unknown category")

class Gtsrb(Dataset):
    category_name = "category.json"

    def __init__(self):
        base_dir = path.join(path.dirname(__file__), "datasets")

        with open(path.join(base_dir, self.category_name)) as f:
            self.synset = eval(f.read())

    def label(self, index):
        return self.synset.get(index, "unknown category")

class Kmnist(Dataset):
    category_name = "category.json"

    def __init__(self):
        base_dir = path.join(path.dirname(__file__), "datasets")

        with open(path.join(base_dir, self.category_name)) as f:
            self.synset = eval(f.read())

    def label(self, index):
        return self.synset.get(index, "unknown category")

class Lfwpeople(Dataset):
    category_name = "category.json"

    def __init__(self):
        base_dir = path.join(path.dirname(__file__), "datasets")

        with open(path.join(base_dir, self.category_name)) as f:
            self.synset = eval(f.read())

    def label(self, index):
        return self.synset.get(index, "unknown category")

class Omniglot(Dataset):
    category_name = "category.json"

    def __init__(self):
        base_dir = path.join(path.dirname(__file__), "datasets")

        with open(path.join(base_dir, self.category_name)) as f:
            self.synset = eval(f.read())

    def label(self, index):
        return self.synset.get(index, "unknown category")

class OxfordIIITPet(Dataset):
    category_name = "category.json"

    def __init__(self):
        base_dir = path.join(path.dirname(__file__), "datasets")

        with open(path.join(base_dir, self.category_name)) as f:
            self.synset = eval(f.read())

    def label(self, index):
        return self.synset.get(index, "unknown category")

class Rendered(Dataset):
    category_name = "category.json"

    def __init__(self):
        base_dir = path.join(path.dirname(__file__), "datasets")

        with open(path.join(base_dir, self.category_name)) as f:
            self.synset = eval(f.read())

    def label(self, index):
        return self.synset.get(index, "unknown category")

class Stl10(Dataset):
    category_name = "category.json"

    def __init__(self):
        base_dir = path.join(path.dirname(__file__), "datasets")

        with open(path.join(base_dir, self.category_name)) as f:
            self.synset = eval(f.read())

    def label(self, index):
        return self.synset.get(index, "unknown category")

class Usps(Dataset):
    category_name = "category.json"

    def __init__(self):
        base_dir = path.join(path.dirname(__file__), "datasets")

        with open(path.join(base_dir, self.category_name)) as f:
            self.synset = eval(f.read())

    def label(self, index):
        return self.synset.get(index, "unknown category")

class MemoryDataset(Dataset):
    def __init__(self, available_dls: typing.List[DataLabelT]):
        self.data = available_dls
        self._max = len(self.data)
        self._index = 0

    def __len__(self):
        return self._max

    def next(self) -> typing.Optional[DataLabelT]:
        if self._index < self._max:
            self._index += 1
            return self.data[self._index-1]
        return None

    def reset(self):
        self._index = 0
