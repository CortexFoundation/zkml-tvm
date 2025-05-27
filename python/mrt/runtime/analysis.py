from os import path
import enum

import tvm
import numpy as np

from mrt.common.types import *

class StatsConfig(enum.Enum):
    NONE    = enum.auto()
    ALL     = enum.auto()

    ACCURACY    = enum.auto()
    """ enable accuracy info in stats, True by default. """
    TIME        = enum.auto()
    """ enable time logger in stats. """
    DL          = enum.auto()
    """ print current DataLabelT's info, this will suppress all other config. """


class Statistics:
    def __init__(self):
        self.stats_info = {}

    def reset(self):
        """ reset statistic status. """
        raise RuntimeError("Accuracy Type Error")

    def merge(self, dl: DataLabelT):
        """ merge model output and update status. """
        raise RuntimeError("Accuracy Type Error")

    def info(self) -> str:
        """ return statistic information. """
        raise RuntimeError("Accuracy Type Error")

    def dl_info(self) -> str:
        """ return current DataLabel information. """
        raise RuntimeError("Accuracy Type Error")


class ClassificationOutput(Statistics):
    def __init__(self):
        self.num_classes = None
        self.data, self.label = None, None

        self.top1_hit = 0
        self.top5_hit = 0
        self.dl_total = 0

        self.dl_top1, self.top1_raw = [], []
        self.dl_top5, self.top5_raw = [], []

    def reset(self):
        self.top1_hit = 0
        self.top5_hit = 0
        self.dl_total = 0

    def merge(self, dl: DataLabelT):
        data, label = dl
        self.argsort = [ np.argsort(d).tolist() \
                for d in data]

        self.dl_top1 = [a[-1] for a in self.argsort]
        self.dl_top5 = [a[-5:] for a in self.argsort]
        self.top1_raw = [ data[i][b] \
                for i, b in enumerate(self.dl_top1) ]
        self.top5_raw = [ [data[i][a] for a in b] \
                for i, b in enumerate(self.dl_top5) ]

        assert len(data.shape) == 2
        self.batch = data.shape[0]
        assert len(label.shape) == 1
        assert self.batch == label.shape[0]
        if self.num_classes is None:
            self.num_classes = data.shape[1]
        else:
            assert self.num_classes == data.shape[1]

        label = label.tolist()
        self.dl_total += self.batch
        for d, l in zip(self.dl_top1, label):
            self.top1_hit += (d == int(l))

        for d, l in zip(self.dl_top5, label):
            self.top5_hit += (int(l) in d)

    def dl_info(self, label_func):
        print("=" * 50)
        print("Batch: {}, Class Number: {}".format(
            self.batch, self.num_classes))
        top1, top1_raw = self.dl_top1, self.top1_raw
        top5, top5_raw = self.dl_top5, self.top5_raw
        for i in range(self.batch):
            print("{:5} Top1: {} | Raw: {}".format(
                i, top1[i], top1_raw[i]))
            print("{:5} Top5: {} | Raw: {}".format(
                i, top5[i], top5_raw[i]))
            if label_func:
                print("{:5} Lab5: {}".format(
                    i, label_func(*self.dl_top5[i])))
        print("=" * 50)

    def info(self):
        return "Top1/5: {:4.2%},{:4.2%}".format(
                (1. * self.top1_hit / self.dl_total),
                (1. * self.top5_hit / self.dl_total))

