import os
from os import path
import sys

ROOT = os.getcwd()
sys.path.insert(0, path.join(ROOT, "python"))

import tvm
from tvm import relay, ir

import numpy as np

batch_size = 1
image_shape = (1, 28, 28)
data_shape = (batch_size,) + image_shape

import torch
import torchvision as tv

transform_mnist = tv.transforms.Compose(
    [tv.transforms.ToTensor(), tv.transforms.Normalize((0.1307,), (0.3081,))]
)

# Mnist Dataset
dataset = tv.datasets.MNIST(
    "~/.mxnet/datasets/mnist/", download=True, transform=transform_mnist
)
test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
# Iteration: 199 | from_expr: Top1/5: 93.00%,98.50% | sim: Top1/5: 93.00%,98.50% | sim-clip: Top1/5: 92.00%,98.50% | sim-round: Top1/5: 93.00%,98.50% | sim-clip-round: Top1/5: 92.50%,98.50% |

# use mrt wrapper to uniform api for dataset.
from tvm.mrt.dataset_torch import TorchWrapperDataset

ds = TorchWrapperDataset(test_loader)

import torch.nn as nn


class MnistNet(nn.Module):
    def __init__(self):
        super(MnistNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 1, 22, 1)
        self.fc1 = nn.Linear(49, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        # output = x
        output = nn.functional.log_softmax(x, dim=1)
        # output = nn.functional.softmax(x, dim=1)
        return output


def load_model_from_torch():  # -> (ir.IRModule, ParametersT):
    model = torch.load(
        "mnist_cnn.pt", map_location=torch.device("cpu")
    )  # this model is trained locally
    model = model.eval()
    input_data = torch.randn(data_shape)
    script_module = torch.jit.trace(model, [input_data]).eval()
    return tvm.relay.frontend.from_pytorch(script_module, [("input", data_shape)])


model_name = "mnist_cnn"

mod, params = load_model_from_torch()
mod: tvm.IRModule = mod
func: tvm.relay.function.Function = mod["main"]
expr: ir.RelayExpr = func.body

from tvm.mrt.trace import Trace
from tvm.mrt.opns import *
from tvm.mrt.symbol import *

tr = Trace.from_expr(expr, params, model_name=model_name)
from tvm.mrt import stats

tr.bind_dataset(ds, stats.ClassificationOutput).log()

dis_tr = tr.discrete(force=True)
sim_tr = dis_tr.export("sim").log()
sim_clip_tr = dis_tr.export("sim-clip").log()
sim_round_tr = dis_tr.export("sim-round").log()
sim_quant_tr = dis_tr.export("sim-clip-round").log()
fixpt_tr = dis_tr.export("fixpt").log()

tr.validate_accuracy(
    sim_tr,
    sim_clip_tr,
    sim_round_tr,
    sim_quant_tr,
    max_iter_num=200,
    device=tvm.runtime.cuda(1),
    target=tvm.target.cuda("3090"),
)
# sys.exit()

from tvm.mrt import trace_to_circom

circomTfm: trace_to_circom.CircomTfm = trace_to_circom.CircomTfm(fixpt_tr)
circomTfm.run(output_name=f"circom_{model_name}")
print("CircomTfm Done.")
sys.exit()
