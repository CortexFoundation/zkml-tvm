import os
import sys

ROOT = os.getcwd()
sys.path.insert(0, os.path.join(ROOT, "python"))

import tvm
from tvm import relay, ir
import numpy as np

batch_size = 16
image_shape = (1, 28, 28)
data_shape = (batch_size,) + image_shape

import torch
import torchvision as tv
data_transform = tv.transforms.Compose([
    tv.transforms.ToTensor(),
    tv.transforms.Normalize(
        [0.5], [0.5])
])
# data_transform = tv.models.MobileNet_V2_Weights.IMAGENET1K_V1.transforms()
dataset = tv.datasets.MNIST(
        '~/.mxnet/datasets/mnist/',
        download=True,
        transform=data_transform)
test_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size, # set dataset batch load
        )

class MnistNet(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 1, 20, 1)
        self.fc1 = torch.nn.Linear(81, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        #output = x
        output = torch.nn.functional.log_softmax(x, dim=1)
        return output

# use mrt wrapper to uniform api for dataset.
from tvm.mrt.dataset_torch import TorchWrapperDataset
ds = TorchWrapperDataset(test_loader)

# model inference context, like cpu, gpu, etc.
config = {"device": tvm.runtime.cuda(1),
        "target": tvm.target.Target("cuda -arch=sm_86") }

model_name = "mnist_cnn"
model = torch.load("mnist_cnn.pt0", map_location=torch.device('cpu')) # this model is trained locally
model = model.eval()
input_data = torch.randn(data_shape)
script_module = torch.jit.trace(model, [input_data]).eval()
mod, params = relay.frontend.from_pytorch(
        script_module, [ ("input", data_shape) ])

# MRT Procedure
mod: tvm.IRModule = mod
func: relay.function.Function = mod["main"]
expr: ir.RelayExpr = func.body

from tvm.mrt import stats
from tvm.mrt.trace import Trace
tr = Trace.from_expr(expr, params, model_name=model_name)
tr.bind_dataset(ds, stats.ClassificationOutput).log()

# tr.validate_accuracy(max_iter_num=1, **config)

fuse_tr = tr.fuse().log()
calib_tr = fuse_tr.calibrate(
        # force=True,
        batch_size=16).log()

from tvm.mrt.config import Pass
with Pass(log_before=True, log_after=True):
    dis_tr = calib_tr.quantize().log()

sim_tr = dis_tr.export().log()
sim_clip_tr = dis_tr.export(with_clip=True).log()
sim_round_tr = dis_tr.export(with_round=True).log()
sim_quant_tr = dis_tr.export(
        with_clip=True, with_round=True).log()

circom_tr = dis_tr.export(force=True, use_simulator=False).log()

tr.validate_accuracy(
        sim_tr,
        sim_clip_tr,
        sim_round_tr,
        sim_quant_tr,
        max_iter_num=1,
        **config)
print("ValidateAccuracy Done!!!")
#sys.exit()

circom_tr.print()

from tvm.mrt.zkml import circom, transformer, model as ZkmlModel
symbol, params = circom_tr.symbol, circom_tr.params
print(">>> Start circom gen...")
symbol, params = ZkmlModel.resize_batch(symbol, params)
symbol, params = transformer.change_name(symbol, params)
ZkmlModel.simple_raw_print(symbol, params)
# set input as params
symbol_first = ZkmlModel.visit_first(symbol)
input_data = torch.randint(255, image_shape)
params[symbol_first.name] = input_data
circom_out, circom_gen_map = transformer.model2circom(symbol, params)
print(">>> Generating circom code ...")
circom_code = circom.generate(circom_out)
print(">>> Generating circom input ...")
input_json = circom.input_json(circom_gen_map, params)

output_name = "circom_model_test"
print(">>> Generated, dump to {} ...".format(output_name))
with open(output_name + ".circom", "w") as f:
    f.write(circom_code)
with open(output_name + ".json", "w") as f:
    import json
    f.write(json.dumps(input_json, indent=2))

print(">>> success exit sys +1 <<<")
sys.exit(+1)
