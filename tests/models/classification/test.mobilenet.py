import os
import sys
ROOT = os.getcwd()
sys.path.insert(0, os.path.join(ROOT, "python"))

import numpy as np
import torch

import tvm
from tvm import ir
from tvm.mrt.utils import *
from tvm.mrt import runtime
from tvm.mrt import stats, dataset
from tvm.mrt import utils

#TODO: error data threshold is too small.
batch_size = 1
image_shape = (3, 224, 224)
data_shape = (batch_size,) + image_shape

def test_accuracy(model, test_loader):
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    model.eval()
    correct = 0
    iter_cnt = 0
    with torch.no_grad():
        for data, target in test_loader:
            iter_cnt += 1
            data, target = data.to(device), target.to(device)
            output = torch.squeeze(model(data))
            pred = torch.argmax(output).numpy()
            correct += (pred == target.numpy())

    print('\nTest set total: Accuracy: {}/{} ({}%)\n'.format(
        correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

from tvm.mrt.dataset_torch import TorchImageNet, TorchWrapperDataset
#  ds = TorchImageNet(
#          batch_size=batch_size,
#          img_size=image_shape[1:],)
#  data, _ = ds.next()

import torchvision
data_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize(image_shape[-2:]),
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
])
data_transform = torchvision.models.MobileNet_V2_Weights.IMAGENET1K_V1.transforms()
dataset_ = torchvision.datasets.ImageFolder(
        '~/.mxnet/datasets/imagenet/val',
        transform=data_transform)
test_loader = torch.utils.data.DataLoader(
        dataset_, batch_size=batch_size)
ds = TorchWrapperDataset(test_loader)
data, _ = ds.next()
#  test_accuracy(model, test_loader)
# finish test eval.

def load_model_from_torch() -> (ir.IRModule, ParametersT):
    model = torchvision.models.mobilenet_v2(weights='DEFAULT')
    model = model.eval()
    # begin test eval.
    #  data_transform = torchvision.transforms.Compose([
    #      torchvision.transforms.Resize(256),
    #      torchvision.transforms.CenterCrop(224),
    #      torchvision.transforms.ToTensor(),
    #      torchvision.transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
    #  ])
    #  dataset_ = torchvision.datasets.ImageFolder('~/.mxnet/datasets/imagenet/val', transform=data_transform)
    #  test_loader = torch.utils.data.DataLoader(dataset_)
    #  from utility import utility
    #  utility.print_dataLoader_first(test_loader)
    #  test_accuracy(model, test_loader)
    # finish test eval.
    input_data = torch.randn(data_shape)
    input_data = data_transform(input_data)
    script_module = torch.jit.trace(model, [input_data]).eval()
    return tvm.relay.frontend.from_pytorch(
            script_module, [ ("input", data_shape) ])

mod, params = load_model_from_torch()
mod: tvm.IRModule = mod
func: tvm.relay.function.Function = mod["main"]
expr: ir.RelayExpr = func.body

from tvm.mrt.trace import Trace
from tvm.mrt.opns import *
from tvm.mrt.symbol import *

with open("/tmp/expr.log", "w") as f:
    f.write(str(expr))

tr = Trace.from_expr(expr, params, model_name="mobilenet_v2")
#  tr = tr.subgraph(onames=["%11"])
tr.checkpoint()
tr.log()
#  tr.print(short=True, param_config={ "use_all": True, })

with open("/tmp/expr-trans.log", "w") as f:
    f.write(str(expr))

config = {
        "device": tvm.runtime.cuda(1),
        "target": tvm.target.cuda() }
runtime.multiple_validate(
        tr.populate(**config),
        dataset=ds,
        stats_type=stats.ClassificationOutput,
        max_iter_num=20,
)
#sys.exit()

from tvm.mrt import fuse
from tvm.mrt import op
from tvm.mrt.calibrate import Calibrator, SymmetricMinMaxSampling

#  const_tr = tr.checkpoint_transform(
#          force=True,)
#  const_tr.log()

fuse_tr = tr.checkpoint_transform(
        fuse.FuseConstant.apply(),
        tr_name = "fuse",
        force=True,
        )
fuse_tr.log()
#  fuse_tr.print(short=True)

fuse_tr = fuse_tr.checkpoint_transform(
        fuse.FuseAvgPool2D.apply(),
        fuse.FuseNaiveSoftmax.apply(),
        tr_name = "fuse-post",
        #  force=True,
        )
fuse_tr.log()

fuse_tr = fuse_tr.checkpoint_transform(
        fuse.FuseTupleGetItem.apply(),
        fuse.FuseDropout.apply(),
        fuse.FuseBatchNorm.apply(),
        #  fuse.FuseNaiveMathmatic.apply(),
        #  force=True,
        )
fuse_tr.log()


calib_tr = fuse_tr.checkpoint_transform(
        Calibrator.apply(data=tvm.nd.array(data)),
        #  print_bf=True,
        print_af=True,
        #  force=True,
)

sample_tr = calib_tr.checkpoint_transform(
        SymmetricMinMaxSampling.apply(),
        #  print_bf=True, print_af=True,
        #  force=True,
        )
sample_tr.log()

#  from tvm.mrt.precision import PrecisionAnnotator

#  prec_tr = sample_tr.checkpoint_transform(
#          PrecisionAnnotator.apply(),
#          print_bf=True, print_af=True,
#          )
#  prec_tr.log()

from tvm.mrt.discrete import Discretor
dis_tr = sample_tr.checkpoint_transform(
        Discretor.apply(),
        fuse.FuseConstant.apply(),
        #  print_bf=True, print_af=True,
        force=True,
        )
dis_tr.log()

from tvm.mrt.fixed_point import FixPoint, Simulator
sim_tr = dis_tr.checkpoint_transform(
        Simulator.apply(),
        # force=True,
        )
# sim_tr.log()
# sim_tr.print(short=True)

qt_tr = dis_tr.checkpoint_transform(
        FixPoint.apply(),
        # print_bf = True, print_af = True,
        # force=True,
)
qt_tr.print(short=False)

config = {
        "device": tvm.runtime.cuda(1),
        "target": tvm.target.cuda() }
runtime.multiple_validate(
        tr.populate(**config),
        sim_tr.populate(**config),
        dataset=ds,
        stats_type=stats.ClassificationOutput,
        max_iter_num=20,
)
#sys.exit()

from tvm.mrt.zkml import circom, transformer, model as ZkmlModel

symbol, params = qt_tr.symbol, qt_tr.params
print(">>> Start circom gen...")
symbol, params = ZkmlModel.resize_batch(symbol, params)
#ZkmlModel.simple_raw_print(symbol, params)
symbol, params = transformer.change_name(symbol, params)
# set input as params
symbol_first = ZkmlModel.visit_first(symbol)
#print(">>> before circom gen ...", symbol_first, symbol_first.is_input(), symbol_first.is_param())
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
