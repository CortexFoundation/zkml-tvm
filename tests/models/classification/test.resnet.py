import os
import sys
ROOT = os.getcwd()
sys.path.insert(0, os.path.join(ROOT, "python"))

import numpy as np
from PIL import Image

import tvm
from tvm import ir
from tvm.mrt.utils import *
from tvm.mrt import runtime
from tvm.mrt import stats, dataset
from tvm.mrt import utils

batch_size = 16
image_shape = (3, 28, 28)
data_shape = (batch_size,) + image_shape

def load_model_from_torch() -> (ir.IRModule, ParametersT):
    import torch
    import torchvision

    model = torchvision.models.resnet18(weights='DEFAULT')
    model = model.eval()
    data_transform = torchvision.models.ResNet18_Weights.IMAGENET1K_V1.transforms()
    input_data = data_transform(torch.randn(data_shape))
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
tr = Trace.from_expr(expr, params, model_name="resnet18")
tr.checkpoint()
tr.print(param_config={ "use_all": True, })

from tvm.mrt import fuse
from tvm.mrt import op
fuse_tr = tr.checkpoint_transform(
        fuse.FuseTupleGetItem.apply(),
        fuse.FuseBatchNorm.apply(),
        fuse.FuseAvgPool2D.apply(),
        tr_name = "fuse",
        # force=True,
        )

from tvm.mrt.calibrate import Calibrator, SymmetricMinMaxSampling

calib_tr = fuse_tr.checkpoint_transform(
        Calibrator.apply(random_config={
            "enabled": True,
            "absmax": 1.0, }),
        print_bf=True, print_af=True,
)

from tvm.mrt.rules import slm
from tvm.mrt.quantize import Quantizer

dt_tr = calib_tr.checkpoint_transform(
        SymmetricMinMaxSampling.apply(),
        slm.SymmetricLinearDiscretor.apply(),
        )
# dt_tr.print(short=True)
dt_tr: Trace = dt_tr.checkpoint_transform(
        Quantizer.apply(),
        # print_bf=True, print_af=True,
        # force=True,
)

from tvm.mrt.fixed_point import FixPoint, Simulator
sim_tr = dt_tr.checkpoint_transform(
        Simulator.apply(),
        # force=True,
        )
# sim_tr.log()
# sim_tr.print(short=True)

qt_tr = dt_tr.checkpoint_transform(
        FixPoint.apply(),
        # print_bf = True, print_af = True,
        # force=True,
)
qt_tr.print(short=False)

from tvm.mrt.zkml import circom, transformer, model as ZkmlModel

symbol, params = qt_tr.symbol, qt_tr.params
print(">>> Start circom gen...")
symbol, params = ZkmlModel.resize_batch(symbol, params)
#ZkmlModel.simple_raw_print(symbol, params)
symbol, params = transformer.change_name(symbol, params)
# set input as params
symbol_first = ZkmlModel.visit_first(symbol)
#print(">>> before circom gen ...", symbol_first, symbol_first.is_input(), symbol_first.is_param())
import torch
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
