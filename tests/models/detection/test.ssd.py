import os
import sys
ROOT = os.getcwd()
sys.path.insert(0, os.path.join(ROOT, "python"))

import numpy as np
from PIL import Image
import torch

import tvm
from tvm import ir
from tvm.mrt.utils import *
from tvm.mrt import runtime
from tvm.mrt import stats, dataset
from tvm.mrt import utils

batch_size = 16
image_shape = (3, 224, 224)
data_shape = (batch_size,) + image_shape

def dict_to_tuple(out_dict):
    if "masks" in out_dict.keys():
        return (out_dict["boxes"], out_dict["scores"], out_dict["labels"], out_dict["masks"])
    return (out_dict["boxes"], out_dict["scores"], out_dict["labels"])

class TraceWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, inp):
        out = self.model(inp)
        return dict_to_tuple(out[0])

def load_model_from_torch() -> (ir.IRModule, ParametersT):
    import torchvision
    model = torchvision.models.detection.ssd300_vgg16(weights=torchvision.models.detection.SSD300_VGG16_Weights.DEFAULT)
    model = TraceWrapper(model)
    model = model.eval()
    data_transform = torchvision.models.detection.SSD300_VGG16_Weights.COCO_V1.transforms()
    input_data = data_transform(torch.randn(data_shape))
    # script_module = torch.jit.script(model, example_inputs=[input_data]) # .eval()
    script_module = torch.jit.trace(model, [input_data])
    # with open("/tmp/script.txt", "w") as f:
    #     f.write(script_module.model.code)
    #     f.write("\n\n\n")
    #     f.write(script_module.code)
    return tvm.relay.frontend.from_pytorch(
            script_module.eval(), [ ("input", data_shape) ])

# mod, params = load_model_from_torch()
# mod: tvm.IRModule = mod
# func: tvm.relay.function.Function = mod["main"]
# expr: ir.RelayExpr = func.body

# with open("/tmp/expr.txt", "w") as f:
#     f.write(expr.astext())

from tvm.mrt.trace import Trace

# tr = Trace.from_expr(expr, params, model_name="ssd300_vgg16")
# tr.dump()
# tr.log()

tr = Trace.load("./data/ssd300_vgg16/from_expr.trace")

from tvm.mrt import config, optype, helper

with config.Pass("fuse", log_before=True, log_after=True):
    tr = tr.fuse().log()

t = tr.subgraph(inames=["%822"], onames=["%823"])
symbol = optype.infer(t.symbol)
helper.format_print(symbol, t.params)
t = Trace(t.model, "type_infer", symbol, t.params)
out = t.eval(np.random.randint(0, 2, size=(8732,)))
print(out.shape)
sys.exit()

with config.Pass(
        name = "type_infer",
        log_before = True, log_after = True):
    symbol = optype.infer(tr.symbol)
tr = Trace(tr.model_name, tr.tr_name, symbol, tr.params)

# TODO: add infer shape & dtype
# TODO: add subgraph for fuse

sys.exit()

from tvm.mrt import fuse
from tvm.mrt import op
fuse_tr = tr.checkpoint_transform(
        fuse.FuseTupleGetItem.apply(),
        fuse.FuseBatchNorm.apply(),
        fuse.FuseAvgPool2D.apply(),
        fuse.FuseNaiveSoftmax.apply(),
        fuse.FuseDropout.apply(),
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
