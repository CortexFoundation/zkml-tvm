import os
from os import path
import sys

ROOT = os.getcwd()
sys.path.insert(0, path.join(ROOT, "python"))

import tvm
from tvm import relay, ir
from tvm.relay import testing
from tvm.mrt.utils import *

from tvm.mrt import runtime
from tvm.mrt import stats, dataset
from tvm.mrt import utils

import numpy as np

batch_size = 1
image_shape = (3, 224, 224) # TODO: change the model's input shape
data_shape = (batch_size,) + image_shape

# TODO: set the dataset for target model
# Example: use the torch vision imagenet dataset.
import torch
import torchvision as tv
# data_transform = tv.transforms.Compose([
#     tv.transforms.Resize(256),
#     tv.transforms.CenterCrop(image_shape[1]),
#     tv.transforms.ToTensor(),
#     tv.transforms.Normalize(
#         [0.485,0.456,0.406], [0.229,0.224,0.225])
# ])
data_transform = tv.models.MobileNet_V2_Weights.IMAGENET1K_V1.transforms()
dataset = tv.datasets.ImageFolder(
        '~/.mxnet/datasets/imagenet/val',
        transform=data_transform)
test_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size, # set dataset batch load
        )

# use mrt wrapper to uniform api for dataset.
from tvm.mrt.dataset_torch import TorchWrapperDataset
ds = TorchWrapperDataset(test_loader)
# prepare data for calibrate pass
data, _ = ds.next()

# model inference context, like cpu, gpu, etc.
config = {
        "device": tvm.runtime.cuda(1),
        "target": tvm.target.cuda() }

# TODO: load the model from torchvision
# model_name = "resnet18"                 # passed
#   Iteration: 19 | from_expr: Top1/5: 77.50%,94.69% | sim: Top1/5: 77.50%,94.69% | clip: Top1/5: 77.81%,94.69% | round: Top1/5: 75.62%,94.06% | quantized: Top1/5: 75.31%,94.06% |
model_name = "mobilenet_v2"             # too much shared data
# model_name = "efficientnet_b2"          # too much shared data
# model_name = "alexnet"                  # passed
#   Iteration: 19 | from_expr: Top1/5: 66.88%,88.44% | sim: Top1/5: 66.88%,88.44% | clip: Top1/5: 67.19%,88.44% | round: Top1/5: 66.56%,89.06% | quantized: Top1/5: 66.56%,89.06% |
# model_name = "densenet121"              # pass
#   Iteration:  19 | from_expr: Top1/5: 83.75%,96.56% | sim: Top1/5: 83.75%,96.56% | clip: Top1/5: 83.75%,96.88% | round: Top1/5: 51.25%,81.25% | quantized: Top1/5: 50.31%,81.25% |
# model_name = "squeezenet1_0"            # pass
#   Iteration:  19 | from_expr: Top1/5: 70.31%,91.56% | sim: Top1/5: 70.31%,91.56% | clip: Top1/5: 70.31%,91.56% | round: Top1/5: 59.06%,85.94% | quantized: Top1/5: 59.38%,85.94% |
# model_name = "vgg11"                    # pass
#   Iteration:  19 | from_expr: Top1/5: 79.06%,95.00% | sim: Top1/5: 79.06%,95.00% | clip: Top1/5: 79.06%,95.00% | round: Top1/5: 77.50%,95.94% | quantized: Top1/5: 77.50%,95.94% |
# model_name = "shufflenet_v2_x0_5"       # pass
#   Iteration:  19 | from_expr: Top1/5: 73.12%,90.00% | sim: Top1/5: 73.12%,90.00% | clip: Top1/5: 73.75%,89.69% | round: Top1/5: 0.94%,4.38% | quantized: Top1/5: 0.94%,4.38% |
model = getattr(tv.models, model_name)(weights="DEFAULT")

model = model.eval()
input_data = torch.randn(data_shape)
script_module = torch.jit.trace(model, [input_data]).eval()
mod, params = relay.frontend.from_pytorch(
        script_module, [ ("input", data_shape) ])

# MRT Procedure
mod: tvm.IRModule = mod
func: relay.function.Function = mod["main"]
expr: ir.RelayExpr = func.body

from tvm.mrt.trace import Trace
tr = Trace.from_expr(expr, params, model_name=model_name)
# tr = tr.subgraph(onames=["%1"])
tr.checkpoint()
tr.log()

# TODO: test model inference accuracy, uncomment this if don't need.
runtime.multiple_validate(
        tr.populate(**config),
        dataset=ds,
        stats_type=stats.ClassificationOutput,
        max_iter_num=20,
        )
sys.exit()

from tvm.mrt import fuse
fuse_tr = tr.checkpoint_transform(
        fuse.FuseConstant.apply(),
        fuse.FuseTupleGetItem.apply(),
        fuse.FuseBatchNorm.apply(),
        fuse.FuseAvgPool2D.apply(),
        fuse.FuseDropout.apply(),
        fuse.FuseMean.apply(),
        fuse.FuseNaiveSoftmax.apply(),
        fuse.FuseConstant.apply(),
        tr_name = "fuse",
        # uncomment below line if want to recalculate mid-result
        #  force=True,
        )
fuse_tr.log()

from tvm.mrt.calibrate import Calibrator, SymmetricMinMaxSampling
calib_tr = fuse_tr.checkpoint_transform(
        Calibrator.apply(data=tvm.nd.array(data), **config),
        # print_{} parameters control whether to print the
        #   symbol before/after the transform, used to debug.
        print_bf=True, print_af=True,
        # repeats=16 // batch_size,
        #  force=True,
)
sample_tr = calib_tr.checkpoint_transform(
        SymmetricMinMaxSampling.apply(),
        #  print_af=True,
        )
# log current trace in human-readable format, cmd line will trigger
#   a output for log path.
sample_tr.log()

# MRT discrete parameters and operators
#   The output quantized-model will contain the input's requant info.
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
        Simulator.apply(with_clip=False, with_round=False),
        tr_name="sim", force=True,
        )
sim_tr.log()
clip_tr = dis_tr.checkpoint_transform(
        Simulator.apply(with_clip=True, with_round=False),
        tr_name="clip", force=True,
        )
clip_tr.log()
round_tr: Trace = dis_tr.checkpoint_transform(
        Simulator.apply(with_clip=False, with_round=True),
        tr_name="round", force=True,
        )
round_tr.log()
qt_tr: Trace = dis_tr.checkpoint_transform(
        Simulator.apply(with_clip=True, with_round=True),
        tr_name="quantized", force=True,
        )
qt_tr.log()

#  from tvm.mrt import op, opns
#  @op.filter_operators(opns.CLIP)
#  def _check_clip_inputs(sym: op.Symbol):
#      if len(sym.args) > 1:
#          print(sym)

#  op.visit(clip_tr.symbol, _check_clip_inputs)
#  sys.exit()


from PIL import Image
from tvm.contrib.download import download_testdata
def get_real_image(im_height, im_width) -> np.ndarray:
    repo_base = "https://github.com/dmlc/web-data/raw/main/tensorflow/models/InceptionV1/"
    img_name = "elephant-299.jpg"
    image_url = path.join(repo_base, img_name)
    img_path = download_testdata(image_url, img_name, module="data")
    image = Image.open(img_path).resize((im_height, im_width))
    data = np.array(image).astype("float32")
    data = np.reshape(data, (1, im_height, im_width, 3))
    data = np.transpose(data, (0, 3, 1, 2))
    data = data / 255.0
    return data

def eval_single_image():
    global tr, sim_tr, qt_tr

    data_shape = (1, ) + image_shape
    print(data_shape, data_shape)
    tr = tr.set_input_shape(data_shape)
    sim_tr = sim_tr.set_input_shape(data_shape)
    clip_tr = clip_tr.set_input_shape(data_shape)
    round_tr = round_tr.set_input_shape(data_shape)

    data = get_real_image(*image_shape[1:])
    res = tr.eval(data, **config)
    print("tr: ", res.flatten()[:10])
    sim_scale = sim_tr.symbol.extra_attrs.get("scale", 1)
    res = sim_tr.eval(data, **config) / sim_scale
    print("sim tr: ", res.flatten()[:10])
    res = clip_tr.eval(data, **config) / sim_scale
    print("clip tr: ", res.flatten()[:10])
    res = round_tr.eval(data, **config) / sim_scale
    print("round tr: ", res.flatten()[:10])

    #  res = qt_tr.eval(data, **config)
    #  print("qt tr: ", res.flatten()[:5])
    sys.exit(-1)
# TODO: eval single image for detailed debug,
#   uncomment below lines if you want to see one image output.
#  eval_single_image()
#  sys.exit(0)

# TODO: do quantized model accuracy comparation.
runtime.multiple_validate(
        tr.populate(**config),
        sim_tr.populate(**config),
        clip_tr.populate(**config),
        round_tr.populate(**config),
        qt_tr.populate(**config),
        dataset=ds,
        stats_type=stats.ClassificationOutput,
        max_iter_num=20,
)
sys.exit()

# TODO: do circom target transformation.
circom_tr: Trace = dis_tr.checkpoint_transform(
        FixPoint.apply(),
        tr_name="circom", force=True,
        )
circom_tr.log()

from tvm.mrt.zkml import circom, transformer, model as ZkmlModel
symbol, params = circom_tr.symbol, circom_tr.params
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
