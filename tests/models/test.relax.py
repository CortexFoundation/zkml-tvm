import os
from os import path
import sys

ROOT = path.dirname(path.dirname(path.dirname(
    path.realpath(__file__))))
sys.path.insert(0, path.join(ROOT, "python"))

import tvm
from tvm import relax
from tvm import ir

import numpy as np

batch_size = 1
image_shape = (3, 224, 224) # TODO: change the model's input shape
data_shape = (batch_size,) + image_shape

# TODO: set the dataset for target model
# Example: use the torch vision imagenet dataset.
import torch
import torchvision as tv
tv.models
data_transform = tv.transforms.Compose([
    tv.transforms.Resize(256),
    tv.transforms.CenterCrop(image_shape[1]),
    tv.transforms.ToTensor(),
    tv.transforms.Normalize(
        [0.485,0.456,0.406], [0.229,0.224,0.225])
])
# data_transform = tv.models.MobileNet_V2_Weights.IMAGENET1K_V1.transforms()
# data_transform = tv.models.EfficientNet_B0_Weights.IMAGENET1K_V1.transforms()
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

# model inference context, like cpu, gpu, etc.
config = {
        "device": tvm.runtime.cuda(1),
        "target": tvm.target.cuda("3090") }

# TODO: load the model from torchvision
model_name = "resnet18"                 # passed
#   Iteration: 19 | from_expr: Top1/5: 77.50%,94.69% | sim: Top1/5: 77.50%,94.69% | clip: Top1/5: 77.81%,94.69% | round: Top1/5: 75.62%,94.06% | quantized: Top1/5: 75.31%,94.06% |
# model_name = "mobilenet_v2"             # too much shared data
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

from tvm.relax.frontend.torch import from_exported_program

with torch.no_grad():
    exported_program = torch.export.export(model, ( input_data, ))
    mod = from_exported_program(
            exported_program, keep_params_as_input=True)
mod, params = relax.frontend.detach_params(mod)
# mod.show()
mod: tvm.IRModule = mod

with open("/tmp/resnet18.log", "w") as f:
    f.write(mod.script(show_meta=True))

target = tvm.target.Target("nvidia/geforce-rtx-3090-ti")  # Change to your target device
compiler: tvm.transform.Pass = relax.get_pipeline(
        "static_shape_tuning", target=target, total_trials=0)

data, label = ds.next()

from tvm.mrt.frontend.relax import expr2symbol, symbol2expr

for (name, func) in mod.functions_items():
    name: relax.expr.GlobalVar = name
    func: relax.Function = func
    #  print(type(func.params[0]), type(func.params[1]))
    num_input = int(func.attrs.get("num_input", 1))
    func_params = {k.name_hint: v.numpy() for (k, v) in zip(
        func.params[num_input:], params[name.name_hint])}
    #  print(name, type(func_params))
    #  print({k: v.shape for k, v in func_params.items()})

    func, fparams = expr2symbol(func.body, func_params)

import numpy as np
from tvm.mrt import runtime

#  cmod = compiler(mod)
#  cmod["main"].show()
#  ex = tvm.compile(cmod, target="cuda")
#  dev = tvm.device("cuda", 0)
#  vm = relax.VirtualMachine(ex, dev)

#  gpu_data = tvm.nd.array(data.astype("float32"), dev)
#  gpu_params = [tvm.nd.array(p, dev) for p in params["main"]]

#  test_executor = runtime.create_executor(
#          mod, fparams, **config, opt_pass=compiler)
#  print(test_executor.input_info)
#  for (mp, p) in zip(test_executor.dev_params[1:], gpu_params):
#      assert mp.shape == p.shape, f"{mp.shape} vs. {p.shape}"

#  gpu_out: tvm.ir.container.Array = vm["main"](
#          gpu_data, *test_executor.dev_params[1:])
#  print(gpu_out)

#  from tvm.mrt import types
#  gpu_out = types.to_numpy(gpu_out)[0]
#  print(gpu_out.shape, label, label.shape)
#  print(np.argmax(gpu_out))
#  sys.exit()

mod = symbol2expr(func, fparams)
out: np.ndarray = runtime.infer(
        mod, fparams, data,
        opt_pass=compiler, **config)
print(out)
print(out.shape, label)
print("agrmax of out:", np.argmax(out))
#  out = runtime.create_executor(
#          mod, fparams,
#          opt_pass=compiler,
#          **config)
sys.exit()

from tvm.mrt import stats
from tvm.mrt.trace import Trace
tr = Trace.from_expr(func, fparams, model_name=model_name)
tr.bind_dataset(ds, stats.ClassificationOutput).log()

# tr.validate_accuracy(max_iter_num=20, **config)
# sys.exit()

dis_tr = tr.discrete(
        calibrate_repeats=16,
        force=True
        )
sim_tr = dis_tr.export("sim").log()
sim_clip_tr = dis_tr.export("sim-clip").log()
sim_round_tr = dis_tr.export("sim-round").log()
sim_quant_tr = dis_tr.export("sim-clip-round").log()
circom_tr = dis_tr.export("circom").log()


# fuse_tr = tr.fuse().log()
# calib_tr = fuse_tr.calibrate(
#         # force=True,
#         batch_size=16).log()

# from tvm.mrt.config import Pass
# with Pass(log_before=True, log_after=True):
#     dis_tr = calib_tr.quantize().log()

# sim_tr = dis_tr.export().log()
# sim_clip_tr = dis_tr.export(with_clip=True).log()
# sim_round_tr = dis_tr.export(with_round=True).log()
# sim_quant_tr = dis_tr.export(
#         with_clip=True, with_round=True).log()

# circom_tr = dis_tr.export(force=True, use_simulator=False).log()

tr.validate_accuracy(
        sim_tr,
        sim_clip_tr,
        sim_round_tr,
        sim_quant_tr,
        max_iter_num=200,
        **config)
sys.exit()

