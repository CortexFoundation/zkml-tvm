import os
import sys

ROOT = os.getcwd()
sys.path.insert(0, os.path.join(ROOT, "python"))

import tvm
from tvm import relay, ir
import numpy as np

batch_size = 16
image_shape = (3, 224, 224)
data_shape = (batch_size,) + image_shape

import torch
import torchvision as tv
data_transform = tv.transforms.Compose([
    tv.transforms.Resize(256),
    tv.transforms.CenterCrop(image_shape[1]),
    tv.transforms.ToTensor(),
    tv.transforms.Normalize(
        [0.485,0.456,0.406], [0.229,0.224,0.225])
])
# data_transform = tv.models.MobileNet_V2_Weights.IMAGENET1K_V1.transforms()
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
config = {"device": tvm.runtime.cuda(1),
        "target": tvm.target.Target("cuda -arch=sm_86") }

model_name = "resnet18"                 # passed
#   Iteration: 19 | from_expr: Top1/5: 77.50%,94.69% | sim: Top1/5: 77.50%,94.69% | clip: Top1/5: 77.81%,94.69% | round: Top1/5: 75.62%,94.06% | quantized: Top1/5: 75.31%,94.06% |
#model_name = "mobilenet_v2"             # too much shared data
#model_name = "efficientnet_b2"          # too much shared data
#model_name = "alexnet"                  # passed
#   Iteration: 19 | from_expr: Top1/5: 66.88%,88.44% | sim: Top1/5: 66.88%,88.44% | clip: Top1/5: 67.19%,88.44% | round: Top1/5: 66.56%,89.06% | quantized: Top1/5: 66.56%,89.06% |
#model_name = "densenet121"              # pass
#   Iteration:  19 | from_expr: Top1/5: 83.75%,96.56% | sim: Top1/5: 83.75%,96.56% | clip: Top1/5: 83.75%,96.88% | round: Top1/5: 51.25%,81.25% | quantized: Top1/5: 50.31%,81.25% |
#model_name = "squeezenet1_0"            # pass
#   Iteration:  19 | from_expr: Top1/5: 70.31%,91.56% | sim: Top1/5: 70.31%,91.56% | clip: Top1/5: 70.31%,91.56% | round: Top1/5: 59.06%,85.94% | quantized: Top1/5: 59.38%,85.94% |
#model_name = "vgg11"                    # pass
#   Iteration:  19 | from_expr: Top1/5: 79.06%,95.00% | sim: Top1/5: 79.06%,95.00% | clip: Top1/5: 79.06%,95.00% | round: Top1/5: 77.50%,95.94% | quantized: Top1/5: 77.50%,95.94% |
#model_name = "shufflenet_v2_x0_5"       # pass
#   Iteration:  19 | from_expr: Top1/5: 73.12%,90.00% | sim: Top1/5: 73.12%,90.00% | clip: Top1/5: 73.75%,89.69% | round: Top1/5: 0.94%,4.38% | quantized: Top1/5: 0.94%,4.38% |
model = getattr(tv.models, model_name)(weights="DEFAULT")

model = model.eval()
input_data = torch.randn(data_shape)
script_module = torch.jit.trace(model, [input_data]).eval()
mod, params = relay.frontend.from_pytorch(
        script_module, [ ("input", data_shape) ])


tune_log_file = "{}_gpu_autotuned.json".format(model_name)
target = "cuda -arch=sm_86" #"llvm -mcpu=skylake-avx512"
dev = tvm.device(str(target), 0)
def gpu_tune():
    print("gpu fine tuning...")
    import tvm.auto_scheduler as auto_scheduler
    from tvm.autotvm.tuner import XGBTuner
    from tvm import autotvm
    number = 10
    repeat = 1
    min_repeat_ms = 0  # 调优 CPU 时设置为 0
    timeout = 10  # 秒
    
    # 创建 TVM 运行器
    # 首先从 模型中提取任务
    print("Extract tasks...")
    tasks, task_weights = tvm.auto_scheduler.extract_tasks(mod["main"], params, target)
    for idx, task in enumerate(tasks):
        print("========== Task %d  (workload key: %s) ==========" % (idx, task.workload_key))
        print(task.compute_dag)
    
    # 按顺序调优提取的任务
    def run_tuning():
        print("Begin tuning...")
        measure_ctx = tvm.auto_scheduler.LocalRPCMeasureContext(repeat=1, min_repeat_ms=200, timeout=10)
        tuner = tvm.auto_scheduler.TaskScheduler(tasks, task_weights)
        tune_option = tvm.auto_scheduler.TuningOptions(
            num_measure_trials=64,  # may need to change with the model ## change this to 20000 to achieve the best performance
            runner=measure_ctx.runner,
            measure_callbacks=[
                tvm.auto_scheduler.RecordToFile(tune_log_file)
            ],
        )
        tuner.tune(tune_option)
    run_tuning()
    
    print("### fine-tuned finished")
    with auto_scheduler.ApplyHistoryBest(tune_log_file):
        with tvm.transform.PassContext(opt_level=3, config={"relay.backend.use_auto_scheduler": True}):
            lib = tvm.relay.build(mod, target=target, params=params)
    
    module = tvm.contrib.graph_executor.GraphModule(lib["default"](dev))
    from tvm.mrt.runtime import set_tuned_module
    set_tuned_module(module)

gpu_tune()
# if tuned:
import tvm.auto_scheduler as auto_scheduler
with auto_scheduler.ApplyHistoryBest(tune_log_file):
    with tvm.transform.PassContext(opt_level=3, config={"relay.backend.use_auto_scheduler": True}):
        lib = tvm.relay.build(mod, target=target, params=params)
module = tvm.contrib.graph_executor.GraphModule(lib["default"](dev))
from tvm.mrt.runtime import set_tuned_module
set_tuned_module(module)
# tuned end

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
