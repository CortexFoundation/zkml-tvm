import os
from os import path
import sys

ROOT = os.getcwd()
sys.path.insert(0, path.join(ROOT, "python"))

import tvm
from tvm import relay, ir

import numpy as np

batch_size = 1
# batch_size = 16
image_shape = (3, 512, 512)
data_shape = (batch_size,) + image_shape

import torch
import torchvision as tv
data_transform = tv.transforms.Compose([
    tv.transforms.Resize(image_shape[-1]),
    tv.transforms.CenterCrop(image_shape[1]),
    tv.transforms.ToTensor(),
    tv.transforms.Normalize(
        [0.485,0.456,0.406], [0.229,0.224,0.225])
])

#  from gluoncv import data
#  from gluoncv.data.batchify import Tuple, Stack, Pad
#  from gluoncv.data.transforms import presets
#  data_transform = presets.ssd.SSDDefaultValTransform(*image_shape[1:])

#  dataset = data.ImageNet(train=False)
#  batchify_fn = Tuple(Stack(), Pad(pad_val=-1))
#  test_loader = data.dataloader.DataLoader(
#          dataset.transform(data_transform),
#          batch_size=batch_size,
#          shuffle=False,
#          batchify_fn=batchify_fn,
#          )
# dataset = tv.datasets.VOCDetection(
#         "~/.mxnet/datasets/voc/",
#         transform=data_transform,
#         )
dataset = tv.datasets.ImageFolder(
        '~/.mxnet/datasets/imagenet/val',
        transform=data_transform)
test_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size, # set dataset batch load
        )

#  for i, b in enumerate(test_loader):
#      if i > 3:
#          break
#      print(b[0].shape, b[1].shape)
#  sys.exit()

# use mrt wrapper to uniform api for dataset.
from tvm.mrt.dataset_torch import TorchWrapperDataset
ds = TorchWrapperDataset(test_loader)

# model inference context, like cpu, gpu, etc.
config = {
        "device": tvm.runtime.cuda(1),
        "target": tvm.target.Target("cuda -arch=sm_86")}

model_name = "mxnet_ssd_512_resnet50_v1_voc"
# model_name = "faster_rcnn_resnet50_v1b_voc"
model_name = "yolo3_darknet53_voc"
# model_name = "ssd_512_resnet50_v1_voc"

# with default params
import gluoncv
from gluoncv import model_zoo
import mxnet as mx
import json
# im_fname = gluoncv.utils.download('https://github.com/dmlc/web-data/blob/master/' +
#                           'gluoncv/detection/biking.jpg?raw=true',
#                           path='biking.jpg')
# img, orig_img = gluoncv.data.transforms.presets.rcnn.load_test(im_fname)
# # data_shape = img.shape
# print(img.shape)

model: mx.gluon.HybridBlock = model_zoo.get_model(model_name, pretrained=True)
model.hybridize()
img, label = ds.next()
img = mx.nd.array(img)
print(img.shape)
model(img)
# model.export("/tmp/model.json")
#input_data = np.random.randn(*data_shape)
#input_data = torch.randn(data_shape)

mod, params = relay.frontend.from_mxnet(model, {"data": data_shape})

# MRT Procedure
mod: tvm.IRModule = mod
func: relay.function.Function = mod["main"]
expr: ir.RelayExpr = func.body

#  with open("/tmp/expr.txt", "w") as f:
#      f.write(expr.astext(show_meta_data=False))

from tvm.mrt import stats, calibrate, opns
from tvm.mrt.trace import Trace
from tvm.mrt.config import Pass

tr = Trace.from_expr(expr, params, model_name=model_name)
tr.bind_dataset(ds, stats.ClassificationOutput).log()


from tvm.mrt import stats
from torchmetrics.detection import MeanAveragePrecision
class TorchStatistics(stats.Statistics):
    def __init__(self):
        self.map = MeanAveragePrecision()

    def reset(self):
        self.map = MeanAveragePrecision()

    def merge(self, dl):
        (pred_label, score, bbox), label = dl
        preds = [ dict(
            boxes=torch.from_numpy(bbox.numpy()),
            scores=torch.from_numpy(score.numpy()),
            labels=torch.from_numpy(pred_label.numpy()),
            ) ]
        target = [ dict(
            boxes=torch.from_numpy(label[0].numpy()),
            labels=torch.from_numpy(label[1].numpy()),
            ) ]
        self.map.update()

dis_tr = tr.discrete(force=True)
sim_tr = dis_tr.export("sim").log()
sim_clip_tr = dis_tr.export("sim-clip").log()
sim_round_tr = dis_tr.export("sim-round").log()
sim_quant_tr = dis_tr.export("sim-clip-round").log()
circom_tr = dis_tr.export("circom").log()

"""tr.validate_accuracy(
        sim_tr,
        sim_clip_tr,
        sim_round_tr,
        sim_quant_tr,
        max_iter_num=20,
        **config)
sys.exit()"""

circom_tr.print()

from tvm.mrt.zkml import circom, transformer, model as ZkmlModel
symbol, params = circom_tr.symbol, circom_tr.params
print(">>> Start circom gen...")
symbol, params = ZkmlModel.resize_batch(symbol, params)
symbol, params = transformer.change_name(symbol, params)
symbol = transformer.change_axis(symbol)
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
