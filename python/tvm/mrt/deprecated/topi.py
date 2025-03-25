import typing
from functools import wraps

import numpy as np

import tvm
from tvm.ir import RelayExpr
from tvm.topi import testing

from .extool import *

TOPI_REGS = {}

DataType = typing.List[np.ndarray]

def register_topi(op_name):
    def _wrapper(f):
        TOPI_REGS[op_name] = f
        return f
    return _wrapper

@register_topi("nn.conv2d")
def run_conv2d(data: DataType, attrs: AttrsT):
    dw_np = topi.testing.dilate_python()
    return testing.conv2d_nchw_python(*data, **attrs)

@register_topi("nn.batchnorm")
def run_batchnorm(data: DataType, attrs: AttrsT):
    return testing.batch_norm(*data, **attrs)

# def execute(op_name: str, attrs: AttrsT, data: DataType) -> np.ndarray:
#     eval("relay." + op_name)(op_name, )
#     return TOPI_REGS[op_name](data, attrs)
