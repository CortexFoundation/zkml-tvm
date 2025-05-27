
import numpy as np

import tvm
from tvm import relay, ir, runtime

from .extool import *
from .types import ParametersT

def random_inputs(expr: ir.expr.RelayExpr,
        params: ParametersT = {}) -> ParametersT:
    input_data = {k: v for k, v in params.items()}
    for v in relay.analysis.free_vars(expr):
        if v.name_hint in params:
            continue

        print(v.name_hint, v.type_annotation)

        ty = v.type_annotation
        # ty = v.checked_type
        np_data = np.random.randn(
                *ty.concrete_shape).astype(ty.dtype)
        input_data[v.name_hint] = tvm.nd.array(np_data)
    return input_data

def set_inputs(expr: ir.expr.RelayExpr,
        params: ParametersT = {}) -> ParametersT:
    free_vars = relay.analysis.free_vars(expr)
    input_data = {k: v for k, v in params.items()}


