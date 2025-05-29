import typing

import tvm
from tvm import relax, ir, tir
import numpy as np

from mrt.common.types import *

# OpOutputT = typing.Union[tvm.nd.NDArray, list]
# OpNumpyT = typing.Union[np.ndarray, list]
# ParametersT = typing.Dict[str, OpOutputT]
# AttrsT = typing.Dict[str, typing.Any]

# ShapeT = typing.List[int]
# """ shape type, list of int, such as [1, 3, 34, 34]. """
# DTypeT = str

# DataLabelT = typing.Tuple[np.ndarray, typing.Any]
# """ a (data, label) representation. """

TVMModule = tvm.IRModule
TVMFunction = relax.Function
TVMExpr = relax.Expr

DefConvertFunc = typing.Callable[[typing.Any], typing.Any]

def to_numpy(data: OpOutputT) -> OpNumpyT:
    return convert_to_py(
            data, log_default_type=False,
            default_convert_func=lambda x:x.numpy())
    #  if isinstance(data, (list, ir.container.Array)):
    #      return [d.numpy() for d in data]
    #  return data.numpy()
    # return [d.numpy() for d in data] \
            #  if isinstance(data, (list, tvm.ir.container.Array)) else data.numpy()

def to_ndarray(data: OpNumpyT) -> OpOutputT:
    return [tvm.nd.array(d) for d in data] \
            if isinstance(data, list) else tvm.nd.array(data)

def convert_to_py(value,
                  log_default_type: bool = False,
                  default_convert_func: DefConvertFunc = lambda x:x,
                  support_numpy: bool = True):
    # need to pass the kwargs iterately.
    kwargs = {
            "log_default_type": log_default_type,
            "default_convert_func": default_convert_func,
            "support_numpy": support_numpy,
    }
    """ TVM type to instrinsic py type. """
    if isinstance(value, relax.expr.ShapeExpr):
        return convert_to_py(value.values, **kwargs)
    elif isinstance(value, relax.expr.PrimValue):
        return convert_to_py(value.value, **kwargs)
    elif isinstance(value, (list, ir.container.Array)):
        return [ convert_to_py(v, **kwargs) for v in value ]
    elif isinstance(value, (
        tir.expr.IntImm, tir.expr.FloatImm, tir.expr.StringImm)):
        return value.value
    elif isinstance(value, relax.expr.Constant):
        return value.data.numpy()
    elif isinstance(value, (str, int, float)):
        return value
    elif value is None:
        return value
    elif isinstance(value, tir.expr.Var):
        return value.name
    elif support_numpy and isinstance(value, np.ndarray):
        return value
    elif log_default_type:
        print(">>> unknown type:", type(value))
    return default_convert_func(value)

def get_struct_info(info: relax.StructInfo, key):
    if isinstance(info, relax.struct_info.TupleStructInfo):
        return [get_struct_info(f, key) for f in info.fields]
    #  return getattr(info, key)
    val = convert_to_py(getattr(info, key))
    return val

def get_struct_shape(info: relax.StructInfo):
    return get_struct_info(info, "shape")

def get_struct_dtype(info: relax.StructInfo):
    return get_struct_info(info, "dtype")
