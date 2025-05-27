import typing
import numpy as np

from .symbol import *
from .opns import *
from .frontend.expr import symbol2expr
from .transform import WithParameters
from . import op, runtime, types

def run(sym: WithParameters,
        args_data: typing.List[OpOutputT],
        op_single: bool = True,
        **kwargs) -> OpOutputT:
    assert op.is_operator(sym), sym

    if sym.is_op(TUPLE_GET_ITEM):
        return args_data[0][sym.parsed.index]
    elif sym.is_op(REQUANT):
        return tvm.nd.array(sym.parsed.rescale * args_data[0].numpy())
    elif sym.is_op(ARANGE):
        args = [a.numpy().item() for a in args_data]
        return tvm.nd.array(np.arange(*args, **sym.attrs))
    elif sym.is_op("stack"):
        args = [types.to_numpy(a) for a in args_data]
        return types.to_ndarray(np.stack(*args, **sym.attrs))
    elif sym.is_op("meshgrid"):
        args = [types.to_numpy(a) for a in args_data[0]]
        return types.to_ndarray(np.meshgrid(*args, **sym.attrs))
    elif sym.is_op(ZEROS_LIKE):
        return tvm.nd.array(np.zeros(sym.shape, sym.dtype))
    elif sym.is_op(ONES_LIKE):
        return tvm.nd.array(np.ones(sym.shape, sym.dtype))

    if op_single:
        sym = sym.copy(args=[
            op.as_variable(a, d.numpy().shape, d.numpy().dtype.name) \
                    for a, d in zip(sym.args, args_data)])
    expr = symbol2expr(sym)
    params = { c.name: args_data[i] for i, c in enumerate(sym.args) }
    return runtime.infer(expr, params, **kwargs)

def _mx_executor(sym: Symbol, inputs):
    from mxnet import nd

    op_name = sym.op

    if op_name == "nn.dense":
        X, W = inputs
        X = nd.expand_dims(X, axis=0)
        num_hidden = W.shape[0]
        out = nd.FullyConnected(X, W,
                nd.zeros((num_hidden,)),
                num_hidden=num_hidden)
        return out[0]
    elif op_name == "nn.conv2d":
        X, W = inputs
        X = nd.expand_dims(X, axis=0)
        kernel = W.shape[-2:]
        num_filter = W.shape[0]
        out = nd.Convolution(
                X, W, nd.zeros((num_filter,)),
                kernel=kernel,
                num_filter=num_filter,
                )
        return out[0]

    raise NotImplementedError(sym.info())

def mx_executor(sym: Symbol, inputs):
    inputs = [nd.array(inp) for inp in inputs]
    return _mx_executor(sym, inputs).asnumpy()

def _np_executor(sym: Symbol, inputs):
    op_name = sym.op_name
    if op_name == "cast":
        dtype_map = {
            "int32": "i4",
            "uint8": "i4",
        }
        return np.ndarray.astype(inputs[0],
                dtype=dtype_map[sym.dtype])
    elif op_name == "flatten":
        return inputs[0].flatten()
    elif op_name == "image.resize2d":
        assert sym.attrs["method"] == "nearest_neighbor"
        shape = sym.attrs["shape"]
        out = np.zeros(shape)
        for i in range(shape[0]):
            out[i] = cv2.resize(inputs[0][i],
                    dsize=sym.attrs["size"],
                    interpolation=cv2.INTER_NEAREST)
        return out
    elif op_name == "nn.pad_scalar":
        shape = sym.attrs["shape"]
        width = sym.attrs["pad_width"][-len(shape):]
        value = sym.attrs.get("scalar", None)
        if value is None:
            value = sym.attrs["pad_value"]
        return np.pad(inputs[0],
                pad_width=width,
                constant_values=value)
    elif op_name == "subtract":
        return inputs[0] - inputs[1]
    elif op_name == "add":
        return inputs[0] + inputs[1]
    elif op_name == "mul_scalar":
        return inputs[0] * sym.attrs["scalar"]
    elif op_name == "add_scalar":
        return inputs[0] + sym.attrs["scalar"]
    elif op_name == "subtract_scalar":
        return inputs[0] - sym.attrs["scalar"]
    elif op_name == "clip":
        return inputs[0].clip(
                sym.attrs["a_min"], sym.attrs["a_max"])
    elif op_name == "right_shift":
        return inputs[0] >> sym.attrs["shift_bit"]
    elif op_name == "sum":
        max_axis = len(inputs[0].shape)
        axes = [a+max_axis if a < 0 else a \
                for a in sym.attrs["axis"]]
        assert all([(a > 0 and a < max_axis) for a in axes])
        out = inputs[0]
        for axis in reversed(sorted(axes)):
            out = np.ndarray.sum(out, axis=axis)
        #  print(list(reversed(sorted(axes))), out.shape)
        assert list(out.shape) == list(sym.shape), sym.info()
        return out

        #  assert len(axes) == 1, sym.info()
        #  return np.ndarray.sum(inputs[0], axis=axes[0])
    elif op_name == "multiply":
        return inputs[0] * inputs[1]
    elif op_name == "reshape":
        return inputs[0].reshape(sym.shape)
    elif op_name == "expand_dims":
        out = inputs[0]
        for _ in range(sym.attrs["num_newaxis"]):
            out = np.expand_dims(out, axis=sym.attrs["axis"])
        assert list(out.shape) == list(sym.shape), sym.info()
        return out
    return None

def np_executor(sym: Symbol, inputs):
    out = _np_executor(sym, inputs)
    if out is None:
        out = mx_executor(sym, inputs)
    assert list(sym.shape) == list(out.shape)
    # out = out.astype("i8")
    #  print(sym.name, out.flatten()[:5])
    return out
