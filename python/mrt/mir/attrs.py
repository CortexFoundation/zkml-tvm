from functools import wraps
from dataclasses import dataclass, fields, Field

import tvm

from mrt.common.types import *
# from .types import *
from .op import *

@dataclass
class _BaseAttrs:
    @classmethod
    def parse(cls, attrs: AttrsT):
        ftypes = {f.name: f.type for f in fields(cls)}
        try:
            data = {k: attrs[k] for k in ftypes}
            #  for k, v in data.items():
            #      assert isinstance(v, ftypes[k]), (
            #              "{}({}) vs. {} in {}"
            #              ).format(type(v), v, ftypes[k], cls.__name__)
        except Exception as e:
            print(f"Attr:{get_class_name(cls)} parsed error,",
                  f"expect: {list(ftypes.keys())},",
                  f"get {list(attrs.keys())}")
            raise e
        return cls(**data)

_ALL_ATTRS = {}

def register_attrs(op_name):
    def _wrapper(cls):
        _ALL_ATTRS[op_name] = cls
        return cls
    return _wrapper

def parse_attrs(op_name, attrs) -> _BaseAttrs:
    if op_name in _ALL_ATTRS:
        return _ALL_ATTRS[op_name].parse(attrs)
    return _BaseAttrs.parse(attrs)

def _format_as_tuple(attrs: AttrsT, *keys):
    for k in keys:
        if k not in attrs:
            continue
        val = attrs[k]
        if not isinstance(val,
                (list, tuple, tvm.ir.container.Array)):
            attrs[k] = [ val, val ]
    return attrs

@dataclass
@register_attrs(CLIP)
class ClipAttrs(_BaseAttrs):
    min: float
    max: float

@dataclass
@register_attrs(PCLIP)
class PClipAttrs(_BaseAttrs):
    precision: int
@dataclass
@register_attrs(RS_PCLIP)
class RSPClipAttrs(PClipAttrs):
    # shiftbit: int
    precision: int

@dataclass
@register_attrs(RESHAPE)
class ReshapeAttrs(_BaseAttrs):
    shape: list

@dataclass
@register_attrs(REQUANT)
class RequantAttrs(PClipAttrs):
    rescale: float
    precision: int

@dataclass
@register_attrs(ADAPTIVE_AVG_POOL2D)
class AdaptiveAvgPool2DAttrs(_BaseAttrs):
    layout: str = "NCHW"
    out_layout: typing.Optional[str] = None
    output_size: typing.Optional[typing.Tuple[int, int]] = None

@dataclass
@register_attrs(AVG_POOL2D)
class AvgPool2DAttrs(_BaseAttrs):
    pool_size: typing.Union[int, typing.Tuple[int]]
    strides: typing.Tuple[int]
    dilation: typing.Union[int, typing.Tuple[int]]
    padding: typing.Tuple[int]
    layout: str
    ceil_mode: bool
    count_include_pad: bool

    @classmethod
    def parse(cls, attrs: AttrsT):
        attrs.setdefault("layout", "NCHW")
        attrs.setdefault("ceil_mode", False)
        attrs.setdefault("count_include_pad", True)
        return super().parse(attrs)

@dataclass
@register_attrs(TUPLE_GET_ITEM)
class TupleGetItemAttrs(_BaseAttrs):
    index: int

@dataclass
@register_attrs(DENSE)
class DenseAttrs(_BaseAttrs):
    units: int
    out_dtype: str

@dataclass
@register_attrs(CONV2D)
class Conv2DAttrs(_BaseAttrs):
    """ Reference to https://tvm.apache.org/docs/reference/api/python/relay/nn.html#tvm.relay.nn.conv2d

    strides (Optional[int, Tuple[int]]) – The strides of convolution.
    padding (Optional[int, Tuple[int]]) – The padding of convolution on both sides of inputs before convolution.
    dilation (Optional[int, Tuple[int]]) – Specifies the dilation rate to be used for dilated convolution.
    groups (Optional[int]) – Number of groups for grouped convolution.
    channels (Optional[int]) – Number of output channels of this convolution.
    kernel_size (Optional[int, Tuple[int]]) – The spatial of the convolution kernel.
    data_layout (Optional[str]) – Layout of the input.
    kernel_layout (Optional[str]) – Layout of the weight.
    out_layout (Optional[str]) – Layout of the output, by default, out_layout is the same as data_layout
    out_dtype (Optional[str]) – Specifies the output data type for mixed precision conv2d.
    """
    strides: typing.Tuple[int, int]
    padding: typing.Tuple[int, int, int, int]
    dilation: typing.Tuple[int, int]
    groups: int
    #  channels: int
    #  kernel_size: typing.Tuple[int, int]
    data_layout: str
    kernel_layout: str
    out_layout: str
    out_dtype: str

    @classmethod
    def parse(cls, attrs: AttrsT):
        attrs = _format_as_tuple(attrs,
                "strides", "dilation",
                "kernel_size", "padding")
        attrs.setdefault("kernel_layout", "OIHW")
        attrs.setdefault("data_layout", "NCHW")
        #  attrs.setdefault("channels", None)
        attrs.setdefault("out_layout", "")
        attrs.setdefault("out_dtype", "")
        return super().parse(attrs)

@dataclass
@register_attrs(BATCH_NORM)
class BatchNormAttrs(_BaseAttrs):
    axis: int = 1
    epsilon: float = 1e-5
    center: bool = True
    scale: bool = True

@dataclass
@register_attrs(LEAKY_RELU)
class LeakyReLUAttrs(_BaseAttrs):
    alpha: float

@dataclass
@register_attrs(MEAN)
class MeanAttrs(_BaseAttrs):
    axis: typing.Optional[int]
    keepdims: bool
    exclude: bool


