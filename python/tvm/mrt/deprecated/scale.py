from __future__ import annotations

import typing
import math
import numpy as np
from dataclasses import dataclass, fields

from . import op
from .opns import *
from .utils import *
from .calibrate import Calibrator
from .transform import Transformer, Pass


class Precision(int):
    def __new__(cls, *args, **kwargs):
        self = super().__new__(cls, *args, **kwargs)
        assert self >= 0
        return self

    @staticmethod
    def default():
        return Precision(0)

    def defined(self):
        return self != self.default()

class FloatScale(float):
    def __new__(cls, *args, **kwargs):
        self = super().__new__(cls, *args, **kwargs)
        assert self == -1 or self >= 0
        return self

    @staticmethod
    def default():
        return FloatScale(1)

    def defined(self):
        return self != self.default()

@dataclass(repr=False)
class _MrtBase(Symbol):
    scale: typing.Any
    precision: Precision

    @classmethod
    def default_dict(cls) -> dict:
        ftypes = {f.name: f.type for f in fields(cls)}
        scale_type = ftypes["scale"]
        return super().default_dict(
                scale=scale_type.default(),
                precision=Precision.default())

    @classmethod
    def update_dict(cls, data: dict, **kwargs) -> dict:
        ftypes = {f.name: f.type for f in fields(cls)}
        scale_type = ftypes["scale"]
        return super().update_dict(data,
                scale=scale_type(data["scale"]),
                precision=Precision(data["precision"])
                **kwargs)

@dataclass(repr=False)
class InferScale(Pass):
    @property
    def arg_scales(self):
        return [a.scale for a in self.args]

    def _infer_index(self, index):
        return self.arg_scales[index]

    def _first_like(self):
        return self._infer_index(0)

    def _uniform_scales(self):
        scales = self.arg_scales
        assert scales.count(scales[0]) == len(scales)
        return self._infer_index(0)

    def _infer_mul(self):
        return np.product(self.arg_scales)


InferScale.test(VAR)(lambda x: 1)
InferScale.test(TUPLE)(InferScale._uniform_scales)
@InferScale.test(TUPLE_GET_ITEM)
def _infer_tuple_get_item_scale(self: InferScale):
    return self._infer_index(self.parsed.index)
InferScale.test(CONV2D, DENSE)(InferScale._infer_mul)
InferScale.test(BIAS_ADD)(InferScale._uniform_scales)
InferScale.test(RELU, MAX_POOL2D)(InferScale._first_like)
InferScale.test(SQUEEZE, RESHAPE)(InferScale._first_like)
InferScale.test(ADD, SUB)(InferScale._uniform_scales)
InferScale.test(MUL)(InferScale._infer_mul)

def number_to_bits(number: float) -> int:
    """ Return the integer bits to represent number.
        precision bit: 1
        number bits:
            [ 0-0 ] => 0, skip
            [ 1-1 ] => 1, ceil(log2(i+1)) = 1
            [ 2-3 ] => 2, ceil(log2(i+1)) = 2
            [ 4-7 ] => 3, ceil(log2(i+1)) = 3
            ...

        return 1 + ceil(log2(number + 1))

        note: consider the abs round int for number.
    """
    number = math.fabs(number)
    number = math.floor(number + 0.5)
    return 1 + math.ceil(math.log2(number + 1))


@dataclass(repr=False)
class Scaler(Transformer, _MrtBase):
    """ scale's priority is larger than precision.

        Once the scale is defined, scaler should consider
            process input into specified scale.
        Once the precision is defined, scaler should check
            whether precision matchs. If not, requantize it.
    """
    data: typing.Any

    @classmethod
    def base(cls, sym: Calibrator, **kwargs):
        assert isinstance(sym, Calibrator)
        data = cls.gen_data(sym.np_data)
        return cls.from_dict(sym.to_dict(**kwargs), data=data)

    @classmethod
    def gen_data(cls, raw_data: np.ndarray):
        raise NotImplementedError()

    def defined(self):
        return self.precision.defined() or self.scale.defined()

    def examined(self):
        checked = self.copy().examine(self.scale, self.precision)
        return self.hash() == checked.hash()

    def rescale(self, base: Scaler, target: Symbol) -> Symbol:
        """ rescale target symbol based on old scaler. """
        raise NotImplementedError()

    def _set_data(self, scale, prec) -> Scaler:
        m = { f.name: f.type for f in fields(cls) }
        scale_type = ftypes["scale"]
        self.scale = scale_type(scale)
        self.precision = Precision(prec)
        return self

    def set(self, scale, prec) -> Scaler:
        return self.set(scale, prec)

    def examine(self, scale, prec) -> Scaler:
        raise NotImplementedError()

    def hash(self) -> int:
        return hash("{}{}".format(self.scale, self.precision))

    def __call__(self, *args, **kw) -> Scaler:
        """ scaler call need to override. """
        raise NotImplementedError()

@dataclass(repr=False)
class SymmetricMinMaxScaler(Scaler):
    data: float
    """ threshold for calib data. """
    scale: FloatScale
    """ scale is more precise than precision. """

    @classmethod
    def gen_data(cls, raw_data: np.ndarray):
        return max([np.abs(d).max() for d in raw_data])

    def rescale(self,
            base: SymmetricMinMaxScaler,
            target: Transformer) -> Transformer:
        if not self.scale.defined():
            # specify output precision
            assert self.precision.defined()

            # base should be well examined or not defined.
            assert base.examined() or (not base.defined())
            if base.examined() and self.precision >= base.precision:
                return target

        self.examine(prec=self.precision)
        rescale = self.scale / base.scale

        if target.is_param():
            out: Transformer = target.copy(name=N.n(self.name))
            out.update_data(target.numpy() * rescale)

            # params out of precision will be cliped
            #   in cvm-runtime.
            check = self.gen_data(out.numpy())
            checked_bit = number_to_bits(check)
            assert checked_bit <= self.precision
            print(checked_bit, self.precision,
                    out.scaler.precision)
        else:
            out = op.requant(target,
                    rescale=rescale,
                    output_precision=self.precision)
        return out

    def examine(self, scale = 1, prec = Precision()):
        assert self.data != 0
        prec = Precision(prec)
        if scale != 1:
            real_max = self.data * self.scale
            prec = number_to_bits(real_max)
        elif prec.defined():
            prec_max = 2 ** (self.precision - 1) - 1
            scale = prec_max / self.data
        return self._set_data(scale, prec)


def cvm_float(number, bits=24):
    """ Recalculate the float value within the given range of bits.

        Parameters
        __________
        number : float
            The input float value.
        bits : int
            The target bits to represent the value.

        Returns
        _______
        ret : tuple
            The recalculated float value with its corresponding bits to be shifted.
    """
    alpha = max((2 ** (bits - 1)) - 1, 1)
    bits -= 1
    assert number >= 0
    if number == 0:
        return 0, 0
    exp = 0
    while (number >= 1):
        number /= 2
        exp += 1
    while (number < 1):
        number *= 2
        exp -= 1
    while (bits > 1):
        if (int(number) == number):
            break
        number *= 2
        exp -= 1
        bits -= 1
    frac, sb = round(number), exp
    return min(frac, alpha), sb
