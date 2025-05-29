from __future__ import annotations

import numpy as np
from dataclasses import dataclass

from mrt.mir import op
from mrt.mir.opns import *
from mrt.mir.symbol import filter_operators
from mrt.mir.attrs import PClipAttrs, RequantAttrs

from mrt.quantization.precision import *
from mrt.quantization.discrete import QuantInfo

from mrt.common.utils import number_to_bits

from .transform import Transformer

@dataclass(repr=False)
class Simulator(QuantInfo):
    def round(self, out: Transformer):
        #  data_0_5 = self.from_const_data(0.5)
        #  out = op.add(out, data_0_5)
        #  out = op.ceil(out)
        orig_dtype = out.dtype
        out = op.cast(out, dtype="int32")
        out = op.cast(out, dtype=orig_dtype)
        return out

    def __call__(self, with_clip=False, with_round=False, **kw):
        out: Transformer = self
        if self.is_input():
            """ input is the original float data, skip. """
            return out

        if self.is_param() and with_round:
            out = self.round(out)

        if self.is_op(PCLIP, RS_PCLIP, REQUANT):
            out: Simulator = self.args[0]
            if self.is_op(REQUANT):
                rescale = self.parsed.rescale
                rescale = self.from_const_data(rescale)
                out = op.mul(out, rescale)
                if with_round:
                    out = self.round(out)
            if with_clip:
                pos = self.int_max()
                # relax api from a_min/a_max to min/max
                out = op.clip(out, min=-pos, max=pos)
                # print(out)
                # sys.exit()
        return out.like(self)


@dataclass(repr=False)
class FixPoint(QuantInfo):
    def map_requant(self) -> FixPoint:
        if (self.args[0]).is_input():
            return self
        self.validate_precision()
        X: FixPoint = self.args[0]
        parsed: RequantAttrs = self.parsed

        anno_bit = WithPrecision.MAX_BIT // 2
        if X.precision > anno_bit:
            rs_bit = X.from_const_data(X.precision - anno_bit)
            X = op.right_shift(X, rs_bit).like(self)
            X.precision = anno_bit

        frac, exp = cvm_float(self.parsed.rescale, anno_bit)
        assert frac >= 1
        assert exp <= 0
        frac_sym = X.from_const_data(frac)
        out = op.mul(X, frac_sym).like(self)

        exp_sym = out.from_const_data(-exp)
        out = op.rs_pclip(out, exp_sym,
                precision=self.precision)
        # pos = self.int_max()
        # out = op.right_shift(out, exp_sym).like(self)
        # out = op.clip(out, a_min=-pos, a_max=pos).like(self)
        return out.like(self)

    def map_pclip(self) -> FixPoint:
        self.validate_precision()
        X: FixPoint = self.args[0]
        pos = self.int_max()
        out = X
        out = op.pclip(X, precision=self.precision).like(self)
        #  out = op.clip(X, a_min=-pos, a_max=pos).like(self)
        return out

    def __call__(self, **kw):
        self.dtype = "int8" if self.precision <= 8 else "int32"

        out = self
        if self.is_input():
            pass
        elif self.is_param():
            self.validate_precision()
            data = np.round(self.numpy()).astype(self.dtype)
            absmax = np.abs(data).max()
            assert absmax <= self.int_max()
            out = self.from_np_data(data)
        elif self.is_op(PCLIP):
            out = self.map_pclip()
        elif self.is_op(REQUANT):
            out = self.map_requant()
        # elif self.is_op(CONV2D, DENSE):
        #     out.attrs["out_dtype"] = "int32"

        #  if self.is_operator():
        #      out = op.cast(out, dtype="int32")
        #      out = op.cast(out, dtype="float32")

        # inames = [a.name for a in self.args]
        # tmp = op.subgraph(out, inames)
        # tmp = op.infer_type(tmp)
        # assert self.dtype == tmp.dtype, (
        #         "expected {}, but get {}, in \n{}"
        # ).format(self.dtype, tmp.dtype, tmp)
        return out.like(self, extra_attrs=self.extra_attrs)

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
