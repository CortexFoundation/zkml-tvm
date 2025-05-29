from __future__ import annotations

import typing
from dataclasses import dataclass, field, make_dataclass

import numpy as np

from .symbol import *
from .opns import *
from .scale import Scaler, Precision
from .transform import Transformer, Pass

def count_to_bits(count: int):
    """
    # get_bit_cnt (mrt) should be consistent with
    # GetReduceSumBit (cvm-runtime)

    """
    prec = 0
    while count != 0:
        prec += 1
        count >>= 1
    return prec

AnnotateT = typing.List[Scaler]

@dataclass(repr=False)
class Annotate(Pass):
    """ set symbol arguments precision and scale. """
    @property
    def arg_scalers(self) -> AnnotateT:
        return [a.scaler.copy() for a in self.args]

    def set_arg_precision(self, prec: int) -> AnnotateT:
        return [s.set(1, prec) for s in self.arg_scalers]

    def identity(self) -> Annotate:
        return self.set_arg_precision(0)

    def first_like_scale(self) -> Annotate:
        """ make other args like the first one.

            Assume that the first argument is welled
            calculated in scale and precision.
        """
        scalers: AnnotateT = self.arg_scalers
        assert scalers[0].defined()
        scale = scalers[0].scale
        return [ s.examine(scale=scale) for s in scalers ]

Annotate.test(VAR)(lambda x: [])
Annotate.test(CONV2D, DENSE)(Annotate.set_arg_precision, 8)
Annotate.test(BIAS_ADD)(Annotate.identity)
Annotate.test(MUL, ADD, SUB)(Annotate.set_arg_precision, 16)
Annotate.test(TUPLE, TUPLE_GET_ITEM)(Annotate.identity)
Annotate.test(RELU, MAX_POOL2D)(Annotate.identity)
Annotate.test(SQUEEZE, RESHAPE)(Annotate.identity)

@dataclass(repr=False)
class InferPrecision(Pass):
    """ infered precision as expected. """
    @property
    def arg_precisions(self):
        assert all([a.precision.defined() for a in self.args])
        return [a.precision for a in self.args]

    def _infer_index(self, index) -> Precision:
        return self.arg_precisions[index]

    def _infer_max(self) -> Precision:
        return max(self.arg_precisions)

    def _infer_mul(self) -> Precision:
        return sum(self.arg_precisions)

    def _first_like(self) -> Precision:
        return self._infer_index(0)

    def _infer_add(self) -> Precision:
        return self._infer_max() + 1

    def _infer_nn(self) -> Precision:
        W = self.args[1]
        add_count = np.product(W.shape[1:])
        add_bits = count_to_bits(add_count)
        return self._infer_mul() + add_bits


InferPrecision.test(VAR)(lambda x: Precision())
InferPrecision.test(TUPLE)(InferPrecision._infer_max)
@InferPrecision.test(TUPLE_GET_ITEM)
def _infer_tuple_get_item(self: InferPrecision):
    return self._infer_index(self.parsed.index)
InferPrecision.test(CONV2D, DENSE)(InferPrecision._infer_nn)
# InferPrecision.test(BIAS_ADD)(InferPrecision._infer_add)
InferPrecision.test(RELU, MAX_POOL2D)(InferPrecision._first_like)
InferPrecision.test(SQUEEZE, RESHAPE)(InferPrecision._first_like)
InferPrecision.test(ADD, SUB)(InferPrecision._infer_add)
InferPrecision.test(MUL)(InferPrecision._infer_mul)
# InferPrecision.test(REQUANT)(InferPrecision._first_like)
