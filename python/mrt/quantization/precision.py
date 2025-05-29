from __future__ import annotations

import typing
from dataclasses import dataclass

import math
import numpy as np

from mrt.mir import op
from mrt.mir.opns import *
from mrt.mir.symbol import Symbol, visit, transform

from mrt.common.utils import \
        number_to_bits, count_to_bits, bits_to_number
from mrt.common.types import ParametersT

from .transform import Transformer

__ALL__ = [ "WithPrecision",
        "InferPrecision", "QuantizedInfo",
]

@dataclass(repr=False)
class WithPrecision(Symbol):
    MAX_BIT: typing.ClassVar[int] = 32

    @classmethod
    def _validate_precision(cls, prec, msg=None):
        assert isinstance(prec, int), self.precision
        assert prec <= cls.MAX_BIT, (
            "precision:{} out of max bit:{} for \n{}"
        ).format(prec, cls.MAX_BIT, msg or str(cls))
        assert prec > 0, msg
        return True

    @property
    def precision(self) -> int:
        return self.extra_attrs.get("precision", -1)
    @precision.setter
    def precision(self, val):
        self._validate_precision(val, str(self))
        self.set_extra_attrs(precision=val)

    @property
    def precision_defined(self) -> bool:
        """ Whether current precision is well-defined. """
        return self.precision > 0 and self.precision < self.MAX_BIT

    def validate_precision(self):
        self._validate_precision(self.precision, msg=str(self))
    def int_max(self):
        return bits_to_number(self.precision)

# @dataclass(repr=False)
# class QuantizedInfo(WithPrecision):
#     @property
#     def dt_type(self) -> str:
#         """ discretization method type. """
#         return self.extra_attrs["dt_type"]
#     @property
#     def dt_info(self) -> typing.Any:
#         """ discretization information. """
#         return self.extra_attrs["dt_info"]
#     @dt_info.setter
#     def dt_info(self, val):
#         assert val is not None
#         self.set_extra_attrs(dt_info=val)

#  CustomRulesFuncT = typing.Callable[[WithPrecision], None]
#  """ Rules Definition Function Type
#      @return: how many precisions for current symbol
#               is confirmed.
#  """
#  _CUSTOM_PREC_RULES: typing.Dict[str, CustomRulesFuncT] = {}

#  def custom_prec_rules(*op_names):
#      def _add_rules(f: CustomRulesFuncT):
#          for op in op_names:
#              _CUSTOM_PREC_RULES[op] = f
#          return f
#      return _add_rules

#  def syms_prec(syms: typing.List[WithPrecision], prec: int):
#      for c in syms:
#          c.precision = prec
#  def args_prec(s: WithPrecision, prec: int):
#      return syms_prec(s.args, prec)

#  custom_prec_rules(VAR)(lambda s: 0)
#  custom_prec_rules(ADD, SUB, BIAS_ADD)(lambda s: args_prec(s, 16))
#  custom_prec_rules(CONV2D, DENSE)(lambda s: args_prec(s, 8))
#  custom_prec_rules(MUL, SUM)(lambda s: args_prec(s, 16))


""" CVM-COMPATIABLE PRECISION INFER RULES. """

RulesFuncT = typing.Callable[[WithPrecision], None]
_INFER_RULES: typing.Dict[str, RulesFuncT] = {}

def prec_rules(*op_names):
    def _add_rules(f: RulesFuncT):
        for op in op_names:
            if op in _INFER_RULES:
                print("precision infer rules for op: %s is overrided" % op)
            _INFER_RULES[op] = f
        return f
    return _add_rules

_infer_mul: RulesFuncT = lambda s: sum([c.precision for c in s.args])
_infer_max: RulesFuncT = lambda s: max([c.precision for c in s.args])

def _infer_index(s: WithPrecision, index: int):
    return s.args[index].precision

prec_rules(TUPLE)(_infer_max)
prec_rules(MAX_AXIS)(_infer_max)
prec_rules(DIV)(_infer_max)
@prec_rules(CONV2D, DENSE)
def _infer_nn(s: WithPrecision):
    W = s.args[1]
    add_count = np.prod(W.shape[1:])
    add_bits = count_to_bits(add_count)
    return _infer_mul(s) + add_bits
# @prec_rules(BIAS_ADD)
@prec_rules(ADD, SUB)
def _infer_add(s: WithPrecision):
    """ op for ADD, SUB should consider scale the same, and then
            to be operated. Here we consider the max precision
            as the max to infer.
    """
    return _infer_max(s) + 1
prec_rules(CONCAT)(_infer_max)

@prec_rules(SOFTMAX)
@prec_rules(LOG_SOFTMAX)
@prec_rules(NEGATIVE)
@prec_rules(EXPAND_DIMS, TILE, REPEAT)
@prec_rules(ADV_INDEX)
@prec_rules(SLICE_LIKE, STRIDED_SLICE)
@prec_rules(TRANSPOSE, FLATTEN, BATCH_FLATTEN)
@prec_rules(SPLIT, TUPLE_GET_ITEM)
@prec_rules(SQUEEZE, RESHAPE)
@prec_rules(RELU, MAX_POOL2D)
@prec_rules(AS_TYPE)
def _first_like(s: WithPrecision):
    return _infer_index(s, 0)
@prec_rules(SUM)
def _infer_sum(s: WithPrecision):
    input_len = np.prod(s.args[0].shape)
    output_len = np.prod(s.shape)
    assert input_len % output_len == 0
    count = int(input_len / output_len)
    sum_bit = count_to_bits(count)
    return _infer_max(s) + sum_bit
prec_rules(MUL, MATMUL)(_infer_mul)
@prec_rules(CLIP)
def _infer_clip(s: WithPrecision):
    a_min = s.extra_attrs["a_min"] if "a_min" in s.extra_attrs else s.attrs["a_min"]
    a_max = s.extra_attrs["a_max"] if "a_max" in s.extra_attrs else s.attrs["a_max"]
    absmax = max(math.fabs(a_min), math.fabs(a_max))
    return number_to_bits(absmax)
@prec_rules(RIGHT_SHIFT)
def _infer_right_shift(s: WithPrecision):
    A, B = s.args[0], s.args[1]
    assert B.is_param()
    b_prec = InferPrecision.bind(B)
    return A.precision - b_prec

@prec_rules(REQUANT, PCLIP, RS_PCLIP)
def _infer_attr_prec(s: WithPrecision):
    assert s.parsed.precision == s.precision
    return s.parsed.precision

@dataclass(repr=False)
class PrecisionRevisor(WithPrecision, Transformer):
    def __call__(self, **kw):
        out = self
        if out.is_input():
            return
        elif out.is_op(REQUANT, PCLIP):
            assert out.precision == out.parsed.precision, f"{out.name} out_prec:{out.precision}, out_parsed_prec:{out.parsed.precision}"
        elif out.is_op(RS_PCLIP):
            assert out.precision == out.parsed.precision, f"rs_pclip: {out.name} out_prec:{out.precision}, out_parsed_prec:{out.parsed.precision}"
        elif out.is_param():
            absmax = np.abs(self.numpy()).max()
            oprec = number_to_bits(absmax)
            if out.precision_defined:
                assert oprec <= out.precision, out
            out.precision = oprec
        else:
            assert out.op_name in _INFER_RULES, (
                    "precision annotator cannot infer op:%s"
                    "'s precision."
                    ) % out.op_name
            oprec = _INFER_RULES[out.op_name](out)
            # TODO: more precision decision?
            if out.precision_defined and oprec > out.precision:
                out.precision, oprec = oprec, out.precision
                out = op.pclip(out, precision=oprec).like(
                        out, extra_attrs=out.extra_attrs)
            out.precision = oprec

        out.validate_precision()
        return out

# def cvm_infer_single_precision(
#         symbol: WithPrecision, params: ParametersT) -> int:
#     oprec = -1
#     if op.is_input(symbol, params):
#         oprec = symbol.precision
#     elif symbol.is_op(REQUANT):
#         oprec = symbol.precision
#     elif op.is_param(symbol, params):
#         absmax = np.abs(params[symbol.name].numpy()).max()
#         oprec = number_to_bits(absmax)
#         assert oprec <= symbol.precision, symbol
#     else:
#         assert symbol.op_name in _INFER_RULES, (
#                 "precision annotator cannot infer op:%s"
#                 "'s precision."
#                 ) % symbol.op_name
#         oprec = _INFER_RULES[symbol.op_name](symbol)

#     assert WithPrecision._validate(oprec, str(symbol))
#     return oprec

# def cvm_infer_precision(symbol: WithPrecision, params: ParametersT) -> int:
#     def _infer(sym: WithPrecision):
#         sym.precision = infer_single_precision(sym, params)
#         sym.validate_precision()
#         return sym
#     out = transform(symbol, _infer)
#     return out.precision
