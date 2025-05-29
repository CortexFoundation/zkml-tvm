from __future__ import annotations

import typing
import math
from dataclasses import dataclass, field

from mrt.mir import op
from mrt.mir.opns import *
from mrt.mir.symbol import *

from mrt.runtime import inference

from mrt.common.utils import *

from .scaler import *
from .calibrate import Sampling
from .transform import Transformer
from .precision import WithPrecision

__ALL__ = [
        "Discretor",
        "InferPrecision", "InferDiscretor",
        "InferOperator", ]


@dataclass(repr=False, unsafe_hash=True)
class DiscreteInfo:
    scale: typing.Optional[typing.Any] = None
    precision: typing.Optional[int] = None

    @property
    def undefined(self) -> bool:
        return self.scale is None and self.precision is None


@dataclass(repr=False)
class QuantInfo(WithScale, WithPrecision, Sampling):
    requant_ops: typing.Dict[DiscreteInfo, Symbol] = field(repr=False)

    @classmethod
    def default_dict(cls, **kwargs) -> dict:
        kwargs.setdefault("requant_ops", {})
        return super().default_dict(**kwargs)

    def scale_to_precision(self, scale):
        real_max = scale * self.data
        return number_to_bits(real_max)
    def precision_to_scale(self, precision):
        assert self.data is not None, self
        if self.data == 0.:
            return 1
        return bits_to_number(precision) / self.data

    def rescale(self, info: DiscreteInfo):
        """ rescale current symbol into other, aka requant.

            if scale specified:
                precision = bit(scale * data)
            if precision specified:
                scale = real(precision) / data
            TODO: choose min if both specified
        """
        scale, precision = info.scale, info.precision
        if info.undefined:
            return self
        elif scale is not None:
            precision = self.scale_to_precision(scale)
        elif precision is not None:
            scale = self.precision_to_scale(precision)

        if info not in self.requant_ops:
            curr_scale = self.scale if self.scale_defined else 1
            #TODO: add pass to check rescale=1 and duplicate requant
            out = op.requant(
                    self,
                    rescale=scale/curr_scale,
                    precision=precision,
                    )
            out = out.like(self)
            out.set_extra_attrs(
                data=self.data, scale=scale, precision=precision)
            self.requant_ops[info] = out
        return self.requant_ops[info]

RequantRulesT = typing.Callable[[QuantInfo], typing.List[DiscreteInfo]]
""" Returns expected scale and precision.

    None, None indicate that none operation to requant.
"""
_DISCRETE_REQUANT_RULES: typing.Dict[str, RequantRulesT] = {}
OpRulesT = typing.Callable[[QuantInfo], Symbol]
_DISCRETE_OP_RULES: typing.Dict[str, OpRulesT] = {}

_requant_identity = lambda s: [DiscreteInfo() for _ in s.args]
_op_identity = lambda s: s

def register_rules(*op_names,
        requant_rule: RequantRulesT | None = None,
        op_rule: OpRulesT | None = None,
        scale_rule: ScaleRulesT | None = None,
        ):
    for op in op_names:
        if requant_rule is not None:
            _DISCRETE_REQUANT_RULES[op] = requant_rule
        if op_rule is not None:
            _DISCRETE_OP_RULES[op] = op_rule
        if scale_rule is not None:
            register_scale_rules(op, rule=scale_rule)

def register_rules_with_default(*op_names,
        requant_rule: RequantRulesT | None = None,
        op_rule: OpRulesT | None = None,
        scale_rule: ScaleRulesT | None = None,
        ):
    return register_rules(
            *op_names,
            requant_rule    = requant_rule or _requant_identity,
            op_rule         = op_rule or _op_identity,
            scale_rule      = scale_rule or scale_identity,
            )

def args_max_prec(prec: int):
    def _rule(s: QuantInfo):
        return [DiscreteInfo(precision=prec) for _ in s.args]
    return _rule

register_rules_with_default(
        CONV2D,
        requant_rule=args_max_prec(8),
        scale_rule=scale_nn)

register_rules_with_default(
        DENSE, MUL, MATMUL,
        requant_rule=args_max_prec(8),
        scale_rule=scale_nn)

register_rules_with_default(SUM, requant_rule=args_max_prec(10))

def uniform_args_scale(args: typing.List[QuantInfo],
                       std_prec: int =15):
    # standard max precision for add/sub children.

    assert len(args) > 0
    #  raw_print(s)
    assert any([c.is_operator() for c in args]), \
            "Need fuse constant for uniform_args_scale"
    scales = []
    for arg in args:
        if arg.scale_defined and arg.precision < std_prec:
            scale = arg.scale
        else:
            scale = arg.precision_to_scale(std_prec)
        scales.append(scale)

    target_scale = min([s for s in scales if s != 1])
    return [DiscreteInfo(scale=target_scale) for c in args]

#  def uniform_add_sub_scales(s: QuantInfo):
#      assert len(s.args) == 2
#      A: QuantInfo = s.args[0]
#      B: QuantInfo = s.args[1]
#      assert A.is_operator() or B.is_operator(), "need fuse constant"
#      if A.scale_defined and A.precision < std_prec:
#          scaleA = A.scale
#      else:
#          scaleA = A.precision_to_scale(std_prec)

#      if B.scale_defined and B.precision < std_prec:
#          scaleB = B.scale
#      else:
#          scaleB = B.precision_to_scale(std_prec)

#      scale = min(scaleA, scaleB)
#      return [DiscreteInfo(scale=scale) for c in s.args]
def scale_like_index(s: WithScale, index: int = 0):
    return s.args[index].scale

register_rules_with_default(
        ADD, SUB,
        # BIAS_ADD,
        MAXIMUM, MINIMUM,
        requant_rule=lambda s: uniform_args_scale(s.args),
        scale_rule=scale_like_index)

def scale_concat(s: WithScale):
    fscale = s.args[0].scale
    if all([a.scale == fscale for a in s.args]):
        return fscale
    return [a.scale for a in s.args]
register_rules_with_default(
        CONCAT, TUPLE,
        requant_rule=lambda s: uniform_args_scale(s.args),
        scale_rule=scale_concat)

def uniform_first_scale(s: QuantInfo):
    target_scale = s.args[0].scale
    return [DiscreteInfo(scale=target_scale) for c in s.args]

register_rules_with_default(
        GREATER,
        requant_rule=uniform_first_scale)

# register_rules_with_default(
#         WHERE,
#         requant_rule=lambda s: uniform_args_scale(s.args[1:]),
#         scale_rule=scale_like_index(s, -1),
#         )

register_rules_with_default(
        MAX_POOL2D, RELU,
        REPEAT, SQUEEZE, FLATTEN, BATCH_FLATTEN,
        RESHAPE, SPLIT, TRANSPOSE,
        EXPAND_DIMS, TILE,
        )

register_rules_with_default(SLICE_LIKE, STRIDED_SLICE)
register_rules_with_default(NEGATIVE)

def scale_tuple_get_item(s: WithScale):
    ascale = s.args[0].scale
    if isinstance(ascale, (list, tuple)):
        return ascale[s.parsed.index]
    return ascale
register_rules_with_default(
        TUPLE_GET_ITEM,
        scale_rule=scale_tuple_get_item)

def op_clip_rules(s: QuantInfo):
    scale = s.args[0].scale
    s.set_extra_attrs(
            a_min=s.parsed.a_min * scale,
            a_max=s.parsed.a_max * scale)
    out = s.copy()
    out.attrs["a_min"]=s.parsed.a_min * scale
    out.attrs["a_max"]=s.parsed.a_max * scale
    return out

register_rules_with_default(CLIP, op_rule=op_clip_rules)

LUT_INP_PREC, LUT_OUT_PREC = 16, 16
def lut_max_prec(s: QuantInfo):
    return [DiscreteInfo(precision=LUT_INP_PREC) for a in s.args]

def lut_scale_rules(s: QuantInfo):
    return s.precision_to_scale(LUT_OUT_PREC)

def op_lut_rules(s: QuantInfo):
    alpha = bits_to_number(LUT_INP_PREC)

    X = s.args[0]
    offset = s.from_np_data(np.array(alpha, "int"))
    indices = op.add(X, offset).like(X)
    indices = op.clip(indices, a_min=0,  a_max=2*alpha).like(X) #a_max=alpha+1)
    indices = op.cast(indices, dtype="int32")

    # arg_min, arg_max = -s.data, s.data
    # if s.is_op(EXP):
    #     arg_max = min(math.log(s.data), arg_max)

    op_inp = np.arange(-alpha, alpha+1) / s.args[0].scale
    table = inference.run(s, [ tvm.nd.array(op_inp), ])
    table = np.clip(table.numpy(), a_min=-s.data, a_max=s.data)
    # table = np.reshape(table, (-1, 1))
    oscale = s.precision_to_scale(LUT_OUT_PREC)
    weight = s.from_np_data(table * oscale)
    out = op.adv_index(weight, indices).like(s)
    # out.scale = s.precision_to_scale(LUT_INP_PREC)
    return out

register_rules_with_default(
        EXP, SIGMOID,
        requant_rule=lut_max_prec,
        op_rule=op_lut_rules,
        scale_rule=lut_scale_rules)

SOFTMAX_PREC = 15 # set by default
def softmax_scale_rules(s: QuantInfo):
    return s.precision_to_scale(SOFTMAX_PREC)

def op_softmax_rules(s: QuantInfo):
    lambd = 10
    X = s.args[0] # get requant rule op
    Xp = X.attrs["precision"]
    Xs = X.scale  #X.attrs["precision"]
    axis = s.attrs["axis"]
    alpha = int(lambd * Xs)
    var = s.from_np_data(np.array(alpha, "int"))

    max_axis = op.max_axis(X, axis = axis, keepdims=True)
    offset = op.sub(max_axis, var)
    offset = op.pclip(offset, precision=Xp)
    offset.set_extra_attrs(precision=Xp)
    norm = op.sub(X, offset)
    norm = op.nn_relu(norm)
    norm = op.pclip(norm, precision=Xp)
    norm.set_extra_attrs(precision=Xp)
    # TODO: norm = op.cast(norm, dtype="int32")
    norm = op.cast(norm, dtype="int32")

    op_inp = np.arange(0, alpha+1) / Xs
    table = np.exp(op_inp)
    tprec = number_to_bits(math.exp(lambd))
    table = np.clip(table, a_min=0, a_max=(bits_to_number(tprec)))
    weight = np.round(table)
    # weight = np.transpose(weight, (1, 0))
    weight = s.from_np_data(weight)
    out_lut = op.adv_index(weight, norm).like(s)
    sum_lut = op.sum(out_lut, axis=axis, keepdims=True).like(out_lut)

    oprec = min(SOFTMAX_PREC, 31 - tprec)
    oscale = bits_to_number(oprec)
    nd_oscale = s.from_np_data(np.array(oscale, "int"))
    prob = op.mul(out_lut, nd_oscale)

    half_lut = op.rs_pclip(sum_lut, s.from_const_data(1), precision=31)
    half_lut.set_extra_attrs(precision=31)
    prob = op.add(prob, half_lut)
    out = op.div(prob, sum_lut)
    out = op.cast(out, dtype="int32")
    out = op.cast(out, dtype="float32")
    out = op.pclip(out, precision=oprec)
    out.set_extra_attrs(scale=oscale, precision=oprec)

    return out

register_rules_with_default(
        SOFTMAX, LOG_SOFTMAX,
        requant_rule=args_max_prec(SOFTMAX_PREC),
        op_rule=op_softmax_rules,
        scale_rule=softmax_scale_rules
)

@dataclass(repr=False)
class Discretor(QuantInfo):
    """
        does operation -> out

        input scales -> output scale
            -> output tighter precision
        # sampling * output scale -> output precision
        input precisions -> output precision
        if output tighter precision < output precision:
            out <- pclip(out, output tighter precision)
            output precision <- output tighter precision

        Case 1: sampling, precision(target) -> scale
        if output precision <= precision:
            scale <- output scale
            precision <- output precision
        else:
            out = requant(out, scale / output scale)
            output precision <- precision(target)
            output scale <- scale

        Case 2: sampling, scale -> precision(target)
        out = requant(out, scale / output scale)
        output precision <- precision(target)
        output scale <- scale
    """
    def __call__(self, **kw):
        if self.is_variable():
            return
        elif self.is_op(TUPLE):
            return

        orig_names = [a.name for a in self.args]

        assert self.op_name in _DISCRETE_REQUANT_RULES, (
                "requant rules not support for op:{}"
                ).format(self.op_name)
        assert self.op_name in _DISCRETE_OP_RULES, (
                "op rewrite rules not support for op:{}"
                ).format(self.op_name)
        assert self.op_name in INFER_SCALE_RULES, (
                "op rewrite rules not support for op:{}"
                ).format(self.op_name)

        # requant input to specific precision
        arg_dts = _DISCRETE_REQUANT_RULES[self.op_name](self)
        for i, arg in enumerate(self.args):
            self.args[i] = arg.rescale(arg_dts[i])

        # calculate the F function
        out = _DISCRETE_OP_RULES[self.op_name](self).like(
                self, extra_attrs=self.extra_attrs)

        # calculate the output data's scale
        out.scale = INFER_SCALE_RULES[self.op_name](out)
        new = op.subgraph(out, inames=[a.name for a in self.args])
        #  self.is_op(EXP) and raw_print(new)
        #  out.scale = infer_scale(new)

        # tight output's precsion based on the threshold and scale
        # out.precision = infer_precision(new, self.params)
        # target_precision = self.scale_to_precision(out.scale)
        # if out.precision > target_precision:
        #     out = op.pclip(out, precision=target_precision).like(
        #             out, extra_attrs=out.extra_attrs)
        #     out.precision = target_precision
        out.precision = self.scale_to_precision(out.scale)

        # TODO: add skip for some operators
        # same_scale = all([a.scale == out.scale for a in self.args])
        # same_data = all([a.data == out.data for a in self.args])
        # if not (same_scale and same_data):
        #     out = op.pclip(out, precision=out.precision).like(
        #             out, extra_attrs=out.extra_attrs)
        #  raw_print(op.subgraph(out, inames=orig_names))
        return out

