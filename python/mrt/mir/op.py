""" op helper function

This module should not be imported with *, since this
    overrides many common builtins' type.
"""

from dataclasses import dataclass

from mrt.common.utils import N

from .opns import *
from .symbol import *

def subgraph(symbol: Symbol, inames=[], onames=[]) -> Symbol:
    out = []
    def _find(sym: Symbol):
        if sym.name in inames:
            return as_variable(sym)
        elif sym.name in onames:
            out.append(sym)

    def_out = transform(symbol, _find)
    out = out or [ def_out, ]
    out = out[0] if len(out) else tuple(*out)
    return out

def variable(name, shape, dtype) -> Symbol:
    """ Create varible for symbol. """
    return Symbol.from_dict({},
            name=name, op_name = VAR, args = [],
            extra_attrs = { "shape": shape, "dtype": dtype })

def as_variable(symbol: Symbol, shape=None, dtype=None) -> Symbol:
    """ inherit extra attrs """
    out = symbol.copy(op_name=VAR, args=[], attrs={})
    out.shape = shape or out.shape
    out.dtype = dtype or out.dtype
    return out

def retrieve_operator(symbol: Symbol) -> Symbol:
    return symbol.copy(args=[as_variable(c) for c in symbol.args])

def _new_op(op_name, *args, extra_attrs=None, **attrs) -> Symbol:
    return Symbol.from_dict({},
            name=N.n(), op_name=op_name,
            args=args or [], attrs=attrs or {},
            extra_attrs=extra_attrs or {})

def _register_op(op_name):
    from . import optype
    def _op(*args, **attrs) -> Symbol:
        op = _new_op(op_name, *args, **attrs)
        out = optype.infer_single(op)
        return out
    return _op

Tuple = _register_op(TUPLE)
TupleGetItem = _register_op(TUPLE_GET_ITEM)

nn_conv2d = _register_op(CONV2D)
nn_dense = _register_op(DENSE)
nn_batch_norm = _register_op(BATCH_NORM)
#  bias_add = _register_op(BIAS_ADD)

nn_relu = _register_op(RELU)

sum = _register_op(SUM)
#  mean = _register_op(MEAN)
clip = _register_op(CLIP)
ceil = _register_op(CEIL)
right_shift = _register_op(RIGHT_SHIFT)
# relax api from cast to astype
#  astype = _register_op(AS_TYPE)
cast = _register_op(AS_TYPE)
#  flatten = _register_op(FLATTEN)
adv_index = _register_op(ADV_INDEX)
zeros_like = _register_op(ZEROS_LIKE)

repeat = _register_op(REPEAT)
reshape = _register_op(RESHAPE)

add = _register_op(ADD)
sub = _register_op(SUB)
max_axis = _register_op(MAX_AXIS)
mul = _register_op(MUL)
div = _register_op(DIV)
exp = _register_op(EXP)
negative = _register_op(NEGATIVE)

requant = _register_op(REQUANT)
pclip = _register_op(PCLIP)
rs_pclip = _register_op(RS_PCLIP)
lut = _register_op(LUT)

def is_operator(symbol: Symbol, params: ParametersT = {}):
    return symbol.op_name != VAR
def is_variable(symbol: Symbol, params: ParametersT = {}):
    return symbol.op_name == VAR
def is_input(symbol: Symbol, params: ParametersT):
    return is_variable(symbol) and symbol.name not in params
def is_param(symbol: Symbol, params: ParametersT):
    return is_variable(symbol) and symbol.name in params

