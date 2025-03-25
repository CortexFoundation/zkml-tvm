from __future__ import annotations
import typing

import copy
from functools import wraps
import pprint

import tvm
from tvm import relay, ir
from tvm.ir.expr import *
from tvm.relay.expr import *

from .types import *

def update_expr_args(old: RelayExpr, expr_map) -> RelayExpr:
    try:
        new = eval("relay." + op_name(old))(
                *args(old), **attrs(old))
    except Exception as e:
        print(op_name(old))
        raise e

    if isinstance(new, relay.TupleWrapper):
        new = new.tuple_value
    return new


def clone(expr: RelayExpr, **kwargs) -> RelayExpr:
    expr = copy.copy(expr)
    for k, v in kwargs.items():
        setattr(expr, k, v)


_VisitorT = typing.Callable[ [RelayExpr], None ]
_TransformerT = typing.Callable[
        [RelayExpr], typing.Optional[RelayExpr]]
""" Expr Transformer

    Return new expr to transform old expr into updated one,
        or just return None for expr visit.
"""

def transform(expr: RelayExpr, callback: _TransformerT) -> RelayExpr:
    expr_list: typing.List[RelayExpr] = []
    def _collect_expr(expr: RelayExpr):
        # primitive ir operators, wrapper by CallNode
        if isinstance(expr, ir.op.Op):
            return

        expr_list.append(expr)
    relay.analysis.post_order_visit(expr, _collect_expr)

    expr_map = {}
    for i, sym in enumerate(expr_list):
        out = update_expr_args(sym, expr_map)
        # pre-clone symbol, to avoid misleading usage in callback
        out = callback(out) or out
        assert isinstance(out, RelayExpr)
        expr_map[sym] = out
    return expr_map[expr]

def infer_type(expr: RelayExpr) -> expr:
    mod = relay.transform.InferType()(ir.IRModule.from_expr(expr))
    return mod["main"].body

def visit(expr: RelayExpr, callback: _VisitorT):
    expr_list: typing.List[RelayExpr] = []
    def _collect_expr(expr: RelayExpr):
        # primitive ir operators, wrapper by CallNode
        if isinstance(expr, ir.op.Op):
            return
        expr_list.append(expr)
    relay.analysis.post_order_visit(expr, _collect_expr)

    for sym in expr_list:
        callback(sym)


def simple_raw_print(expr: RelayExpr, params: ParametersT = {}):
    info = { "op": 0, "param": 0 }
    def _simple_visit(sym):
        if not is_operator(sym):
            print("{:68} /* attrs */ \t{}".format(
                sym.name, sym.attrs))
            if is_param(sym, params):
                info["param"] += utils.product(sym.attrs["shape"])
            return

        info["op"] += 1
        print("{:15} = {:>20}{:30} /* attrs */ \t{}".format(
            sym.name, sym.op_name,
            "(" + ", ".join([i.name for i in sym.args]) + ")",
            sym.attrs,
        ))
    transform(expr, _simple_visit)
    print("="*50)
    print("Operators: {} | Parameters: {}".format(
        info["op"], info["param"]))
    print("="*50)

def to_json(expr: RelayExpr):
    json_map = {}
    def _cast(expr: RelayExpr):
        data = {
            "op_name": op_name(expr),
            "args": [],
            "attrs": {},
        }

        json_map[expr] = data
    visit(expr, _cast)
    return json_map[expr]


def filter_operators(*op_names: typing.List[str]):
    def _pass(f):
        @wraps(f)
        def _wrapper(expr: RelayExpr, *args, **kw):
            if op_name(expr) not in op_names:
                return
            return f(expr, *args, **kw)
        return _wrapper
    return _pass

VAR_NAME = "var"
TUPLE_NAME = "Tuple"
TUPLE_GET_ITEM_NAME = "TupleGetItem"

def op_name(expr: RelayExpr):
    if isinstance(expr, Call):
        return expr.op.name
    elif isinstance(expr, TupleGetItem):
        return TUPLE_GET_ITEM_NAME
    elif isinstance(expr, Tuple):
        return TUPLE_NAME
    elif isinstance(expr, Var):
        return VAR_NAME
    assert False, type(expr)

def args(expr: RelayExpr) -> List[RelayExpr]:
    if isinstance(expr, Call):
        return expr.args
    elif isinstance(expr, TupleGetItem):
        return [ expr.tuple_value ]
    elif isinstance(expr, Tuple):
        return expr.fields
    elif isinstance(expr, Var):
        return []
    assert False, type(expr)

def attrs(expr: RelayExpr) -> dict:
    if isinstance(expr, Call):
        attrs = expr.attrs or {}
        return {k: attrs[k] for k in attrs.keys()}
    elif isinstance(expr, TupleGetItem):
        return { "index": expr.index }
    elif isinstance(expr, Tuple):
        return {}
    elif isinstance(expr, Var):
        return {
            "name_hint": expr.name_hint,
            "shape": expr.type_annotation.concrete_shape,
            "dtype": expr.type_annotation.dtype,
        }
    assert False, type(expr)


def is_operator(expr: RelayExpr, params: ParametersT = {}):
    return not isinstance(expr, Var)
def is_variable(expr: RelayExpr, params: Parameters = {}):
    return isinstance(expr, Var)
def is_param(expr: RelayExpr, params: Parameters):
    return is_variable(expr) and expr.name_hint in params
def is_input(expr: RelayExpr, params: Parameters):
    return is_variable(expr) and expr.name_hint not in params



