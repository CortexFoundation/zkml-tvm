"""
==============================================================
API from relay.Function to Symbol.
==============================================================
"""
import typing
from dataclasses import dataclass

from tvm import relay, ir, tir
from tvm.ir.expr import *
from tvm.relay.expr import *

from ..opns import *
from ..symbol import *
from ..types import *
from .. import op

__ALL__ = [ "expr2symbol", "symbol2expr", "tvm_type_infer" ]

def tvm_type_infer(expr: RelayExpr):
    mod = relay.transform.InferType()(ir.IRModule.from_expr(expr))
    return mod["main"].body

def _expr_type(checked_type: ir.type.Type, key):
    if isinstance(checked_type, ir.type.TupleType):
        return [_expr_type(f, key) for f in checked_type.fields]
    return getattr(checked_type, key)

def _convert_to_py(value):
    if isinstance(value, ir.container.Array):
        return [ _convert_to_py(v) for v in value ]
    elif isinstance(value, ir.container.Map):
        return {k: _convert_to_py(v) for k, v in value.items()}
    elif isinstance(value, tir.expr.IntImm):
        return int(value)
    # elif isinstance(value, relay.expr.Constant):
    #     return value.data.numpy().tolist()
    # # arange attrs may contains other operators' output
    # elif isinstance(value, relay.expr.Call):
    #     return None
    return value

def _format_containers(attrs):
    for k, v in attrs.items():
        attrs[k] = _convert_to_py(v)

def expr2symbol(
        expr: RelayExpr,
        params: ParametersT = {},
        ) -> (Symbol, ParametersT):
    params = {k: v for k, v in params.items()}

    mod = relay.transform.InferType()(ir.IRModule.from_expr(expr))
    expr = mod["main"].body

    symbol_map = {}
    def _cast_expr(node: RelayExpr):
        if isinstance(node, ir.op.Op):
            """ processed in Call expr. """
            return

        if isinstance(node, relay.expr.Constant):
            name = N.n("const_")
            params[name] = node.data
            symbol_map[node] = op.variable(name,
                    node.data.shape, node.data.dtype)
            return

        try:
            dtype = _expr_type(node.checked_type, "dtype")
        except Exception as e:
            # print(type(node))
            dtype = None

        try:
            shape = _expr_type(node.checked_type, "concrete_shape")
        except Exception as e:
            shape = None
            # print(type(node), e)
            # if isinstance(node, relay.expr.Call):
            #     print(node.op.name)

        attrs = { "extra_attrs": { "shape": shape, "dtype": dtype }, }
        _format_containers(attrs)

        if isinstance(node, relay.expr.Var):
            name = node.name_hint or N.n(prefix="input_")
            symbol_map[node] = op.variable(name, shape, dtype)
        elif isinstance(node, relay.expr.If):
            args = [ node.cond, node.true_branch, node.false_branch ]
            args = [symbol_map[i] for i in args]
            symbol_map[node] = op._new_op(IF, *args, **attrs)
        elif isinstance(node, relay.expr.Call):
            op_name = node.op.name
            if op_name in [CONCAT, ADV_INDEX]:
                args = [symbol_map[f] for f in node.args[0].fields]
            else:
                args = [symbol_map[i] for i in node.args]

            nattrs = node.attrs or {}
            attrs.update({k: nattrs[k] for k in nattrs.keys()})
            _format_containers(attrs)
            # op:arange has duplicate attrs for (start, stop, step)
            if op_name in [ ARANGE, ]:
                for k in ["start", "stop", "step"]:
                    attrs.pop(k)
            elif op_name == "broadcast_to":
                attrs.pop("dtype")
            elif op_name == GET_VALID_COUNT:
                attrs.pop("score_threshold")
            symbol_map[node] = op._new_op(op_name, *args, **attrs)
        elif isinstance(node, relay.TupleGetItem):
            args = [ symbol_map[node.tuple_value], ]
            attrs['index'] = node.index
            symbol_map[node] = op._new_op(
                    TUPLE_GET_ITEM, *args, **attrs)
        elif isinstance(node, relay.Tuple):
            args = [ symbol_map[f] for f in node.fields ]
            symbol_map[node] = op._new_op(TUPLE, *args, **attrs)
        else:
            raise RuntimeError(
                "MRT not support expr type:{}".format(type(node)))


    with N():
        relay.analysis.post_order_visit(expr, _cast_expr)
    return symbol_map[expr], params

def symbol2expr(symbol: Symbol,
        params: ParametersT={}, expr_map={}) -> RelayExpr:
    expr_map.clear()
    def _make_expr(sym: Symbol, args, attrs) -> relay.expr.Expr:
        try:
            return eval("relay." + sym.op_name)(*args, **attrs)
        except Exception as e:
            print(sym, [type(a) for a in args], attrs)
            raise e

    def _cast_symbol(sym: Symbol):
        args = [expr_map[i.name] for i in sym.args]

        attrs = {k: v for k, v in sym.attrs.items()}
        # operator creator don't need shape or dtype attrs,
        #   except for the variable.
        if op.is_variable(sym):
            if isinstance(sym.dtype, (list, tuple)):
                inputs = []
                for i, (s, d) in enumerate(zip(sym.shape, sym.dtype)):
                    attrs.update({
                        "shape": s, "dtype": d,
                        "name_hint": "%s.%s" % (sym.name, i)})
                    out = _make_expr(sym, args, attrs)
                    inputs.append(out)
                expr_map[sym.name] = relay.Tuple(inputs)
                return

            attrs.update({
                "shape": sym.shape, "dtype": sym.dtype,
                "name_hint": sym.name,
            })

        # tvm dropout output is not tuple.
        if sym.is_op(TUPLE_GET_ITEM):
            if sym.args[0].is_op(DROP_OUT):
                expr_map[sym.name] = args[0]
                return
        if sym.is_op(TUPLE):
            out = relay.Tuple(args)
        elif sym.is_op(CONCAT):
            out = relay.concatenate(args, **attrs)
        elif sym.is_op(ADV_INDEX):
            out = relay.adv_index(args)
        elif op.is_param(sym, params) and len(sym.shape) == 0:
            out = relay.Constant(params[sym.name])
        else:
            out = _make_expr(sym, args, attrs)

        if isinstance(out, relay.TupleWrapper):
            out = out.tuple_value
        expr_map[sym.name] = out

    visit(symbol, _cast_symbol)

    return expr_map[symbol.name]
