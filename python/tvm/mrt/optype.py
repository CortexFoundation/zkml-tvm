import typing

from .types import *

from . import op, opns
from .frontend.expr import symbol2expr, expr2symbol
from . import config
from .symbol import Symbol, transform

InferTypeT = typing.Callable[[Symbol], Symbol]
_INFER_TYPE_REG: typing.Dict[str, InferTypeT] = {}

def _tvm_type_infer(symbol: Symbol) -> Symbol:
    from tvm import relay, ir
    expr = symbol2expr(symbol)
    mod = relay.transform.InferType()(ir.IRModule.from_expr(expr))
    expr = mod["main"].body
    out, _ = expr2symbol(expr)
    return out

def register_type_infer(
        *op_names,
        rule: typing.Optional[InferTypeT] = None):
    def _set_rule(func: InferTypeT):
        for op in op_names:
            _INFER_TYPE_REG[op] = func
        return func

    if rule is not None:
        return _set_rule(rule)
    return _set_rule

def infer(symbol: Symbol) -> Symbol:
    def _tinfer(sym: Symbol):
        out = op.retrieve_operator(sym)
        _infer = _INFER_TYPE_REG.get(out.op_name, _tvm_type_infer)
        out: Symbol = _infer(out) or out
        assert out.shape is not None, out
        assert out.dtype is not None, out
        type_info = { "shape": out.shape, "dtype": out.dtype }
        return sym.copy(extra_attrs={
            **sym.extra_attrs, **type_info })

    return transform(symbol, _tinfer)

def infer_single(symbol: Symbol) -> Symbol:
    single = op.retrieve_operator(symbol)
    single = infer(single)
    return symbol.copy(extra_attrs=single.extra_attrs)

@register_type_infer(opns.ARGWHERE)
def _argwhere(symbol: Symbol) -> Symbol:
    X: Symbol = symbol.args[0]
    symbol.shape = X.shape
    symbol.dtype = "int32"

@register_type_infer(opns.TUPLE)
def _tuple(symbol: Symbol) -> Symbol:
    symbol.shape = [a.shape for a in symbol.args]
    symbol.dtype = [a.dtype for a in symbol.args]

@register_type_infer(opns.TUPLE_GET_ITEM)
def _tuple_get_item(symbol: Symbol) -> Symbol:
    X: Symbol = symbol.args[0]
    index = symbol.attrs["index"]
    symbol.shape = X.shape[index]
    symbol.dtype = X.dtype[index]

@register_type_infer(opns.REQUANT, opns.PCLIP, opns.RS_PCLIP)
def _type_like_first(symbol: Symbol) -> Symbol:
    X: Symbol = symbol.args[0]
    symbol.shape = X.shape
    symbol.dtype = X.dtype
