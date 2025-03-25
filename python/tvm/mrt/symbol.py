from __future__ import annotations
import typing

import json
from functools import wraps
from dataclasses import dataclass, fields, is_dataclass

from . import config
from .utils import *
from .types import *

__ALL__ = [
        "Symbol",
        "visit", "transform",
        "filter_operators",
        ]

def _format_printer(data):
    if isinstance(data, dict):
        data = ["{}={}".format(k, _format_printer(v)) \
                for k, v in data.items()]
        return ", ".join(data)
    elif isinstance(data, (tuple, list)):
        return "(" + ",".join([_format_printer(d) \
                for d in data]) + ")"
    elif isinstance(data, float):
        return "{:.3f}".format(data)
    return str(data)[-20:]

@dataclass(repr=False)
class _BaseSymbol:
    """ Symbol should record neccessary infomation about
            the transformer, such as discretization method,
            precision, etc.
    """
    name: str
    op_name: str
    args: typing.List[Symbol]
    attrs: typing.Dict[str, typing.Any]
    extra_attrs: typing.Dict[str, typing.Any]
    """ extra attributes will be inherited automatically. """

    @classmethod
    def update_extra_attrs(cls, data_dict, **kwargs):
        extra_attrs: dict = data_dict.get("extra_attrs", {})
        extra_attrs.update(kwargs)
        data_dict["extra_attrs"] = extra_attrs
        return data_dict
    def set_extra_attrs(self, **kwargs) -> _BaseSymbol:
        self.extra_attrs.update(kwargs)
        return self

    @classmethod
    def base(cls, symbol: Symbol, **kwargs):
        """ create current class instance based on another.
            Enable the inherit class to override.
        """
        return cls.from_dict(symbol.to_dict(), **kwargs)
    def like(self, other: Symbol, **kwargs) -> Symbol:
        """ cast current symbol to child class. """
        #  assert self.shape == other.shape, "%s vs.\n %s" % (self, other)
        #  assert self.dtype == other.dtype , "%s vs.\n %s" % (self, other)
        data = other.to_dict()
        data.update(self.to_dict())
        # copy extra attrs by default.
        #  data["extra_attrs"] = other.extra_attrs
        return type(other).from_dict(data, **kwargs)
    def copy(self, **kwargs) -> typing.Type[_BaseSymbol]:
        """ clone current symbol. """
        return type(self).from_dict(
            self.to_dict(), **kwargs) # kwargs override self

    @classmethod
    def default_dict(cls, **kwargs) -> dict:
        """ possible dict to initialize symbol class. """
        kwargs.setdefault("extra_attrs", {})
        return kwargs
    @classmethod
    def update_dict(cls, data_dict: dict, **kwargs) -> dict:
        data_dict.update(kwargs)
        return data_dict
    @classmethod
    def from_dict(cls, d: dict, **kwargs):
        data = cls.default_dict()
        data.update(d)
        data.update(kwargs)
        data = cls.update_dict(data)
        fnames = [f.name for f in fields(cls)]
        data = {k: data[k] for k in data if k in fnames}
        try:
            out = cls(**data)
        except Exception as e:
            #  print(cls, list(data.keys()))
            #  raise e
            raise untimeError((
                "Error for type:{} create from dict, "
                "expected: {}, but get {}"
                ).format(get_class_name(cls),
                    fnames, data.keys()))
        return out
    def to_dict(self, **kwargs) -> dict:
        data = dataclass_to_dict(self)
        data.update(**kwargs)
        data["args"] = [a for a in data["args"]]
        data["attrs"] = {k: v for k, v in self.attrs.items()}
        data["extra_attrs"] = {k: v \
                for k, v in data["extra_attrs"].items()}
        return data

    def __repr__(self, **attrs) -> str:
        def _uniform(n: str, max_size: int) -> str:
            if len(n) <= max_size:
                return n
            return "..." + n[3-max_size:]

        arg_len = 40 - 2
        if len(self.args) > 0:
            arg_len = (arg_len-2*(len(self.args)-1)) // len(self.args)
            arg_len = max(arg_len, 7)
        args_info = "({})".format(", ".join(
            [_uniform(i.name, arg_len) for i in self.args]))
        oattrs = {k: v for k, v in self.extra_attrs.items()}
        oattrs.update(attrs)
        #  oattrs.update(self.extra_attrs)
        return "{:>20} = {:>15}{:40} /* attrs */ {} | {}".format(
                _uniform(self.name, 20),
                self.op_name, args_info,
                _format_printer(self.attrs),
                _format_printer(oattrs))


@dataclass
class Symbol(_BaseSymbol):
    """ Uniform Symbol Representation for RelayExpr

    RelayExpr has different format for operators, functions,
        which is hard to apply uniform transformation pass.
        Such as the `TupleGetItem`.

    Abstract representation allows different definitions
        for operators, which can be easier for graph
        transformation. Like the `BatchNorm` op returns
        a 3-tuple, whereas the return is first in cvm.

    We need to consistently print symbol information such as name,
        for the user's config about quantization layers.
    """

    # Overridable Methods, inheritted from _BaseSymbol
    #   to support multi-inherit design.
    @classmethod
    def update_extra_attrs(cls, data_dict, **kwargs):
        return super().update_extra_attrs(data_dict, **kwargs)
    def set_extra_attrs(self, **kwargs):
        return super().set_extra_attrs(**kwargs)
    @classmethod
    def base(cls, symbol: Symbol, **kwargs) -> Symbol:
        return super().base(symbol, **kwargs)
    def like(self, other: Symbol, **kwargs) -> Symbol:
        return super().like(other, **kwargs)
    def copy(self, **kwargs) -> Symbol:
        return super().copy(**kwargs)
    @classmethod
    def from_dict(cls, d: dict, **kwargs):
        return super().from_dict(d, **kwargs)
    @classmethod
    def default_dict(cls, **kwargs) -> dict:
        kwargs.setdefault("args", [])
        kwargs.setdefault("attrs", {})
        return super().default_dict(**kwargs)
    @classmethod
    def update_dict(cls, data_dict: dict, **kwargs) -> dict:
        return super().update_dict(data_dict, **kwargs)
    def to_dict(self, **kwargs) -> dict:
        return super().to_dict(**kwargs)
    def __repr__(self, **attrs) -> str:
        return super().__repr__(**attrs)
    def info(self, **attrs) -> str:
        return super().__repr__(**attrs)

    # Naive Methods
    def is_op(self, *op_names) -> bool:
        """ Check current symbol is in the op name list. """
        assert len(op_names) > 0
        return self.op_name in op_names

    @property
    def shape(self) -> ShapeT:
        shp = self.extra_attrs.get("shape", None)
        return shp and list(shp)
    @shape.setter
    def shape(self, val):
        self.extra_attrs["shape"] = list(val)

    @property
    def dtype(self):
        return self.extra_attrs.get("dtype", None)
    @dtype.setter
    def dtype(self, val):
        self.extra_attrs["dtype"] = val

    @property
    def subgraph(self):
        return self.extra_attrs.get("subgraph", None)
    def set_subgraph(self, val):
        self.extra_attrs["subgraph"] = val

    def __hash__(self) -> int:
        return hash(str(self))
    def hash(self) -> int:
        return hash(str(self))


def _topo_sort(symbol: Symbol, sym_list: typing.List[Symbol]):
    assert isinstance(symbol, Symbol), type(symbol)

    if sym_list.count(symbol) > 0:
        return
    for c in symbol.args:
        _topo_sort(c, sym_list)
    sym_list.append(symbol)

def sym2list(symbol: Symbol) -> typing.List[Symbol]:
    sym_list: typing.List[Symbol]  = []
    _topo_sort(symbol, sym_list)
    return sym_list

_SymbolNodesT = typing.List[typing.Dict[str, typing.Any]]
_SymbolJsonT = typing.Dict[str, typing.Any]


def dump_json(symbol: Symbol) -> _SymbolJsonT:
    nodes = []
    def _to_json(sym: Symbol):
        node = dataclass_to_dict(sym, check_repr=True)
        node.update({
            "args": [a.name for a in node["args"]],
            "_class_type": get_class_name(sym),
            })
        nodes.append(node)
    with config.Pass():
        visit(symbol, _to_json)
    return { "nodes": nodes, }

def load_json(data: _SymbolJsonT, **extra_attrs) -> Symbol:
    nodes: _SymbolNodesT = data["nodes"]

    sym_map: typing.Dict = {}
    for node in nodes:
        args = [sym_map[a] for a in node["args"]]
        sym_type: typing.Type[Symbol] = eval(node["_class_type"])
        sym = sym_type.from_dict(node, args=args, **extra_attrs)
        sym_map[sym.name] = sym
    return sym_map[nodes[-1]["name"]]

_VisitorT = typing.Callable[[Symbol], None]
_TransformerT = typing.Callable[[Symbol], typing.Optional[Symbol]]
""" Symbol Transformer

    Return new symbol to transform old symbol into updated one,
        or just return None for symbol visit.
"""

def visit(symbol: Symbol, callback: _VisitorT):
    """ Visitor mode, possible modify symbol itself. """
    C = config.Pass.G()
    for sym in sym2list(symbol):
        C.log_before and print("[{} <<] {}".format(C.name, sym))
        callback(sym)
        C.log_after and print("[{} >>] {}".format(C.name, sym))

def transform(symbol: Symbol, callback: _TransformerT) -> Symbol:
    """ Transform symbol from old to new, with inputs updated.

        Only the return value indicates mutation, while changing
        attributes in parameter passed in args does nothing.
    """
    sym_map: typing.Dict = {}
    C = config.Pass.G()
    for sym in sym2list(symbol):
        args = [sym_map[c.name] for c in sym.args]
        # pre-clone symbol, to avoid misleading usage in callback
        sym = sym.copy(args=args)
        C.log_before and print("[{} <<] {}".format(C.name, sym))

        new_conf = C if C.inherit else config.Pass()
        with new_conf:
            out = callback(sym) or sym
        assert isinstance(out, Symbol), out
        # default const_ prefix symbol means parameters
        assert sym.name not in sym_map, sym.name
        # assert sym.name.startswith("const_") or \
        #         sym.name not in sym_map, sym.name
        sym_map[sym.name] = out
        C.log_after and print("[{} >>] {}".format(C.name, out))
    return sym_map[symbol.name]

def raw_print(symbol: Symbol):
    msg = "{f} Raw Print {f}".format(f = "="*25)
    print(msg)
    def _print(sym: Symbol):
        print(sym)
    with config.Pass():
        visit(symbol, _print)
    print("=" * len(msg))

def filter_operators(*op_names: typing.List[str]):
    def _pass(f):
        @wraps(f)
        def _wrapper(sym: Symbol, *args, **kw) -> typing.Any:
            if sym.is_op(*op_names):
                return f(sym, *args, **kw)
        return _wrapper
    return _pass
