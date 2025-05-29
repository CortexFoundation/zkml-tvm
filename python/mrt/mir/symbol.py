from __future__ import annotations
import typing

import json
from functools import wraps
from dataclasses import dataclass, fields, is_dataclass, field
from collections.abc import MutableMapping

import mrt
from mrt.common import config
from mrt.common.utils import *
from mrt.common.types import *

# from . import config
# from .utils import *
#  from .types import *
from .opns import *

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

# Model Graph/Symbol 
# Operators: Y = f(X): Conv/Dense/MatMul, ReLu/Softmax/Sigmoid
# Model: Y = f1(f2(f3(X), W, B)), Node, Var(Input/Param)
# for/if(cycle)
# CNN/RNN 

# Static Quantization / Dynamic
# y = f(x, w) * S   => batchnorm (0~1) => (0~128/256)
# precision
# MAX_BIT(32/64), model accleration/model size, int8, cuda/cpu int8 mat mul acce. int4/2/1
# X * W = up(log(n))
# => Y(y * s3) = F(X(x * s1), W(w * s2)) (MatMul/Conv/Dense)
# s3 = s1 * s2 / max(s1, s2) / s1 + s2
#
# requant op
# X(int24, xs1) => (int8, xs2) (X / xs1 = x)
# calibrate => X (-3 ~ -7) 7 -> 127 , xs2
# X xs1 => max_value => prec
# X1 = clip(clip(round(X), prec) * (xs2 / xs1), int8) => int8
# int <=> float32 multiply
#
# X1 = clip(round(x * xs2), int8)
# Y = F(X1, W(w * ws)) time/memory 5min/10G => 1min/1G
#
# requant? model quantization/acceleration
# all operations quantization.
# llama_qint4(MatMul)
# 
# calibrate method? SymmetricMinMaxSampling: out = max(abs(out))
# sample? 1, 16, 160 16
# outlier, 0, 1, 100
# hemetric sampling(95, calibrate threshold): 1hour
# finetune training 
# 
# (O, I, H, W) threshold => (O, I,) layerwise quantizaion
# 
# CNN (value) argmax(1000, index) / LLM/RNN
#
# float parallel calculate problem
# 1 + 0.00000001 * 100000000 = 1 = 2
# 
# float format: exp bits, int bits
# IEEE float scheme?
# 
# int8 sum/mul, avx256 instruct

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
            [_uniform(i.name if isinstance(i, Symbol) else str(i),
                      arg_len) for i in self.args]))
        oattrs = {k: v for k, v in self.extra_attrs.items()}
        oattrs.update(attrs)
        #  oattrs.update(self.extra_attrs)
        #  with config.LogConfig(name_width=20):
        #      return config.log_str(
        #          f"{_uniform(self.name, 20)} =",
        #          f"{self.op_name:>15}{args_info:40}",
        #          f"*attrs*", _format_printer(self.attrs),
        #          _format_printer(oattrs),
        #              )
        return "{:>20} = {:>15}{:40} | *attrs:* {} | {}".format(
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
    def is_near(self, *names) -> bool:
        """ Check is near passed names,
                examine args, name, op_name, etc.
        """
        return (config.SYMBOL_ALL_NEAR in names) or \
                (self.name in names) or \
                (self.op_name in names) or \
                all([a.name in names for a in self.args])

    @property
    def shape(self) -> ShapeT:
        shp = self.extra_attrs.get("shape", None)
        return None if shp is None else list(shp)
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
    assert isinstance(symbol, Symbol), \
            f"({type(symbol).__name__}){str(symbol)}"

    if symbol in sym_list:
    #  if sym_list.count(symbol) > 0:
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
    with config.PassConfig():
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
    C = config.LogConfig.G()
    for sym in sym2list(symbol):
        if callback.__name__ in C.log_vot_cbs:
            config.log(callback.__name__, f"<< {sym}")
        callback(sym)
        if callback.__name__ in C.log_vot_cbs:
            config.log(callback.__name__, f">> {out}")

def transform(symbol: Symbol, callback: _TransformerT) -> Symbol:
    """ Transform symbol from old to new, with inputs updated.

        Only the return value indicates mutation, while changing
        attributes in parameter passed in args does nothing.
    """
    sym_map: typing.Dict = {}
    C = config.LogConfig.G()
    for sym in sym2list(symbol):
        # pre-clone symbol with updated args,
        # to avoid misleading usage in callback.
        args = [sym_map[c.name] for c in sym.args]
        sym = sym.copy(args=args)

        if callback.__name__ in C.log_vot_cbs:
            config.log(callback.__name__, f"<< {sym}")

        out = callback(sym) or sym
        assert isinstance(out, Symbol), out
        # default const_ prefix symbol means parameters
        assert sym.name not in sym_map, sym.name
        # assert sym.name.startswith("const_") or \
        #         sym.name not in sym_map, sym.name
        sym_map[sym.name] = out
        if callback.__name__ in C.log_vot_cbs:
            config.log(callback.__name__, f">> {out}")

        #  C.log_before and print("[{} <<] {}".format(C.name, sym))
        #  new_conf = C if C.inherit else config.PassConfig()
        #  with new_conf:
        #      out = callback(sym) or sym
        #  assert isinstance(out, Symbol), out
        #  # default const_ prefix symbol means parameters
        #  assert sym.name not in sym_map, sym.name
        #  # assert sym.name.startswith("const_") or \
        #  #         sym.name not in sym_map, sym.name
        #  sym_map[sym.name] = out
        #  C.log_after and print("[{} >>] {}".format(C.name, out))
    return sym_map[symbol.name]

# =============== MultiHeadSymbol API ==================
#  @dataclass(repr=False, init=False)
#  class MultiHeadSymbol(MutableMapping, Symbol):
#      """ multihead symbol store as dict,
#          consistent with tvm.IRModule.

#          use dict or {} to initialize instance.
#      """
#      arg_names: typing.List[str]
#      _mhs: typing.Dict[str, Symbol] = field(repr=False)

#      def __repr__(self, **attrs):
#          attrs["arg_names"] = self.arg_names
#          return super(Symbol, self).__repr__(**attrs)

#      def __init__(self,
#                   name: str = None, op_name: str = None,
#                   args = [], arg_names = [],
#                   attrs = {}, extra_attrs = {},
#                   _mhs = {}, **kwargs):
#          self.name = name or N.n("mhs_")
#          assert op_name is None or op_name == TUPLE, \
#              f"MHS op_name init must be TUPLE, but get:{op_name}"
#          self.op_name = TUPLE

#          if args or arg_names:
#              assert not kwargs, f"kwargs is not empty."
#              assert len(args) == len(arg_names)
#              self.args = args
#              self.arg_names = arg_names
#              if not _mhs:
#                  _mhs = dict(zip(arg_names, args))
#              else:
#                  assert all([(n in _mhs) for n in arg_names]), \
#                      f"_mhs and arg_names:{arg_names} not consistent."
#              self._mhs = _mhs
#          else:
#              assert not _mhs
#              self._mhs = dict(kwargs)
#              self.args = list(self._mhs.values())
#              self.arg_names = list(self._mhs.keys())
#          assert all([isinstance(a, Symbol) for a in self.args])

#          self.attrs = attrs or {}
#          self.extra_attrs = extra_attrs or {}

#      def __getitem__(self, key: str) -> Symbol:
#          return self.mhs[key]
#      def __setitem__(self, key: str, val: Symbol):
#          self._mhs[key] = val
#          self.args = list(self._mhs.values())
#          self.arg_names = list(self._mhs.keys())
#      def __delitem__(self, key: str):
#          del self._mhs[key]
#          self.args = list(self._mhs.values())
#          self.arg_names = list(self._mhs.keys())
#      def __len__(self) -> int:
#          return len(self.args)
#      def __iter__(self):
#          return iter(self._mhs)
#      def items(self):
#          return self._mhs.items()

#      @classmethod
#      def from_symbol(cls,
#                      symbol: Symbol,
#                      name: str = "main") -> MultiHeadSymbol:
#          return MultiHeadSymbol(**{ name: symbol })

class MultiHeadSymbol(dict):
    origin: typing.Optional[Symbol] = None

    @classmethod
    def from_symbol(cls, symbol: Symbol, name: str = "main"):
        return MultiHeadSymbol({ name: symbol })

    def as_tuple(self) -> (typing.List[str], Symbol):
        from . import op
        #  args = list(self.values())
        #  sym_type = type(args[0]) if args else Symbol
        mhs = self.origin or op.Tuple(*list(self.values()))
        return list(self.keys()), mhs

    @classmethod
    def from_tuple(cls, tuple_names, symbol):
        assert symbol.is_op(TUPLE), symbol
        mhs = cls(zip(tuple_names, symbol.args))
        mhs.origin = symbol
        return mhs

#  MultiHeadSymbol = typing.Dict[str, Symbol]

#  def mhs_sym2list(mhs: MultiHeadSymbol) -> typing.List[Symbol]:
#      sym_list: typing.List[Symbol] = []
#      for name, sym in mhs.items():
#          _topo_sort(sym, sym_list)
#      return sym_list

#  def mhs_visit(mhs: MultiHeadSymbol, callback: _VisitorT):
#      C = config.LogConfig.G()
#      for sym in mhs_sym2list(mhs):
#          C.log_in_vot(callback.__name__, f"<< {sym}")
#          callback(sym)
#          C.log_in_vot(callback.__name__, f">> {sym}")

#  def mhs_transform(
#          mhs: MultiHeadSymbol,
#          callback: _TransformerT) -> MultiHeadSymbol:
#      sym_map: typing.Dict[str, Symbol] = {}
#      C = config.LogConfig.G()

#      from . import op
#      for sym in mhs_sym2list(mhs):
#          # pre-clone symbol with updated args,
#          # to avoid misleading usage in callback.
#          args = [sym_map[c.name] for c in sym.args]
#          sym = sym.copy(args=args)

#          C.log_in_vot(callback.__name__, f"<< {sym}")
#          out = callback(sym) or sym
#          assert isinstance(out, Symbol), out
#          # default const_ prefix symbol means parameters
#          assert sym.name not in sym_map, sym.name
#          # assert sym.name.startswith("const_") or \
#          #         sym.name not in sym_map, sym.name
#          sym_map[sym.name] = out
#          new = op.subgraph(out, [a.name for a in sym.args])
#          C.log_in_vot(callback.__name__, f">> {raw_log(new, False)}")
#      return {k: sym_map[v.name] for k, v in mhs.items()}

#  _MultiHeadSymbolJsonT = typing.Dict[str, _SymbolJsonT]
#  def mhs_dump_json(mhs: MultiHeadSymbol) -> _MultiHeadSymbolJsonT:
#      return {k: dump_json(v) for k, v in mhs.items()}
#  def mhs_load_json(
#          data: _MultiHeadSymbolJsonT,
#          **extra_attrs) -> MultiHeadSymbol:
#      return {k: load_json(v) for k, v in data}

# ^^^^^^^^^^^^^^^ MultiHeadSymbol API ^^^^^^^^^^^^^^^^^^

Graph = typing.Union[Symbol, MultiHeadSymbol]
""" Notice that Symbol and MultiHeadSymbol can both
        be regarded as a model Graph.
"""
#  def graph_visit(graph: Graph, callback: _VisitorT):
#      return visit(graph, callback)
#      #  visit_func = visit if isinstance(graph, Symbol) else mhs_visit
#      #  return visit_func(graph, callback)
#  def graph_transform(graph: Graph, callback: _TransformerT):
#      return transform(graph, callback)
#      #  trans_func = transform if isinstance(graph, Symbol) \
#      #          else mhs_transform
#      #  return trans_func(graph, callback)
#  def get_graph_outputs(graph: Graph) -> typing.List[Symbol]:
#      return [ graph, ] if isinstance(graph, Symbol) else \
#              [v for v in graph.values()]

def raw_log(symbol: Symbol, use_header = True) -> str:
    header = "{f} Raw Info {f}\n".format(f = "="*25)
    msg = [ header, ] if use_header else []
    def _log(sym: Symbol):
        msg.append(str(sym))
    visit(symbol, _log)
    use_header and msg.append("=" * len(header))
    return "\n".join(msg)

def raw_print(symbol: Symbol):
    with config.Pass():
        print(raw_log(symbol))
    # msg = "{f} Raw Print {f}".format(f = "="*25)
    # print(msg)
    # def _print(sym: Symbol):
    #     print(sym)
    # with config.Pass():
    #     visit(symbol, _print)
    # print("=" * len(msg))

def filter_operators(*op_names: typing.List[str]):
    def _pass(f):
        @wraps(f)
        def _wrapper(sym: Symbol, *args, **kw) -> typing.Any:
            if sym.is_op(*op_names):
                return f(sym, *args, **kw)
        return _wrapper
    return _pass



