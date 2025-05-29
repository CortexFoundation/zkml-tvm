import typing

import numpy as np

from mrt.common import config
from mrt.runtime.types import *

from .symbol import *
from . import op

MultiHeadSymbol = typing.Dict[str, Symbol]
""" multihead symbol store as dict,
    consistent with tvm.IRModule.
"""

# class MultiHeadSymbol:
#     mod: typing.Dict[str, Symbol] = {}

#     def __setitem__(self, name: str, val: Symbol):
#         self.mod[name] = val
#     def __getitem__(self, name: str):
#         return self.mod[name]

#     def merge_mod_params(self, new_params: ParametersT):
#         for k, v in new_params.items():
#             if k not in self.params:
#                 continue
#             assert np.allclose(self.params[k].numpy(), v.numpy()), \
#                 f"parameter:{k} not equal, don't know use which one."
#         self.params.update(new_params)

#     def validate_input_or_params(self) -> (
#             typing.List[Symbol],
#             typing.List[Symbol]):
#         """ validate input and params and return. """
#         sym_inputs = []
#         sym_params = []
#         def _check_params(sym: Symbol):
#             if op.is_input(sym, self.params):
#                 sym_inputs.append(sym)
#             if op.is_param(sym, self.params):
#                 data = self.params[sym.name]
#                 assert sym.shape == list(data.shape), \
#                     f"param:{sym.name} shape inconsistent: " + \
#                     f"{sym.shape} vs. {data.shape}"
#                 assert sym.dtype == data.dtype, \
#                     f"param:{sym.name} dtype inconsistent: " + \
#                     f"{sym.dtype} vs. {data.dtype}"
#                 sym_params.append(sym)
#         self.visit(_check_params)
#         # remove unused params
#         self.params = {s.name: self.params[s.name] \
#                 for s in sym_params}
#         return sym_inputs, sym_params

#     def copy(self) -> Graph:
#         """ deep copy. """
#         return Graph(
#             {k: v for k, v in self.mod.items()},
#             {k: v for k, v in self.params},)

#     def to_json(self):
#         return {
#             "mod": {k: dump_json(v) for k, v in self.mod.items()},
#             "params": {k: to_numpy(v) for k, v in self.params.items()}
#         }

#     @classmethod
#     def from_json(cls, self, data) -> Graph:
#         return Graph(
#             {k: load_json(v) for k, v in data["mod"]},
#             {k: to_ndarray(v) for k , v in data["params"]})

def topo_sort(graph: MultiHeadSymbol) -> typing.List[Symbol]:
    sym_list = []
    for name, sym in graph.items():
        visit(sym, sym_list, clear_list=False)
    return sym_list

def visit(graph: MultiHeadSymbol, callback: _VisitorT):
    C = config.LogConfig.G()
    for sym in topo_sort(graph):
        C.log_in_vot(callback.__name__, "<< " + sym)
        callback(sym)
        C.log_in_vot(callback.__name__, ">> " + sym)

def transform(self, callback: _TransformerT):
    sym_map: typing.Dict[str, Symbol] = {}
    C = config.LogConfig.G()
    for sym in self.topo_sort():
        # pre-clone symbol with updated args,
        # to avoid misleading usage in callback.
        args = [sym_map[c.name] for c in sym.args]
        sym = sym.copy(args=args)

        C.log_in_vot(callback.__name__, "<< " + sym)
        out = callback(sym) or sym
        assert isinstance(out, Symbol), out
        # default const_ prefix symbol means parameters
        assert sym.name not in sym_map, sym.name
        # assert sym.name.startswith("const_") or \
        #         sym.name not in sym_map, sym.name
        sym_map[sym.name] = out
        C.log_in_vot(callback.__name__, ">> " + out)
    return sym_map[symbol.name]



