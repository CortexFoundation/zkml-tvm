from __future__ import annotations

import typing
from functools import wraps
from dataclasses import dataclass, field

import numpy as np

import tvm

from mrt.mir.symbol import *

from mrt.mir import op, opns
from mrt.mir.attrs import _BaseAttrs, parse_attrs

from mrt.frontend.tvm.types import *
from mrt.common.utils import N

@dataclass(repr=False)
class WithParameters(Symbol):
    parsed: _BaseAttrs = field(repr=False)
    params: ParametersT = field(repr=False)
    """ Parameters should not be changed in transformer,
            use copy mode instead to avoid possible errors.

        deep copy params in trace `checkpoint_run` api.
    """

    @classmethod
    def update_dict(cls, data_dict: dict, **kwargs) -> dict:
        data_dict.update(kwargs)
        parsed = parse_attrs(
                data_dict["op_name"], data_dict["attrs"])
        return super().update_dict(data_dict, parsed=parsed)

    def __repr__(self, **attrs):
        if self.is_param():
            attrs["absmax"] = np.abs(self.numpy()).max(initial=0)
        return super().__repr__(**attrs)

    def ndarray(self) -> OpOutputT:
        return to_ndarray(self.numpy())

    def numpy(self) -> OpNumpyT:
        assert self.is_param(), f"{self.name} is not parameter."
        data = self.params[self.name]
        assert isinstance(data, (tuple, list, np.ndarray)), \
                f"param:{self.name} not OpNumpyT, get {type(data)}"
        return data

        return to_numpy(self.ndarray())

    def as_parameter(self, data: OpNumpyT):
        def _f(data, dtype):
            if isinstance(data, list):
                assert len(data) == len(dtype)
                return [_f(d, t) for d, t in zip(data, dtype)]
            assert isinstance(data, np.ndarray), type(data)
            return data.astype(dtype)

        self.params[self.name] = _f(data, self.dtype)
        return op.as_variable(self)

    def from_const_data(self, data: typing.Union[int, float]) -> WithParameters:
        return self.from_np_data(data)

    def from_np_data(self, data: np.ndarray, prefix=None) -> Symbol:
        name = N.n(prefix=prefix)
        # some data is np.float/int type, use np.array to wrap it.
        data = np.array(data)
        self.params[name] = data.astype(self.dtype)
        return op.variable(name, data.shape, self.dtype).like(self)

    def is_input(self) -> bool:
        return op.is_input(self, self.params)
    def is_param(self) -> bool:
        return op.is_param(self, self.params)
    def is_variable(self) -> bool:
        return op.is_variable(self, self.params)
    def is_operator(self) -> bool:
        return op.is_operator(self, self.params)

TransformerT = typing.Callable[[Graph], Graph]
""" Transformer Callback Function Type,
        inherited from WithParameters.
"""

@dataclass(repr=False)
class Transformer(WithParameters):
    """ Symbol Transformer """

    RUN_ONCE: typing.ClassVar[bool] =False
    """ whether to run callback once? """

    # def to_dict(self, **kwargs):
    #     """ override to dict, since transformer may want to
    #             access the previous tfm. Thus, the next
    #             update_dict has the `origin` key by default.
    #     """
    #     data = super().to_dict(**kwargs)
    #     data["extra_attrs"]["origin"] = self
    #     return data

    @classmethod
    def get_transformer(cls, name: typing.Optional[str] = None):
        name = name or cls.__name__
        def _func(graph: Symbol, params: ParametersT, **kwargs):
            def _run(sym: Symbol):
                # use current cls to apply transform, this
                #   may loss some information from origin
                #   symbol, so record as `origin` in call.
                out = cls.base(sym, params=params)
                out = out(origin=sym, **kwargs) or out
                assert isinstance(out, cls), (
                        "transform output type should be {},"
                        " but get {}"
                        ).format(cls, type(out))
                return out
            _run.__name__ = name
            with N(name):
                return _run(graph) if cls.RUN_ONCE \
                        else transform(graph, _run)
        _func.__name__ = name
        return _func

    # @classmethod
    # def apply(cls, *args, **kw):
    #     """ Static apply function to generator transformer pass.

    #     All the parameters are used to invoke `call` method.
    #     """
    #     def _tfm(sym: Symbol, params: ParametersT):
    #         ins = cls.base(sym, params=params)
    #         out = ins(*args, **kw) or ins
    #         assert isinstance(out, cls), (
    #             "expected {}, but get {}"
    #                 ).format(cls, type(out))
    #         return out

    #     _tfm.__name__ = cls.__name__
    #     return _tfm

    def __call__(self, *args, **kw) -> typing.Optional[Transformer]:
        """
            Parameters:
            origin: original symbol passed from last transformer.
        """
        raise NotImplementedError()

@dataclass(repr=False)
class RunOnce(Transformer):
    RUN_ONCE: typing.ClassVar[bool] = True

