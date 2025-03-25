from __future__ import annotations

import typing
import numpy as np

import tvm

from dataclasses import dataclass, field, InitVar

from .types import *
from .symbol import *
from . import runtime
from . import op, opns, inference
from .transform import Transformer

SamplingFuncT = typing.Callable[
        [typing.Union[OpNumpyT, float]], typing.Any]

@dataclass(repr=False)
class Calibrator(Transformer):
    """ skip dump, and restore from np_data. """
    nd_data: OpOutputT | None = field(repr=False, default=None)
    """ calibrate may be processed multi-times """
    data: typing.List[OpNumpyT] = field(default_factory=list)

    def _rand_data(self,
            enabled: bool = False,
            absmax: float | None = None,
    ):
        assert enabled, "symbol:{} don't have data".format(
                self.name)
        out = np.random.randn(*self.shape)
        out = out.astype(self.dtype)
        if absmax is not None:
            assert absmax > 0
            norm = np.abs(out).max()
            out = out * absmax / norm
        return tvm.nd.array(out)

    def __call__(self,
            data: tvm.nd.NDArray | None = None,
            data_dict: ParametersT = {},
            random_config: typing.Dict[str, typing.Any] = {},
            sampling_func: SamplingFuncT = None,
            **kwargs):
        kwargs.pop("origin", None)

        if self.is_input():
            out = data_dict.get(self.name, data)
            if out is None:
                out = self._rand_data(**random_config)
        elif self.is_param():
            out = self.params[self.name]
        else:
            out = inference.run(
                    self, [a.nd_data for a in self.args],
                    **kwargs)

        assert isinstance(out, (tvm.nd.NDArray, list)), type(out)
        if isinstance(out, tvm.nd.NDArray):
            self._assert(out.dtype, self.dtype)
            self._assert(out.shape, self.shape)
        else:
            self._assert([o.dtype for o in out], self.dtype)
            self._assert([o.shape for o in out], self.shape)

        self.nd_data = out
        data = to_numpy(out)
        if sampling_func is not None:
            data = sampling_func(data)
        self.data.append(data)

    def sampling(self, data):
        if isinstance(data, list):
            return max([self.sampling(d) for d in data]) \
                if data else 0
        return float(np.abs(data).max())

    def __repr__(self, **attrs):
        return super().__repr__(
            data=self.sampling(self.data), **attrs)

    def _assert(self, val, expect):
        if isinstance(val, (list, tuple)):
            assert len(val) == len(expect), (
                    "{} vs. {}").format(val, expect)
            for v, e in zip(val, expect):
                self._assert(v, e)
            return
        assert val == expect, "{} vs. {}".format(val, expect)


@dataclass(repr=False)
class Sampling(Transformer):
    @property
    def data(self) -> typing.Any:
        return self.extra_attrs.get("data", None)
    @data.setter
    def data(self, val):
        self.set_extra_attrs(data=val)

    @classmethod
    def sampling(cls, np_data: np.ndarray) -> typing.Any:
        raise NotImplementedError()

    def __call__(self, origin: Calibrator, **kw):
        if self.is_op(opns.CLIP):
            # TODO: remove clip if threshold is less than a_max
            a_min, a_max = self.parsed.a_min, self.parsed.a_max
            self.data = max(abs(a_min), abs(a_max))
        else:
            self.data = self.sampling(origin.data)
        return self

@dataclass(repr=False)
class SymmetricMinMaxSampling(Sampling):
    threshold: typing.ClassVar[float] = 1e-5

    @classmethod
    def sampling(cls, data: typing.List[OpNumpyT]) -> float:
        if isinstance(data, list):
            assert data
            return max([cls.sampling(d) for d in data])
        data = float(np.abs(data).max())
        data = 0 if data < cls.threshold else data
        #  assert data > 0
        return data


