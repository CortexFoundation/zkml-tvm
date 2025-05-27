from __future__ import annotations

import typing
from dataclasses import dataclass, field

from . import op
from .discrete import *
from .precision import *
from .transform import Transformer
from .annotate import ArgAnnotator

__ALL__ = [ "Quantizer" ]

@dataclass(repr=False)
class Quantizer(QuantizedInfo, Transformer):
    """ MRT quantization class.

        Lose the discretor information if dump.
    """
    args: typing.List[Quantizer]
    revised: Quantizer | None = field(repr=False)
    requants: typing.Dict[str, Quantizer] = field(repr=False)

    @classmethod
    def default_dict(cls, **kwargs) -> dict:
        kwargs.setdefault("revised", None)
        kwargs.setdefault("requants", {})
        return super().default_dict(**kwargs)

    @property
    def discretor(self) -> Discretor:
        dt_type: Discretor = eval(self.dt_type)
        return dt_type.base(self,
            args=[dt_type.base(a) for a in self.args])

    def assign(self, other: Discretor) -> Quantizer:
        return self.set_extra_attrs(**other.extra_attrs)

    def __call__(self):
        if self.is_variable():
            return self

        arg_dts = ArgAnnotator.bind(self.discretor)
        for i in range(len(self.args)):
            arg: Quantizer = self.args[i]
            arg = arg.revised or arg
            self.args[i] = arg.requantize(arg_dts[i])

        self.set_extra_attrs(
            dt_info = InferDiscretor.bind(self),
            precision = InferPrecision.bind(self),)
        self.examine_precision(self.discretor)
        return InferOperator.bind(self).like(self)

    def examine_precision(self, dt: Discretor):
        """ set revised target with discretor examine.
                use explicit clip to annotate precision
                if necessary.
        """
        dt.examine()
        out = self
        if self.precision > dt.precision:
            out = op.pclip(out, precision=dt.precision)
        self.revised = out.like(self).assign(dt)
        # raw_print(self.revised)

    def requantize(self, dt: Discretor):
        dt.examine()
        key = dt.summary()
        if key in self.requants:
            return self.requants[key]

        out: Quantizer = self.copy(name=N.n())
        if self.is_input():
            pass
        elif self.is_param():
            out.update_data(dt.mapping(self.numpy()))
        else:
            out = dt.remapping(self.discretor, self)
        out = out.like(self).assign(dt)
        # out.is_op(REQUANT) and print("[  Requant]>> {}".format(out))
        # print("[  Requant]>> {}".format(out))
        self.requants[key] = out
        return out

