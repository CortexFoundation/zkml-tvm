from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

import tvm
from tvm import relay, ir

from .extool import *
from .types import *
from . import runtime, topi

VisitorT = typing.Callable[ [RelayExpr, ParametersT], None ]
TransformerT = typing.Callable[
        [RelayExpr, ParametersT], typing.Optional[RelayExpr]]

@dataclass
class Trace:
    name: str
    """ Trace Name """
    expr: RelayExpr
    params: ParametersT

    input_vars: typing.List[Var] = field(init=False)
    param_vars: typing.List[Var] = field(init=False)

    def __post_init__(self):
        self.input_vars = []
        self.param_vars = []

        for v in relay.analysis.free_vars(self.expr):
            if v.name_hint in self.params:
                self.param_vars.append(v)
            else:
                self.input_vars.append(v)

    @property
    def input_names(self) -> typing.List[str]:
        return [i.name for i in self.input_vars]

    def random_inputs(self) -> typing.Dict[str, np.ndarray]:
        inputs = {}
        for v in self.input_vars:
            shape = v.type_annotation.concrete_shape
            dtype = v.type_annotation.dtype
            data = np.random.randn(*shape).astype(dtype)
            inputs[v.name_hint] = data
        return inputs

    def calibrate(self,
            data: typing.Optional[np.ndarray] = None,
            data_dict: typing.Dict[str, np.ndarray] = {},
        ) -> typing.Dict[str, np.ndarray]:
        self.infer_type()

        calibrate_outputs: typing.Dict[str, np.ndarray] = {
                k: v.numpy() for k, v in params.items()}

        # set input data
        for v in self.input_vars:
            shape = v.type_annotation.concrete_shape
            dtype = v.type_annotation.dtype
            val = data_dict.get(v.name_hint, data)
            if val is None:
                print("input: {} use random data".format(
                    v.name_hint))
                val = np.random.randn(*shape).astype(dtype)
            calibrate_outputs[v.name_hint] = val

        def _calibrate(expr: RelayExpr, params: ParametersT):
            data = [ calibrate_outputs[e] for e in args(expr) ]
            out = topi.execute(expr, data)
            shape = list(expr.checked_type.concrete_shape)
            assert list(out.shape) == shape
            assert str(out.dtype) == expr.checked_type.dtype
            calibrate_outputs[expr] = out
        self.visit(_calibrate)
        return calibrate_outputs

    def run(self,
            data: typing.Optional[np.ndarray] = None,
            data_dict: typing.Dict[str, np.ndarray] = {},
            device: tvm.runtime.Device = tvm.runtime.cpu(0),
    ) -> typing.List[np.ndarray]:
        inputs = {k: v for k, v in self.params.items()}
        for v in self.input_vars:
            shape = v.type_annotation.concrete_shape
            dtype = v.type_annotation.dtype
            val = data_dict.get(v.name_hint, data)
            assert val is not None
            assert list(shape) == list(val.shape), (
                    "{}: {} vs. {}").format(
                            v.name_hint, shape, val.shape)
            assert dtype == val.dtype
            inputs[v.name_hint] = val
        return runtime.infer(self.expr, inputs)

    def eval(self, device) -> runtime.ValidateFunctionT:
        return runtime.validator(
                self.expr, self.params, self.name,
                device=device)

    def visit(self, callback: VisitorT):
        def _visitor(expr: RelayExpr):
            callback(expr, self.params)
        visit(self.expr, _visitor)

    def transform(self, callback: TransformerT) -> Trace:
        def _tfm(expr: RelayExpr):
            return callback(expr, self.params)
        return Trace(callback.__name__,
                transform(self.expr, _tfm), self.params)

    def infer_type(self) -> Model:
        return Trace("infer_type",
                infer_type(self.expr), self.params)


