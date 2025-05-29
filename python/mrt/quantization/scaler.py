from __future__ import annotations

import typing
from dataclasses import dataclass, field

from mrt.mir import op
from mrt.mir.opns import *
from mrt.mir.symbol import *

@dataclass(repr=False)
class WithScale(Symbol):
    @classmethod
    def _validate_scale(cls, scale, msg=None):
        if isinstance(scale, (list, tuple)):
            return [cls._validate_scale(s, msg) for s in scale]
        assert isinstance(scale, float), scale
        assert scale >= 0, ("scale: {} invalid for \n{}").format(
                scale, msg or str(cls))

    def __repr__(self, **attrs):
        return super().__repr__(scale=self.scale)

    @property
    def scale(self):
        return self.extra_attrs.get("scale", -1)

    @scale.setter
    def scale(self, val):
        self._validate_scale(val)
        self.set_extra_attrs(scale=val)

    @classmethod
    def _scale_defined(cls, scale):
        if isinstance(scale, (list, tuple)):
            return all([cls._scale_defined(s) for s in scale])
        return scale >= 0

    @property
    def scale_defined(self) -> bool:
        return self._scale_defined(self.scale)

ScaleRulesT = typing.Callable[[WithScale], typing.Any]
INFER_SCALE_RULES: typing.Dict[str, ScaleRulesT] = {}

def register_scale_rules(*op_names, rule: ScaleRulesT = None):
    assert rule is not None
    for op in op_names:
        INFER_SCALE_RULES[op] = rule

def scale_rules(*op_names):
    def _add_rules(f: ScaleRulesT):
        for op in op_names:
            INFER_SCALE_RULES[op] = f
        return f
    return _add_rules

def scale_index(s: WithScale, index: int):
    return s.args[index].scale

def scale_nn(s: WithScale):
    return s.args[0].scale * s.args[1].scale

def scale_identity(s: WithScale):
    return s.args[0].scale

def infer_scale(symbol: WithScale):
    def _infer(sym: Symbol):
        if op.is_variable(sym):
            return
        if sym.op_name not in INFER_SCALE_RULES:
            return

        if isinstance(sym, WithScale):
            sym.scale = INFER_SCALE_RULES[sym.op_name](sym)
        #  if op.is_variable(sym):
        #      assert sym.scale_defined, ("var: %s cannot deduct scale"
        #              ) % sym.name
        #      return
        #  assert sym.op_name in _INFER_SCALE_RULES, (
        #          "infer scale not support for op:%s"
        #          ) % sym.op_name
        #  sym.scale = _INFER_SCALE_RULES[sym.op_name](sym)
        return sym
    out: WithScale = transform(symbol, _infer)
    assert isinstance(out, WithScale), out
    assert out.scale_defined, ("op: %s cannot deduct scale") % out.name
    return out.scale
