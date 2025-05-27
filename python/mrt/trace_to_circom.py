from __future__ import annotations

import numpy as np
from dataclasses import dataclass

from .zkml import circom, transformer as circom_transformer, model as ZkmlModel
from .trace import Trace

@dataclass(repr=False)
class CircomTfm():
    trace: Trace = None

    def __init__(self, trace:Trace):
        self.trace = trace

    #def run(self, symbol_, params_, output_name=None, **kw):
    def run(self, output_name=None, **kw):
        symbol_, params_ = self.trace.symbol, self.trace.params

        # resize batch
        symbol, params = ZkmlModel.resize_batch(symbol_, params_)

        # >>> change_symbol_name ...
        #ZkmlModel.simple_raw_print(symbol, params)
        symbol, params = circom_transformer.change_name(symbol, params)

        # set input as params
        symbol_first = ZkmlModel.visit_first(symbol)

        # >>> start model2circom ...
        outBind = circom_transformer.model2circom(symbol, params)
        outcircomobj = outBind[0]
        outcircommap = outBind[1]

        # >>> Generating circom code ...
        code = circom.generate(outcircomobj)
        input_json = circom_transformer.input_json(outcircommap , params)

        # >>> Generated, dumping code ...
        output_name = output_name or "circom_model_test"
        #  print(code)

        with open(output_name + ".circom", "w") as f:
            f.write(code)

        with open(output_name + ".json", "w") as f:
            import json
            f.write(json.dumps(input_json, indent=2))

        return symbol

