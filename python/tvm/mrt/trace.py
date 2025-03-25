from __future__ import annotations
import typing

import os
import pickle
import numpy as np
from functools import wraps
from contextlib import redirect_stdout
from dataclasses import dataclass, field

import tvm
from tvm import relay, ir
from tvm.contrib import graph_executor as graph

from . import runtime, config
from . import op, optype, fuse, helper
from . import calibrate as calib
from . import fixed_point as fp
from . import segement as seg
from .stats import *
from .transform import Transformer, TransformerT
from .discrete import Discretor
from .precision import PrecisionRevisor
from .types import *
from .symbol import *
from .frontend.expr import symbol2expr, expr2symbol
from .dataset import Dataset

@dataclass
class Trace:
    model: str
    """ Model Name """
    name: str
    """ Trace Name """
    symbol: Symbol
    params: ParametersT

    # post init and inherit
    _force: bool = False
    _dataset: typing.optional[Dataset] = None
    _stat_type: typing.Optional[typing.Type[Statistics]] = None

    # post init and no inherit
    _sym_inputs: typing.List[Symbol] = field(init=False)
    _sym_params: typing.List[Symbol] = field(init=False)
    _executor: typing.Optional[graph.GraphModule] = None

    BASE_DIR: typing.ClassVar[str] = "./data"

    def __post_init__(self):
        """ Verify inputs and params. """
        self._sym_inputs = []
        self._sym_params = []
        def _init(sym: Symbol):
            if op.is_input(sym, self.params):
                self._sym_inputs.append(sym)
            elif op.is_param(sym, self.params):
                data = self.params[sym.name]
                assert sym.shape is None or \
                        list(sym.shape) == list(data.shape), (
                    "param:{} shape inconsistent: {} vs. {}"
                ).format(sym, sym.shape, data.shape)
                assert sym.dtype == data.dtype, (
                    "params:{} dtype inconsistent: {} vs. {}"
                ).format(sym.name, sym.dtype, data.dtype)
                self._sym_params.append(sym)
        with config.Pass():
            visit(self.symbol, _init)

        # if len(self._sym_inputs) > 1:
        #     print([str(s) for s in self._sym_inputs])
        #     assert False
        self.params = {s.name: self.params[s.name] \
                for s in self._sym_params}

    @property
    def input_names(self) -> typing.List[str]:
        return [i.name for i in self._sym_inputs]

    @property
    def input_shapes(self) -> typing.List[ShapeT]:
        return [i.attrs["shape"] for i in self._sym_inputs]

    @property
    def input_shape_dict(self) -> typing.Dict[str, ShapeT]:
        return {s.name: s.shape for s in self._sym_inputs}

    def bind_dataset(self,
            dataset: Dataset,
            stat_type: typing.Optional[typing.Type[Statistics]] = None):
        # dataset.reset()
        data, label = dataset.next()
        # verify and assert the input data
        runtime.validate_runtime_inputs(self._sym_inputs, data)

        dataset.reset()
        self._dataset = dataset
        if stat_type is not None:
            assert issubclass(stat_type, Statistics)
            self._stat_type = stat_type
        return self

    def validate_accuracy(self,
            *traces: typing.List[Trace],
            max_iter_num: int = 0,
            **kwargs):
        all_traces = [ self, ] + list(traces)
        all_stats = [t._stat_type() for t in all_traces]
        assert all([t._dataset is not None for t in all_traces]), \
                "trace databset not binded."
        assert all([t._stat_type is not None for t in all_traces]), \
                "trace statistic not binded."

        log_str = "Iteration: {:3d} | "
        for t in all_traces:
            log_str += t.name + ": {} | "

        for i in range(max_iter_num or 99999999999999):
            dls = [t._dataset.next() for t in all_traces]
            if any([dl is None for dl in dls]):
                break
            for t, (data, label), stat in zip(
                    all_traces, dls, all_stats):
                out = t.eval(data, **kwargs)
                stat.merge((out, label))
            msg = log_str.format(i, *[s.info() for s in all_stats])
            print(msg)
        print("Trace Accuracy Eval Done!")

    def eval(self,
            data: typing.Optional[np.ndarray] = None,
            **kwargs,) -> np.ndarray:
        if self._executor is None:
            self._executor = runtime.create_executor(
                    symbol2expr(self.symbol, self.params),
                    self.params, **kwargs)

        data = runtime.validate_runtime_inputs(self._sym_inputs, data)
        res = runtime.run_executor(self._executor, data)
        assert len(res) == 1
        return res[0]

    def _new(self, tr_name: str,
            symbol: Symbol, params: ParametersT) -> Trace:
        return Trace(self.model, tr_name,
                symbol, params,
                _force = self._force,
                _dataset = self._dataset,
                _stat_type = self._stat_type)

    def checkpoint_run(self,
            *callbacks: typing.List[TransformerT],
            tr_name: typing.Optional[str] = None,
            force: bool = False,
            **kwargs) -> Trace:
        self._force = self._force or force

        assert len(callbacks) > 0
        tr_name = tr_name or callbacks[-1].__name__
        tr_path = self._get_checkpoint_path(tr_name)
        if path.exists(tr_path) and not self._force:
            out = Trace.load(tr_path)
            return self._new(tr_name, out.symbol, out.params)

        out: Trace = self
        for cb in callbacks:
            # deep copy params to avoid conflict status
            params = {k: v for k, v in out.params.items()}
            print("Apply Trace: {:25} Transformer: {}".format(
                tr_name, cb.__name__))
            symbol = cb(out.symbol, params, **kwargs)
            out = out._new(tr_name, symbol, params)
        out.dump(tr_path)
        return out

    def discrete(
            self,
            calibrate_repeats: int = 1,
            calibrate_sampling: calib.SamplingFuncT = None,
            force: bool = False) -> Trace:
        fuse_tr = self.fuse(force=force)
        seg_tr = fuse_tr.checkpoint_run(seg.Spliter.get_transformer())

        calib_tr = seg_tr.calibrate(
                repeats=calibrate_repeats,
                sampling_func=calibrate_sampling)
        quant_tr = calib_tr.quantize()
        quant_tr = quant_tr.checkpoint_run(
                seg.Merger.get_transformer(),
                spliter=seg_tr.symbol)
        return quant_tr

    def fuse(self, **kwargs) -> Trace:
        kwargs.setdefault("tr_name", "fuse")
        return self.checkpoint_run(
                fuse.FuseConstant.get_transformer(),
                fuse.FuseTupleGetItem.get_transformer(),
                fuse.FuseBatchNorm.get_transformer(),
                fuse.FuseLeakyReLU.get_transformer(),
                fuse.FuseDivide.get_transformer(),
                fuse.FuseAvgPool2D.get_transformer(),
                fuse.FuseDropout.get_transformer(),
                fuse.FuseMean.get_transformer(),
                fuse.FuseNaiveSoftmax.get_transformer(),
                fuse.FuseConstant.get_transformer(),
                **kwargs,
                )

    def calibrate(self, repeats: int = 1, **kwargs) -> Trace:
        assert self._dataset is not None
        tr_name = kwargs.pop("tr_name", "calibrate")
        out = self
        for i in range(repeats):
            data, _ = self._dataset.next()
            out = out.checkpoint_run(
                    calib.Calibrator.get_transformer(),
                    data = tvm.nd.array(data),
                    #  tr_name = tr_name,
                    tr_name = "%s_run_%d"%(tr_name, i),
                    **kwargs)
        out = out.checkpoint_run(
                calib.SymmetricMinMaxSampling.get_transformer(),
                tr_name = "%s_sampling" % tr_name)
        return out

    def quantize(self, **kwargs):
        kwargs.setdefault("tr_name", "quantize")
        return self.checkpoint_run(
                Discretor.get_transformer(),
                fuse.FuseConstant.get_transformer(),
                PrecisionRevisor.get_transformer(),
                **kwargs)

    def export(self, target: str, use_simulator: bool = True, **kwargs):
        assert target in ["sim-clip-round", "sim-clip", "sim-round", "sim", "circom", ]
        kwargs.setdefault("tr_name", target)

        if "sim" in target:
            return self.checkpoint_run(
                    fp.Simulator.get_transformer(),
                    with_clip = "clip" in target,
                    with_round = "round" in target,
                    **kwargs)
        elif "circom" in target:
            return self.checkpoint_run(
                    fp.FixPoint.get_transformer(), **kwargs)
        elif "cvm" in target:
            pass

        raise RuntimeError("Not Implemented Trace Target: " + target)

    def print(self, **kwargs):
        helper.format_print(
                self.symbol, self.params, self.name, **kwargs)

    def log(self, **kwargs):
        fname = self._get_checkpoint_path(self.name) + ".log"
        print("Log   Trace {:20} into {}".format(
            self.name, fname))
        with open(fname, "w") as f:
            with redirect_stdout(f):
                self.print(**kwargs)
        return self

    def subgraph(self, inames=[], onames=[]) -> Trace:
        out = op.subgraph(self.symbol, inames, onames)
        return self._new("subgraph", out, self.params)

    def _get_checkpoint_path(self, tr_name: str = None):
        base_dir = os.path.join(self.BASE_DIR, self.model)
        os.makedirs(base_dir, exist_ok=True)

        tr_name = tr_name or self.name
        return os.path.join(base_dir, tr_name + ".trace")

    def dump(self, tr_path: str = None):
        tr_path = tr_path or self._get_checkpoint_path()
        print("Dump  Trace {:20} into {}".format(self.name, tr_path))
        data = dump_json(self.symbol)
        data.update({
            "_model_name": self.model,
            "_trace_name": self.name,
            "params": {k: v.numpy() \
                    for k, v in self.params.items()},
        })
        try:
            with open(tr_path, "wb") as f:
                pickle.dump(data, f)
        except Exception as e:
            # clean generated empty path
            os.remove(tr_path)
            raise e

    @staticmethod
    def load(tr_path: str) -> Trace:
        with open(tr_path, "rb") as f:
            data = pickle.load(f)

        model  = data["_model_name"]
        name = data["_trace_name"]
        params = {k: tvm.nd.array(v) \
                for k, v in data["params"].items()}
        symbol = load_json(data, params=params)
        print("Load Trace {:20} from {}".format(name, tr_path))
        return Trace(model, name, symbol, params)

    @staticmethod
    def from_expr(
            expr: RelayExpr, params: ParametersT,
            tr_name = "from_expr",
            model_name="unknown-model") -> Trace:
        print("Init  Trace {:20} from model {}'s expr".format(
            tr_name, model_name))
        symbol, params = expr2symbol(expr, params)
        return Trace(model_name, tr_name, symbol, params)


