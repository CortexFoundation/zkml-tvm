import typing

import numpy as np

import tvm
from tvm import relay, ir, runtime
from tvm.contrib import graph_executor
from tvm.ir import RelayExpr

from .types import *
from .symbol import Symbol
from .dataset import Dataset
from .stats import Statistics

__all__ = ["infer"]

def validate_runtime_inputs(
        sym_inputs: typing.List[Symbol],
        data: typing.Optional[np.ndarray] = None,
        data_dict: ParametersT = {}) -> ParametersT:
    input_dict = {}
    for sym in sym_inputs:
        val = data_dict.get(sym.name, data)
        assert val is not None
        val = tvm.nd.array(val)
        assert list(sym.shape) == list(val.shape), (
                "{}: {} vs. {}"
                ).format(sym.name, sym.shape, val.shape)
        assert sym.dtype == val.dtype, (
                "{} vs. {}").format(sym.dtype, val.dtype)
        input_dict[sym.name] = val
    return input_dict

def create_executor(
        expr: RelayExpr, params: ParametersT,
        device: tvm.runtime.Device = tvm.runtime.cpu(),
        target: tvm.target.Target = tvm.target.arm_cpu(),
        opt_level=0,
) -> graph_executor.GraphModule:
    with tvm.transform.PassContext(opt_level=opt_level):
        lib = relay.build_module.build(
                ir.IRModule.from_expr(expr),
                target=target, params=params)

    rt_mod: graph_executor.GraphModule = \
            graph_executor.GraphModule(lib["default"](device))
    return rt_mod

def run_executor(
        rt_mod: graph_executor.GraphModule,
        input_dict: ParametersT,
        ) -> typing.List[np.ndarray]:
    # for n, d in input_dict.items():
    #     print("executor input: ", n, np.abs(d.numpy()).max())
    rt_mod.run(**input_dict)
    return [ rt_mod.get_output(i).numpy() \
            for i in range(rt_mod.get_num_outputs())]

def infer(expr: RelayExpr, params: ParametersT,
        **kwargs) -> OpOutputT:
    """
        @param device: tvm.runtime.cpu() | None
        @param target: tvm.target.arm_cpu() | "llvm"
    """
    result = tvm.relay.create_executor(
        "graph", mod=ir.IRModule.from_expr(expr),
        **kwargs).evaluate()(**params)
    return result

def as_numpy(res) -> typing.List[tvm.nd.NDArray]:
    if isinstance(res, tvm.nd.NDArray):
        return [ res.numpy(), ]
    else:
        return [ o.numpy() for o in res ]


ValidateFunctionT = typing.Callable[[np.ndarray], np.ndarray]

def multiple_validate(
        base_func: ValidateFunctionT,
        *comp_funcs: typing.List[ValidateFunctionT],
        dataset: Dataset = None,
        stats_type: typing.Type[Statistics] = None,
        max_iter_num: typing.Optional[int] = None,
):
    assert dataset is not None
    assert stats_type is not None

    all_funcs = [ base_func, ] + list(comp_funcs)
    all_stats = [stats_type() for _ in all_funcs]

    log_str = "Iteration: {:3d} | "
    for func in all_funcs:
        log_str += func.__name__ + ": {} | "
    for i in range(max_iter_num or 99999999999999):
        dl = dataset.next()
        if dl is None:
            break
        for func, stats in zip(all_funcs, all_stats):
            out = func(dl[0])
            stats.merge((out, dl[1]))
        msg = log_str.format(i, *[s.info() for s in all_stats])
        print(msg)

    print("Multiple Validation Done!")

