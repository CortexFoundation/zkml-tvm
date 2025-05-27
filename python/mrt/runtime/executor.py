import typing

import numpy as np
from collections import namedtuple

import tvm
# import logging
from tvm import relax
from tvm import relay, ir, runtime
#  from tvm.contrib import graph_executor
#  from tvm.ir import RelayExpr

from mrt.frontend.tvm.types import *

from mrt.mir.symbol import Symbol
from mrt.dataset.base import Dataset

# from .types import *
#  from .symbol import Symbol
#  from .dataset import Dataset
from .analysis import Statistics

__all__ = [ "create_executor", "run_executor", "infer"]

#  logger = logging.getLogger("runtime")

#  def validate_runtime_inputs(
#          sym_inputs: typing.List[Symbol],
#          data: typing.Optional[np.ndarray] = None,
#          data_dict: ParametersT = {}) -> ParametersT:
#      input_dict = {}
#      for sym in sym_inputs:
#          val = data_dict.get(sym.name, data)
#          assert val is not None
#          val = tvm.nd.array(val)
#          assert list(sym.shape) == list(val.shape), (
#                  "{}: {} vs. {}"
#                  ).format(sym.name, sym.shape, val.shape)
#          assert sym.dtype == val.dtype, (
#                  "{} vs. {}").format(sym.dtype, val.dtype)
#          input_dict[sym.name] = val
#      return input_dict

InputInfo = namedtuple("InputInfo", ["index", "shape", "dtype"])
Executor = namedtuple("Executor",
                      ["vm", "dev_params", "input_info", "device"])

def create_executor(
        mod: tvm.IRModule, params: ParametersT,
        device: tvm.runtime.Device = tvm.runtime.cpu(),
        target: tvm.target.Target = tvm.target.arm_cpu(),
        opt_pass: typing.Optional[tvm.transform.Pass] = None,
        ):
    if opt_pass is not None:
        mod = opt_pass(mod)
    ex = tvm.compile(mod, target=target)
    vm = relax.VirtualMachine(ex, device)

    func: relax.Function = mod["main"]
    dev_params = [None] * len(func.params)
    input_info = {}
    for i, var in enumerate(func.params):
        if var.name_hint in params:
            dev_params[i] = tvm.nd.array(
                    params[var.name_hint], device)
        else:
            shape = get_struct_shape(var.struct_info)
            dtype = get_struct_dtype(var.struct_info)
            input_info[var] = InputInfo(i, shape, dtype)
    return Executor(vm, dev_params, input_info, device)

def run_executor(
        executor: Executor,
        data: typing.Optional[np.ndarray] = None,
        data_dict: ParametersT = {},
        ) -> typing.List[np.ndarray]:
    (vm, dev_params, input_info, device) = executor
    for k, v in input_info.items():
        (index, shape, dtype) = v
        val = data_dict.get(k, data)
        if val is not None:
            dev_params[index] = tvm.nd.array(val, device)
        val: tvm.nd.NDArray = dev_params[index]
        assert val is not None, f"input:{k} is empty"
        # batch dim should also matched.
        assert list(shape) == list(val.shape), \
                f"{k} shape not matched: {shape} vs. {val.shape}"
        assert dtype == val.dtype, \
                f"{k} dtype not matched: {dtype} vs. {val.dtype}"

    out = vm["main"](*dev_params)
    assert isinstance(out, (list, tvm.ir.container.Array)), type(out)
    return to_numpy(out)

def infer(mod: tvm.IRModule, params: ParametersT,
          data: typing.Optional[np.ndarray] = None,
          data_dict: ParametersT = {},
          device: tvm.runtime.Device = tvm.runtime.cpu(),
          **kwargs) -> OpNumpyT:
    executor = create_executor(mod, params, device=device, **kwargs)
    out = run_executor(executor, data, data_dict)

    #  print("infer:", type(out))

    if len(out) == 1:
        out = out[0]
    return out


# def create_executor(
#         expr: RelayExpr, params: ParametersT,
#         device: tvm.runtime.Device = tvm.runtime.cpu(),
#         target: tvm.target.Target = tvm.target.arm_cpu(),
#         opt_level=0,
# ) -> graph_executor.GraphModule:
#     with tvm.transform.PassContext(opt_level=opt_level):
#         lib = relay.build_module.build(
#                 ir.IRModule.from_expr(expr),
#                 target=target, params=params)

#     rt_mod: graph_executor.GraphModule = \
#             graph_executor.GraphModule(lib["default"](device))
#     return rt_mod

# def run_executor(
#         rt_mod: graph_executor.GraphModule,
#         input_dict: ParametersT,
#         ) -> typing.List[np.ndarray]:
#     # for n, d in input_dict.items():
#     #     print("executor input: ", n, np.abs(d.numpy()).max())
#     rt_mod.run(**input_dict)
#     return [ rt_mod.get_output(i).numpy() \
#             for i in range(rt_mod.get_num_outputs())]

#  def infer(expr: RelayExpr, params: ParametersT,
#          **kwargs) -> OpOutputT:
#      """
#          @param device: tvm.runtime.cpu() | None
#          @param target: tvm.target.arm_cpu() | "llvm"
#      """
#      result = tvm.relay.create_executor(
#          "graph", mod=ir.IRModule.from_expr(expr),
#          **kwargs).evaluate()(**params)
#      return result

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

