import typing
from ..symbol import *

import cvm

op_dict = {
    "sum": "sum",
    "squeeze": "squeeze",
    "nn.conv2d": "conv2d",
    "nn.dense": "dense",
    "nn.max_pool2d": "max_pool2d",
    "mrt.rs_pclip": "cvm_right_shift",
    "right_shift": "cvm_right_shift",
    "multiply": "broadcast_mul",
    "mrt.pclip": "cvm_clip",
    "add": "elemwise_add",
    "reshape": "reshape",
    "nn.relu": "relu"
}

def get_node(nodes: list, node_name: str) -> list:
    for index in range(len(nodes)):
        if nodes[index]["name"] == node_name:
            return [index, 0, 0]
    return []

def get_cvm_op(op_name: str) -> typing.Type[cvm.symbol.Symbol]:
    op = getattr(cvm.symbol, op_name, None)

    if op is None:
        raise RuntimeError("cvm not register operator: {}".format(op_name))
    return op

def to_cvm(symbol: Symbol, params: ParametersT, cvm_params: dict) -> dict:
    cvm_data = {}

    cvm_nodes = []
    cvm_arg_nodes = []
    cvm_node_row_ptr = []
    cvm_attrs = {
        "op_attrs": ["list_str", []],
        "dltype": ["list_str", []],
        "precision": ["list_int", []],
        "storage_id": ["list_int", []],
        "shape": ["list_shape", []]
    }
    sym_list = sym2list(symbol)
    # transform symbol args
    for i in range(len(sym_list)):
        if sym_list[i].op_name != "var" and sym_list[i].args[0].op_name == "cast":
            sym_list[i].args[0] = sym_list[i].args[0].args[0]
        if sym_list[i].op_name == "mrt.rs_pclip":
            shift_bit = int(params[sym_list[i].args[1].name].asnumpy())
            if int(params[sym_list[i].args[1].name].asnumpy()) > 32:
                shift_bit = 32
            sym_list[i].attrs["shift_bit"] = shift_bit
            del(cvm_params[sym_list[i].args[1].name])
            del(sym_list[i].args[1])
        if sym_list[i].op_name == "right_shift":
            shift_bit = int(params[sym_list[i].args[1].name].asnumpy())
            sym_list[i].attrs["shift_bit"] = shift_bit
            sym_list[i].attrs["precision"] = sym_list[i].extra_attrs["precision"]
            del(cvm_params[sym_list[i].args[1].name])
            del(sym_list[i].args[1])
        if sym_list[i].op_name == "nn.bias_add":
            sym_list[i].op_name = sym_list[i].args[0].args[0].op_name
            sym_list[i].attrs = sym_list[i].args[0].args[0].attrs
            sym_list[i].args.append(sym_list[i].args[1])
            sym_list[i].args[1] = sym_list[i].args[0].args[0].args[1]
            sym_list[i].args[0] = sym_list[i].args[0].args[0].args[0]
        if sym_list[i].op_name == "multiply":
            sym_list[i - 1].extra_attrs["precision"] = 16
    # symbol topo sort
    index = 0
    for sym in sym2list(symbol):
        if sym.op_name == "nn.conv2d":
            sym.attrs["layout"] = sym.attrs["data_layout"]
            del(sym.attrs["data_layout"])
            sym.attrs["out_dtype"] = "int32"
            del(sym.attrs["out_layout"])
            del(sym.attrs["padding"][2:])
        elif sym.op_name == "nn.max_pool2d":
            del(sym.attrs["out_layout"])
            del(sym.attrs["dilation"])
            del(sym.attrs["padding"][2:])
        elif sym.op_name == "reshape":
            del(sym.attrs["allowzero"])
            sym.attrs["shape"] = sym.attrs["newshape"]
            del(sym.attrs["newshape"])
        elif sym.op_name == "nn.dense":
            del(sym.attrs["out_dtype"])
            sym.attrs["units"] = sym.extra_attrs["shape"][1]
        for key in sym.attrs:
            sym.attrs[key] = str(sym.attrs[key])
        cvm_attrs["op_attrs"][1].append(json.dumps(sym.attrs))
        cvm_attrs["dltype"][1].append("int32")
        try:
            precision = sym.extra_attrs["precision"]
        except KeyError:
            precision = -1
        cvm_attrs["precision"][1].append(precision)
        shape = sym.extra_attrs["shape"]
        shape = list(shape)
        if shape == []:
            shape = [1]
        cvm_attrs["shape"][1].append(shape)
        cvm_attrs["storage_id"][1].append(index)
        op = sym.op_name
        inputs: typing.List = []
        args = sym.args
        if op == "var":
            op = "null"
            cvm_arg_nodes.append(index)
            cvm_nodes.append({
                "op": op,
                "name": sym.name,
                "inputs": inputs
                })
        else:
            for arg in args:
                inputs.append(get_node(cvm_nodes, arg.name))
            attrs = {}
            attrs["flatten_data"] = "0"
            attrs["func_name"] = op_dict[op]
            attrs["num_inputs"] = str(len(args))
            attrs["num_outputs"] = "1"
            op = "cvm_op"
            cvm_nodes.append({
                "op": op,
                "name": sym.name,
                "attrs": attrs,
                "inputs": inputs
                })
        cvm_node_row_ptr.append(index)
        index = index + 1
    cvm_nodes[0]["name"] = "data"
    cvm_node_row_ptr.append(index)
    cvm_data["nodes"] = cvm_nodes
    cvm_data["arg_nodes"] = cvm_arg_nodes
    cvm_data["node_row_ptr"] = cvm_node_row_ptr
    cvm_data["heads"] = [[index-1, 0, 0]]
    cvm_data["attrs"] = cvm_attrs
    return {"symbol": cvm_data, "params": cvm_params}

