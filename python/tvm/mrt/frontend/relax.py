import json

import tvm
from tvm.script import relax as R
from tvm import relax, ir, tir
from tvm.relax.expr import *
from tvm.runtime import _ffi_node_api

from ..opns import *
from ..symbol import *
from ..types import *
from .. import op

__ALL__ = [ "expr2symbol", "symbol2expr", "tvm_type_infer" ]

NamedParametersT = typing.Dict[str, R.Tensor]

def _convert_to_py(value):
    """ ShapeVar not operated yet. """
    if isinstance(value, ShapeExpr):
        return _convert_to_py(value.values)
    elif isinstance(value, PrimValue):
        return _convert_to_py(value.value)
    elif isinstance(value, ir.container.Array):
        return [ _convert_to_py(v) for v in value ]
    elif isinstance(value, (
        tir.expr.IntImm, tir.expr.FloatImm, tir.expr.StringImm)):
        return value.value
    elif isinstance(value, (str, int, float)):
        return value
    elif value is None:
        return value
    elif isinstance(value, tir.expr.Var):
        return value.name
    else:
        print(">>> unknown type:", type(value), value)
    return value

def _list_node_attrs_names(obj):
    fnames = _ffi_node_api.NodeListAttrNames(obj)
    size = fnames(-1)
    return sorted([fnames(i) for i in range(size)])

def _struct_info(info: relax.StructInfo, key):
    if isinstance(info, relax.struct_info.TupleStructInfo):
        return [_struct_info(f, key) for f in info.fields]
    #  return getattr(info, key)
    val = _convert_to_py(getattr(info, key))
    return val

def expr2symbol(
        expr: Expr,
        params: ParametersT = {},
        ):
    params = {k: v for k, v in params.items()}

    symbol_map = {}
    binding_info: typing.List[VarBinding] = []
    def _cast_relax(node: Expr):
        if node in symbol_map:
            return

        if isinstance(node, ShapeExpr):
            return

        tvm.ir.RelaxExpr
        tvm.relax.StructInfo
        tvm.relax.expr.ShapeExpr
        try:
            dtype = _struct_info(node.struct_info, "dtype")
        except Exception as e:
            # print(type(node))
            dtype = None

        try:
            shape = _struct_info(node.struct_info, "shape")
        except Exception as e:
            shape = None

        attrs = { "extra_attrs": { "shape": shape, "dtype": dtype }, }
        tvm.relax.struct_info.TupleStructInfo
        #  print(dtype, shape, type(node.struct_info.fields[0]))
        #  try:
        #      print(type(node), node.checked_type.script())
        #  except:
        #      pass

        tvm.script.relax.Tensor
        print(type(node), str(node).replace("\n", "")[:30] + "...")
        if isinstance(node, ( PrimValue, )):
            name = N.n("const_")
            params[name] = _convert_to_py(node)
            out = op.variable(name, [], dtype)
            #  print(node.struct_info, params[name], out)
            #  sys.exit()
            #  out = _convert_to_py(node)
        elif isinstance(node, ir.op.Op):
            out = node.name
        elif isinstance(node, ir.expr.GlobalVar):
            out = node.name_hint
        elif isinstance(node, SeqExpr):
            for b in node.blocks:
                for vb in b.bindings:
                    assert isinstance(vb, VarBinding), vb
                    _cast_relax(vb.var)
                    binding_info.append(vb)
                    #  binding_map[symbol_map[vb.value]] = symbol_map[vb.var]
            out = symbol_map[node.body]
        elif isinstance(node, Var):
            # tvm.relax.expr.DataflowVar
            name = node.name_hint or N.n(prefix="input_")
            out = op.variable(name, shape, dtype)
        elif isinstance(node, TupleGetItem):
            args = [ symbol_map[node.tuple_value], ]
            attrs['index'] = node.index
            out = op._new_op(TUPLE_GET_ITEM, *args, **attrs)
        elif isinstance(node, Tuple):
            args = [ symbol_map[f] for f in node.fields ]
            out = op._new_op(TUPLE, *args, **attrs)
        elif isinstance(node, Call):
            if node.attrs is not None:
                attr_names = _list_node_attrs_names(node.attrs)
                attrs.update({k: _convert_to_py(
                    getattr(node.attrs, k)) for k in attr_names})

            op_name = node.op.name
            if op_name.startswith("relax."):
                op_name = op_name[6:]

            if op_name in [CONCAT, ADV_INDEX]:
                args = [symbol_map[f] for f in node.args[0].fields]
            elif op_name in [RESHAPE]:
                args = [ symbol_map[node.args[0]] ]
                attrs['shape'] = _convert_to_py(node.args[1])
            else:
                args = [symbol_map[i] for i in node.args]

            # op:arange has duplicate attrs for (start, stop, step)
            if op_name in [ ARANGE, ]:
                for k in ["start", "stop", "step"]:
                    attrs.pop(k)
            elif op_name == BROADCAST_TO:
                attrs.pop("dtype")
            elif op_name == GET_VALID_COUNT:
                attrs.pop("score_threshold")
            elif op_name in [ CALL_TIR, CALL_DPS_PACKED, ]:
                attrs["func_name"] = args.pop(0)
            out = op._new_op(op_name, *args, **attrs)

            #  if "call_dps_packed" in op_name:
            #      print([str(a) for a in args])
            #      print(node.attrs)
            #      print(node.sinfo_args)
            #      sys.exit()
        elif isinstance(node, ExternFunc):
            out = str(node.global_symbol)
        else:
            print("unsupported expr:", node)
            tvm.ir.expr.GlobalVar
            tvm.relax.expr.ExternFunc
            tvm.relax.expr.TupleGetItem
            sys.exit()

        print("=>", out)
        assert out is not None
        symbol_map[node] = out

    with N():
        relax.analysis.post_order_visit(expr, _cast_relax)


    print(type(expr))
    binding_map = {}
    for vb in binding_info:
        print(vb.var.name_hint, vb.value)
        # change op output into binding var name
        symbol_map[vb.value].name = vb.var.name_hint
        binding_map[symbol_map[vb.var]] = symbol_map[vb.value]

    # maybe multi original expr points to new symbol, so we need to 
    #   scan all symbol_map to update value.
    for k, v in symbol_map.items():
        if not isinstance(v, Symbol):
            continue
        v = binding_map.get(v, v)
        v.args = [binding_map.get(a, a) for a in v.args]
        symbol_map[k] = v

    print("out: ", expr.body, symbol_map[expr])
    with open("/tmp/relax_out.log", "w") as f:
        f.write(raw_log(symbol_map[expr]))

def symbol2expr(
        symbol: Symbol,
        params: ParametersT = {},
        ):
    pass
