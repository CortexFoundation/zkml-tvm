
from .circom import *


"""
    Generator Implementation

    override apply function, several variables to be set:

        1. circom_input
        2. circom_args
        3. circom_output
        4. shape 
"""

class InputGenerator(CircomGenerator):
    def apply(self):
        self.circom_output = self.name

    def fill_circom(self, code: str) -> str:
        if self._visit_flag:
            return code
        self._visit_flag = True

        circom_shape = [str(s) for s in self.shape ]
        circom_shape = ["["+s+"]" for s in circom_shape]
        return inject_signal(code, "signal input {}{};".format(
                self.name, "".join(circom_shape)))


class OutputGenerator(CircomGenerator):
    def apply(self):
        pass

    def fill_circom(self, code: str) -> str:
        if self._visit_flag:
            return code
        self._visit_flag = True

        assert len(self.inputs) == 1
        assert self.shape == self.inputs[0].shape

        for inp in self.inputs:
            code = inp.fill_circom(code)

        circom_shape_lst = ["["+str(s)+"]" for s in self.shape]
        code = inject_signal(code, "signal output {}{};".format(
                    self.name, "".join(circom_shape_lst)))

        circom_shape = "{main}"
        for idx, dim in enumerate(self.shape):
            circom_for = (
                "for (var i{idx} = 0; i{idx} < {dim}; i{idx}++) {brace_left}\n"
                "{main}\n"
                "{brace_right}\n"
            ).format_map(SafeDict(idx=idx, dim=dim))
            circom_shape = circom_shape.format_map(
                    SafeDict(main=circom_for.strip()))

        circom_index = ["[i"+str(i)+"]" \
                for i in range(len(self.shape))]
        circom_assign = "\t{}{} <== {}{};".format(
                self.name, "".join(circom_index),
                self.inputs[0].circom_output, "".join(circom_index),
                )
        circom_shape = circom_shape.format_map(
                SafeDict(main=circom_assign))
        return inject_main(code, circom_shape)

class OperatorGenerator(CircomGenerator):
    def apply(self):
        input_shapes = [inp.shape for inp in self.inputs]
        # check input shape dimensions match operators in cirom circuit operator
        # print(self.comp.input_dims, input_shapes, self.info(), self.comp.input_names)

        assert len(self.comp.input_names) == len(self.inputs), "{};{}".format(len(self.comp.input_names), len(self.inputs))
        # op dim contains 1-dim batch, not support in circom circuits
        input_index = 0
        for shape in zip(self.comp.input_dims, input_shapes):
            # model input shape dimensions should match cirom circuit operator shape
            assert shape[0] == len(shape[1]), (
                "{}({}) input[{}] shape dim not matched, "
                "{} vs. {}, maybe apply shape-adaptor pass. shape is:{}"
            ).format(self.name, self.comp.op_name, input_index,
                    shape[0], len(shape[1]), shape)
            input_index += 1

        self.circom_inputs = [
                Signal(self, *info) for info in zip(
                    self.comp.input_names, input_shapes) ]

        args = self.arguments()
        # all arguments of circom circuit must be integers.
        assert all([isinstance(a, int) for a in args]), print("bad arg display!!", ["{};".format(a) for a in args], self.info()) #self.info()
        self.circom_args = ", ".join([
            str(s) for s in self.arguments()])

        #  self.circom_output = self.output_name()
        assert len(self.comp.output_names) == 1, "names:{}, dims:{}".format(self.comp.output_names, self.comp.output_dims)
        self.circom_output = "{}.{}".format(self.name, self.comp.output_names[0])

        # check output shape dimensions match
        assert self.comp.output_dims[0] == len(self.shape), self.info()

    def arguments(self):
        raise NotImplementedError(self.comp.op_name)

class ShapeGenerator(OperatorGenerator):
    def arguments(self):
        return [ *self.shape ]
class Broadcast2DAxis0SubGenerator(ShapeGenerator):
    pass
class Broadcast2DAxis0AddGenerator(ShapeGenerator):
    pass
class Broadcast2DAxis1SubGenerator(ShapeGenerator):
    pass
class Broadcast2DAxis1AddGenerator(ShapeGenerator):
    pass
class Broadcast3DAxis0SubGenerator(ShapeGenerator):
    pass
class Broadcast3DAxis0AddGenerator(ShapeGenerator):
    pass
class Broadcast3DAxis1SubGenerator(ShapeGenerator):
    pass
class Broadcast3DAxis1AddGenerator(ShapeGenerator):
    pass
class Broadcast3DAxis1Add_0_2Generator(ShapeGenerator):
    pass
class Broadcast4DAxis0AddGenerator(ShapeGenerator):
    pass
class Broadcast4DAxis1AddGenerator(ShapeGenerator):
    pass
class Broadcast4DAxis1Add_FirstGenerator(ShapeGenerator):
    pass
class Element1DAddGenerator(ShapeGenerator):
    pass
class Element2DAddGenerator(ShapeGenerator):
    pass
class Element3DAddGenerator(ShapeGenerator):
    pass
class Element1DSubGenerator(ShapeGenerator):
    pass
class Element2DSubGenerator(ShapeGenerator):
    pass
class Element3DSubGenerator(ShapeGenerator):
    pass
class Element1DMulGenerator(ShapeGenerator):
    pass

#  class ElementGenerator(OperatorGenerator):
#      def arguments(self):
#          return [ self.shape[0], ]
#  class ElementAddGenerator(ElementGenerator):
#      pass
#  class ElementSubGenerator(ElementGenerator):
#      pass
#  class ElementMulGenerator(ElementGenerator):
#      pass

# just for test, invalid now
class Conv2D_NCHWGenerator(OperatorGenerator):
    def arguments(self):
        padding = self.attrs["padding"]
        assert all([p == 0 for p in padding])

        filters = self.attrs["channels"]
        kernel_size = self.attrs["kernel_size"]
        if isinstance(kernel_size, int):
            kernel_size = [kernel_size, kernel_size]
        assert kernel_size[0] == kernel_size[1]
        kernel_size = kernel_size[0]
        return [ *self.inputs[0].shape, filters, kernel_size, 1, ]

class Conv2D_CHWGenerator(OperatorGenerator):
    def arguments(self):

        strides = self.attrs["strides"] if len(self.attrs["strides"]) > 0 else [1]
        assert all([s == strides[0] for s in strides])
        padding = self.attrs["padding"]
        assert all([p == 0 for p in padding])
        dilation = self.attrs["dilation"]
        assert all([d == 1 for d in dilation])

        filters = self.attrs["channels"]
        kernel_size = self.attrs["kernel_size"]
        if isinstance(kernel_size, int):
            kernel_size = [kernel_size, kernel_size]
        assert kernel_size[0] == kernel_size[1]
        kernel_size = kernel_size[0]
        return [ *self.inputs[0].shape, filters, kernel_size, strides[0], ]

class MaxPool2DGenerator(OperatorGenerator):
    def arguments(self):
        strides = self.attrs["strides"] if len(self.attrs["strides"]) > 0 else [1]
        assert all([s == strides[0] for s in strides])
        padding = self.attrs["padding"]
        assert all([p == 0 for p in padding])
        dilation = self.attrs["dilation"]
        assert all([d == 1 for d in dilation])
        pool_size = self.attrs["pool_size"]
        assert all([p == pool_size[0] for p in pool_size])

        return [ *self.inputs[0].shape, pool_size[0], strides[0], ]


class Pad2DGenerator(OperatorGenerator):
    def arguments(self):
        pad_value = self.attrs.get("scalar", None)
        if pad_value is None:
            pad_value = self.attrs["pad_value"]
        pad_width = [p for p in self.attrs["padding"]]
        return [ *self.inputs[0].shape, pad_value, *pad_width ]

class BiasAdd1Generator(OperatorGenerator):
    def arguments(self):
        assert len(self.inputs) == 2
        assert len(self.inputs[0].shape) == 1
        assert len(self.inputs[1].shape) == 1
        assert self.inputs[0].shape[0] == self.inputs[1].shape[0]
        return [ *self.inputs[0].shape ]
class BiasAdd2Generator(OperatorGenerator):
    def arguments(self):
        assert len(self.inputs) == 2
        assert len(self.inputs[0].shape) == 2
        assert len(self.inputs[1].shape) == 1
        assert self.inputs[0].shape[0] == self.inputs[1].shape[0]
        return [ *self.inputs[0].shape ]
class BiasAdd3Generator(OperatorGenerator):
    def arguments(self):
        assert len(self.inputs) == 2
        assert len(self.inputs[0].shape) == 3
        assert len(self.inputs[1].shape) == 1
        assert self.inputs[0].shape[0] == self.inputs[1].shape[0]
        return [ *self.inputs[0].shape ]


class Resize2DGenerator(OperatorGenerator):
    def arguments(self):
        method = self.attrs.get("method", "nearest_neighbor")
        assert method == "nearest_neighbor"

        input_shape = self.inputs[0].shape
        scaleX = self.shape[1] / input_shape[1]
        scaleY = self.shape[2] / input_shape[2]
        assert scaleX == scaleY
        assert int(scaleX) == scaleX
        return self.inputs[0].shape + [ int(scaleX), ]


def reshape_validate(shape_one, shape_arr, msg):
        assert len(shape_one) == 1
        total_len = 1
        for s in shape_arr:
            total_len *= s
        assert shape_one[0] == total_len, msg

class ReShapeGenerator(OperatorGenerator):
    def arguments(self):
        reshape_validate(
                self.inputs[0].shape,
                self.shape, self.info())
        return self.shape
class ReShape2DGenerator(ReShapeGenerator):
    pass
class ReShape3DGenerator(ReShapeGenerator):
    pass
class ReShape4DGenerator(ReShapeGenerator):
    pass

class FlattenGenerator(OperatorGenerator):
    def arguments(self):
        reshape_validate(
                self.shape,
                self.inputs[0].shape, self.info())
        return self.inputs[0].attrs["shape"]
class Flatten2DGenerator(FlattenGenerator):
    pass
class Flatten3DGenerator(FlattenGenerator):
    pass
class Flatten4DGenerator(FlattenGenerator):
    pass

class Dense2Generator(OperatorGenerator):
    def arguments(self):
        return self.inputs[1].shape

class ScalarGenerator(OperatorGenerator):
    def arguments(self):
        ishape = self.inputs[0].shape
        assert len(ishape) == 1
        return [ishape[0], self.attrs["scalar"]]
class MulScalarGenerator(ScalarGenerator):
    pass
class MulScalarCHGenerator(ScalarGenerator):
    def arguments(self):
        i_shape = self.inputs[0].shape
        s_shape = self.inputs[1].shape
        # s_shape[0] is batch, should be 1, then just ignored
        assert len(i_shape) == 2
        assert len(s_shape) == 3
        assert s_shape[0] == 1 and s_shape[2] == 1
        assert i_shape[0] == s_shape[1]
        return [ *i_shape ]
class MulScalarCHWGenerator(ScalarGenerator):
    def arguments(self):
        i_shape = self.inputs[0].shape
        s_shape = self.inputs[1].shape
        # s_shape[0] is batch, should be 1, then just ignored
        assert len(i_shape) == 3
        assert len(s_shape) == 4
        assert s_shape[0] == 1 and s_shape[2] == 1 and s_shape[3] == 1
        assert i_shape[0] == s_shape[1]
        return [ *i_shape ]
class MulScalarCHW_ByHWGenerator(ScalarGenerator):
    def arguments(self):
        i_shape = self.inputs[0].shape
        s_shape = self.inputs[1].shape
        # s_shape[0] is batch, should be 1, then just ignored
        assert len(i_shape) == 3
        assert len(s_shape) == 4
        assert s_shape[0] == 1 and s_shape[1] == 1 and s_shape[2] != 1 and s_shape[3] != 1
        assert i_shape[1] == s_shape[2]
        assert i_shape[2] == s_shape[3]
        return [ *i_shape ]
class MulScalar3D3D_InputGenerator(ScalarGenerator):
    def arguments(self):
        i_shape = self.inputs[0].shape
        s_shape = self.inputs[1].shape
        assert len(i_shape) == 3
        assert len(s_shape) == 3
        assert i_shape[0] == s_shape[0] and (i_shape[1]==s_shape[1] or s_shape[1]==1) and (i_shape[2]==s_shape[2] or s_shape[2]==1), "i:{}, s:{}".format(i_shape,s_shape)
        return [ *i_shape, s_shape[1], s_shape[2] ]

class AddScalarGenerator(ScalarGenerator):
    pass
class SubScalarGenerator(ScalarGenerator):
    pass
class AddScalarCHGenerator(ScalarGenerator):
    pass
class SubScalarCHGenerator(ScalarGenerator):
    pass

#class MulScalarGenerator(OperatorGenerator):
#    def arguments(self):
#        return [ *self.shape,  ]
class RightShiftGenerator(OperatorGenerator):
    def arguments(self):
        return [ *self.shape, self.attrs["scalar"]]

class ReLU1DGenerator(OperatorGenerator):
    def arguments(self):
        return [ *self.shape ]
class ReLU2DGenerator(OperatorGenerator):
    def arguments(self):
        return [ *self.shape ]
class ReLU3DGenerator(OperatorGenerator):
    def arguments(self):
        return [ *self.shape ]

class Pass1DGenerator(OperatorGenerator):
    def arguments(self):
        return [ *self.inputs[0].shape ]
class Pass2DGenerator(OperatorGenerator):
    def arguments(self):
        return [ *self.inputs[0].shape ]
class Pass3DGenerator(OperatorGenerator):
    def arguments(self):
        return [ *self.shape ]
class Pass4DGenerator(OperatorGenerator):
    def arguments(self):
        return [ *self.inputs[0].shape ]
class Pass5DGenerator(OperatorGenerator):
    def arguments(self):
        return [ *self.inputs[0].shape ]

class Sum_CHWGenerator(OperatorGenerator):
    def arguments(self):
        keepdims = self.attrs["keepdims"]
        assert(keepdims == None or keepdims == True)
        return [ *self.inputs[0].shape ]

class Sum_CHW_0Generator(OperatorGenerator):
    def arguments(self):
        keepdims = self.attrs["keepdims"]
        assert(keepdims == False)
        return [ *self.inputs[0].shape ]

class Squeeze_CHWGenerator(OperatorGenerator):
    def arguments(self):
        iShape = self.inputs[0].shape
        assert(len(iShape)==3 and iShape[1]==1 and iShape[2]==1)
        return [ *self.inputs[0].shape ]

class Clip1DGenerator(OperatorGenerator):
    def arguments(self):
        assert int(self.attrs["a_min"]) == self.attrs["a_min"]
        assert int(self.attrs["a_max"]) == self.attrs["a_max"]
        return [ self.shape[0],
                int(self.attrs["a_min"]), int(self.attrs["a_max"]) ]

class Clip2DGenerator(OperatorGenerator):
    def arguments(self):
        assert int(self.attrs["a_min"]) == self.attrs["a_min"]
        assert int(self.attrs["a_max"]) == self.attrs["a_max"]
        return [ self.shape[0], self.shape[1],
                int(self.attrs["a_min"]), int(self.attrs["a_max"]) ]

class Clip3DGenerator(OperatorGenerator):
    def arguments(self):
        assert int(self.attrs["a_min"]) == self.attrs["a_min"]
        assert int(self.attrs["a_max"]) == self.attrs["a_max"]
        return [ self.shape[0], self.shape[1], self.shape[2],
                int(self.attrs["a_min"]), int(self.attrs["a_max"]) ]

class Transpose2DGenerator(OperatorGenerator):
    def arguments(self):
        assert(len(self.shape)==2)
        assert(self.attrs["axes"][:]==[2,1]), self.attrs["axes"]
        return [ *self.inputs[0].shape ]
class Transpose3D_312Generator(OperatorGenerator):
    def arguments(self):
        assert(len(self.shape)==3)
        assert(self.attrs["axes"][:]==[3,1,2]), self.attrs["axes"]
        # only transpose C to the end
        return [ *self.inputs[0].shape ]
class Transpose3D_231Generator(OperatorGenerator):
    def arguments(self):
        assert(len(self.shape)==3)
        assert(self.attrs["axes"][:]==[2,3,1]), self.attrs["axes"]
        # only transpose C to the end
        return [ *self.inputs[0].shape ]
class Transpose4D_2134Generator(OperatorGenerator):
    def arguments(self):
        assert(len(self.shape)==4)
        assert(self.attrs["axes"][:]==[2,1,3,4]), self.attrs["axes"]
        # only transpose C1 and C2
        return [ *self.inputs[0].shape ]

class TupleGetItem2D0AGenerator(OperatorGenerator):
    def arguments(self):
        assert(len(self.inputs[0].shape)==2)
        return [ *self.inputs[0].shape, self.attrs["parts"], self.attrs["index"] ]
class TupleGetItem2D1AGenerator(OperatorGenerator):
    def arguments(self):
        assert(len(self.inputs[0].shape)==2)
        return [ *self.inputs[0].shape, self.attrs["parts"], self.attrs["index"] ]
class TupleGetItem3DGenerator(OperatorGenerator):
    def arguments(self):
        assert(len(self.inputs[0].shape)==3)
        return [ *self.inputs[0].shape, self.attrs["parts"], self.attrs["index"] ]
class TupleGetItem3D0AGenerator(TupleGetItem3DGenerator):
    pass
class TupleGetItem3D1AGenerator(TupleGetItem3DGenerator):
    pass
class TupleGetItem3D2AGenerator(TupleGetItem3DGenerator):
    pass

class Tuple3ItemGenerator(OperatorGenerator):
    def arguments(self):
        assert(len(self.inputs)==3)
        assert all([self.inputs[0].shape[0]==self.inputs[1].shape[0], self.inputs[1].shape[0]==self.inputs[2].shape[0]])
        assert all([self.inputs[0].shape[1]==1, self.inputs[1].shape[1]==1, self.inputs[2].shape[1]==4])
        return [ self.inputs[0].shape[0] ]

class TupleGetItem_VisCount_0Generator(OperatorGenerator):
    def arguments(self):
        assert(len(self.inputs)==2)
        assert(self.attrs["id_index"]==0)
        assert(self.attrs["score_index"]==1)
        return [ *self.inputs[0].shape ]
class TupleGetItem_VisCount_1Generator(OperatorGenerator):
    def arguments(self):
        assert(len(self.inputs)==2)
        assert(self.attrs["id_index"]==0)
        assert(self.attrs["score_index"]==1)
        return [ *self.inputs[0].shape ]
class TupleGetItem_VisCount_2Generator(OperatorGenerator):
    def arguments(self):
        assert(len(self.inputs)==2)
        assert(self.attrs["id_index"]==0)
        assert(self.attrs["score_index"]==1)
        return [ *self.inputs[0].shape ]

class StrideSlice2DGenerator(OperatorGenerator):
    def arguments(self):
        assert(len(self.inputs[0].shape)==2)
        return [ *self.inputs[0].shape, *self.attrs["begin"][1:], *self.attrs["end"][1:], *self.attrs["strides"] ]
class StrideSlice3DGenerator(OperatorGenerator):
    def arguments(self):
        assert(len(self.inputs[0].shape)==3)
        return [ *self.inputs[0].shape, *self.attrs["begin"][1:], *self.attrs["end"][1:], *self.attrs["strides"] ]

class Greater2DGenerator(OperatorGenerator):
    def arguments(self):
        assert(len(self.shape)==2)
        assert(len(self.inputs)==2)
        return [ *self.shape ]

class Where2DGenerator(OperatorGenerator):
    def arguments(self):
        assert(len(self.shape)==2)
        assert(len(self.inputs)==3)
        assert all([self.shape[0]==self.inputs[0].shape[0], self.inputs[0].shape[0]==self.inputs[1].shape[0], self.shape[1]==self.inputs[0].shape[1], self.inputs[0].shape[1]==self.inputs[1].shape[1]])
        return [ *self.shape ]

class AdvIndex2DGenerator(OperatorGenerator):
    def arguments(self):
        assert(len(self.inputs)==2)
        # this circuit only support input[0] 1-dim, input[1] 2-dim
        assert(len(self.inputs[0].shape)==1)
        assert(len(self.inputs[1].shape)==2)
        return [ *self.inputs[0].shape, *self.inputs[1].shape ]
class AdvIndex3DGenerator(OperatorGenerator):
    def arguments(self):
        assert(len(self.inputs)==2)
        # this circuit only support input[0] 1-dim, input[1] 2-dim
        assert(len(self.inputs[0].shape)==1)
        assert(len(self.inputs[1].shape)==3)
        return [ *self.inputs[0].shape, *self.inputs[1].shape ]

class Vision_GetValidCountsGenerator(OperatorGenerator):
    def arguments(self):
        assert(len(self.inputs)==2)
        return [ *self.inputs[0].shape ]

class Vision_NonMaxSuppressionGenerator(OperatorGenerator):
    def arguments(self):
        assert(len(self.inputs)==5)
        assert(self.attrs["invalid_to_bottom"]==1)
        top_k = self.attrs["top_k"] if self.attrs["top_k"]>0 else self.inputs[0].shape[0]
        return [ *self.inputs[0].shape, top_k ]

class Concatenate1D0AGenerator(OperatorGenerator):
    def arguments(self):
        assert(len(self.shape)==1)
        assert(len(self.inputs)==2)
        return [ self.inputs[0].shape[0], *self.inputs[1].shape ]
class Concatenate2D0AGenerator(OperatorGenerator):
    def arguments(self):
        assert(len(self.shape)==2)
        assert(len(self.inputs)==2)
        assert all([self.inputs[0].shape[1] == self.inputs[1].shape[1]])
        return [ self.inputs[0].shape[0], *self.inputs[1].shape ]
class Concatenate2D1AGenerator(OperatorGenerator):
    def arguments(self):
        assert(len(self.shape)==2)
        assert(len(self.inputs)==2)
        assert all([self.inputs[0].shape[0] == self.inputs[1].shape[0]])
        return [ *self.inputs[0].shape, self.inputs[1].shape[1] ]
class Concatenate3D0AGenerator(OperatorGenerator):
    def arguments(self):
        assert(len(self.shape)==3)
        assert(len(self.inputs)==2)
        assert all([self.inputs[0].shape[1] == self.inputs[1].shape[1], self.inputs[0].shape[2] == self.inputs[1].shape[2]])
        return [ self.inputs[0].shape[0], *self.inputs[1].shape ]
class Concatenate3D2AGenerator(OperatorGenerator):
    def arguments(self):
        assert(len(self.shape)==3)
        assert(len(self.inputs)==2)
        assert all([self.inputs[0].shape[0] == self.inputs[1].shape[0], self.inputs[0].shape[1] == self.inputs[1].shape[1]])
        return [ *self.inputs[0].shape, self.inputs[1].shape[2] ]
class Concatenate4D3AGenerator(OperatorGenerator):
    def arguments(self):
        assert(len(self.shape)==4)
        assert(len(self.inputs)==2)
        assert all([self.inputs[0].shape[0] == self.inputs[1].shape[0], self.inputs[0].shape[1] == self.inputs[1].shape[1], self.inputs[0].shape[2] == self.inputs[1].shape[2]])
        return [ *self.inputs[0].shape, self.inputs[1].shape[3] ]


class Negative3DGenerator(OperatorGenerator):
    def arguments(self):
        assert(len(self.shape)==3)
        assert(len(self.inputs)==1)
        return [ *self.shape ]

class ExpandDims3D_3AGenerator(OperatorGenerator):
    def arguments(self):
        num_newaxis = self.attrs["num_newaxis"]
        assert(len(self.shape)==4) # add one dim in the end
        assert(len(self.inputs)==1)
        return [ *self.inputs[0].shape, num_newaxis ]

class SliceLike3D_2_3Generator(OperatorGenerator):
    def arguments(self):
        assert self.attrs["axes"]==[2,3]
        assert(len(self.shape)==3)
        assert(len(self.inputs[0].shape)==3)
        assert(len(self.inputs[1].shape)==3)
        assert(len(self.inputs)==2)
        return [ *self.inputs[0].shape, *self.inputs[1].shape ]

class Repeat3D_0AGenerator(OperatorGenerator):
    def arguments(self):
        assert(len(self.shape)==3)
        assert(len(self.inputs)==1)
        return [ *self.inputs[0].shape, int(self.attrs["repeats"]) ]

class Repeat3D_1AGenerator(OperatorGenerator):
    def arguments(self):
        assert(len(self.shape)==3)
        assert(len(self.inputs)==1)
        return [ *self.inputs[0].shape, int(self.attrs["repeats"]) ]

class Repeat3D_2AGenerator(OperatorGenerator):
    def arguments(self):
        assert(len(self.shape)==3)
        assert(len(self.inputs)==1)
        return [ *self.inputs[0].shape, int(self.attrs["repeats"]) ]

class Tile3DGenerator(OperatorGenerator):
    def arguments(self):
        assert(len(self.inputs)==1)
        assert all([j==1 for j in self.attrs["reps"][1:]])
        assert(len(self.shape)==4)
        return [ *self.inputs[0].shape, self.attrs["reps"][0] ]

class Tile4DGenerator(OperatorGenerator):
    def arguments(self):
        assert(len(self.inputs)==1)
        assert all([j==1 for j in self.attrs["reps"][1:]])
        assert(len(self.shape)==5)
        return [ *self.inputs[0].shape, self.attrs["reps"][0] ]

