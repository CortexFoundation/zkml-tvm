from dataclasses import InitVar
from collections import namedtuple

import numpy as np

from mrt.mir import op
from mrt.mir.opns import *
from mrt.mir.symbol import *
from mrt.mir.attrs import *

from mrt.runtime import inference
from mrt.common.utils import N, product

from .transform import Transformer

# TODO: add op pass register map.

class FuseDropout(Transformer):
    #out = filter_operators(DROP_OUT)(__call__)
    # def out():
    @filter_operators(DROP_OUT)
    def __call__(self, **kwargs):
        return self.args[0]

class FuseConstant(Transformer):
    threshold: typing.ClassVar[float] = 1e-5

    def np_is_zero(self, data) -> float:
        return np.abs(data).max() < self.threshold

    def __call__(self: Transformer, **kw):
        if self.is_operator() and all([c.is_param() for c in self.args]):
            data = inference.run_single(
                    self, [a.numpy() for a in self.args])
            return self.as_parameter(data)
        elif self.is_op(ADD, SUB): # , BIAS_ADD):
            strips = []
            for arg in self.args:
                if arg.is_param() and self.np_is_zero(arg.numpy()):
                #  if arg.is_param() and np.abs(arg.numpy()).max() == 0:
                    strips.append(arg)
            args = [a for a in self.args if a not in strips]
            if len(args) == 1:
                return args[0]
        elif self.is_op(SLICE_LIKE):
            if not self.args[0].is_param():
                return
            a, b = self.args
            arg1 = np.zeros(b.shape, b.dtype)
            data = inference.run_single(
                self, [a.numpy(), np.zeros(b.shape, b.dtype)])
            return self.as_parameter(data)
        elif self.is_op(REQUANT):
            if self.parsed.rescale == 1:
                return self.args[0]
        elif self.is_op(ZEROS_LIKE, ONES_LIKE):
            data = inference.run_single(self, [])
            return self.as_parameter(data)


class FuseBatchNorm(Transformer):
    @filter_operators(BATCH_NORM)
    def __call__(self, **kw):
        X, gamma, beta, mean, var = self.args
        parsed: BatchNormAttrs = self.parsed

        gamma, beta = gamma.numpy(), beta.numpy()
        mean, var = mean.numpy(), var.numpy()
        #  print(gamma.shape, beta.shape, mean.shape, var.shape)

        assert parsed.axis == 1
        beta = beta if parsed.center else 0
        gamma = gamma if parsed.scale else 1

        # (X - mean) / sqrt(var + epsilon) * gamma + beta
        gamma = gamma / np.sqrt(var + parsed.epsilon)
        # (X - mean) * gamma + beta
        # X * gamma + (beta - mean * gamma)
        bias: np.ndarray = (beta - mean * gamma)
        #  print(np.abs(gamma).max(), np.abs(bias).max())
        # X * gamma + bias

        if X.is_op(CONV2D):
            A, W = X.args
            conv_parsed: Conv2DAttrs = X.parsed

            assert conv_parsed.kernel_layout == "OIHW"
            K = gamma.shape[0]
            assert W.shape[0] == K

            # (A * W) * gamma + bias
            # A * (W * gamma) + bias
            W_data = W.numpy() * gamma.reshape(K, 1, 1, 1)
            W_sym = W.from_np_data(W_data)
            out = op.nn_conv2d(A, W_sym, **X.attrs)
        elif X.is_op(DENSE):
            A, W = X.args
            dense_parsed: DenseAttrs = X.parsed

            # (A * W) * gamma + bias
            # A * (W * gamma) + bias
            W_data = W.numpy() * gamma.reshape(K, 1)
            W_sym = W.from_np_data(W_data)
            out = op.nn_dense(A, W_sym, **X.attrs)
        else:
            reshp = [s if i == parsed.axis else 1 \
                    for i, s in enumerate(X.shape)]
            W = X.from_np_data(gamma.reshape(reshp))
            out = op.mul(X, W)

        bias = bias.reshape([s if i == parsed.axis else 1 \
                for i, s in enumerate(out.shape)])
        B = out.like(self).from_np_data(bias)
        out = op.add(out, B)
        # out = op.bias_add(out, B, axis=parsed.axis)
        return out.like(self)

class FuseTupleGetItem(Transformer):
    @filter_operators(TUPLE_GET_ITEM)
    def __call__(self, **kw):
        X: Symbol = self.args[0]
        if X.is_op(BATCH_NORM, DROP_OUT):
            return X
        #  assert X.is_op(BATCH_NORM, DROP_OUT), X.name
        #  assert self.parsed.index == 0
        #  return X

class FuseAvgPool2D(Transformer):
    def __call__(self, **kw):
        out = self._fuse_adaptive_avg_pool2d()
        out = out or self._fuse_avg_pool2d()
        return out

    @filter_operators(AVG_POOL2D)
    def _fuse_avg_pool2d(self):
        X: Transformer = self.args[0]
        parsed: AvgPool2DAttrs = self.parsed
        assert parsed.layout == "NCHW"
        # TODO: ignore for unstrict mode
        assert parsed.count_include_pad == True
        attrs = {
            "kernel_size": parsed.pool_size,
            "strides": parsed.strides,
            "padding": parsed.padding,
            "dilation": parsed.dilation,
            "data_layout": parsed.layout,
            "groups": X.shape[1],
            "channels": X.shape[1],
            }
        W_shape = (X.shape[1], 1, *parsed.pool_size)
        W = X.from_np_data(np.full(
            W_shape, 1 / product(parsed.pool_size)))
        out = op.nn_conv2d(X, W, **attrs)
        return out.like(self)


    @filter_operators(ADAPTIVE_AVG_POOL2D)
    def _fuse_adaptive_avg_pool2d(self):
        X: Transformer = self.args[0]
        parsed: AdaptiveAvgPool2DAttrs = self.parsed
        assert parsed.layout == "NCHW"
        ins = X.shape[2:]
        ous = parsed.output_size or ins
        if not isinstance(ous, (list, tuple)):
            ous = (ous, ous)
        parsed.output_size = ous

        assert len(X.shape) == 4
        if all([s == 1 for s in parsed.output_size]):
            scale = np.array(1 / np.prod(X.shape[-2:]))
            out = op.sum(X, axis=list(range(4))[-2:], keepdims=True)
            scale = self.from_np_data(scale.astype(X.dtype))
            return op.mul(out, scale).like(self)
        elif ous[0] > ins[0] or ous[1] > ins[1]:
            assert all([s == 1 for s in ins])
            out = op.repeat(X, repeats=ous[0], axis=-2)
            out = op.repeat(out, repeats=ous[1], axis=-1)
            return out.like(self)

        # calculate the attributes refers to:
        # https://stackoverflow.com/questions/53841509/how-does-adaptive-pooling-in-pytorch-work
        strides = [i // o for i, o in zip(ins, ous)]
        kernel = [i-(o-1)*s for i, o, s in zip(ins, ous, strides)]
        attrs = {
            "kernel_size": kernel,
            "strides": strides,
            "padding": (0, 0),
            "dilation": (1, 1),
            "data_layout": parsed.layout,
            "groups": X.shape[1],
            "channels": X.shape[1],
        }
        W_shape = (X.shape[1], 1, *kernel)
        W = X.from_np_data(np.full(W_shape, 1 / product(kernel)))
        out = op.nn_conv2d(X, W, **attrs)
        return out.like(self)

class FuseNaiveSoftmax(Transformer):
    def __call__(self, **kw):
        return self # not fuse pass

        if self.is_op(SOFTMAX, LOG_SOFTMAX):
            return self.args[0]
        assert self.is_variable() or not self.args[0].is_op(SOFTMAX, LOG_SOFTMAX)
        return self

class FuseMean(Transformer):
    @filter_operators(MEAN)
    def __call__(self, **kw):
        X: Transformer = self.args[0]
        #  max_axis = len(X.shape)
        #  axis = X.attrs.get("axis", None)
        #  axis = axis or [i for i in range(max_axis)]
        #  axis = [a if a >= 0 else a + max_axis for a in axis]
        #  assert all([a >= 0 and a < max_axis for a in axis])
        #  if exclude:
        #      axis = [a for a in range(max_axis) if a not in axis]
        #  axis_len = product([X.shape[a] for a in axis])

        out = op.sum(X, **self.attrs)
        scale = self.from_np_data(np.array(
            1. * product(out.shape) / product(X.shape)))
        out = op.mul(out, scale)
        return out.like(self)

class FuseLeakyReLU(Transformer):
    @filter_operators(LEAKY_RELU)
    def __call__(self, **kw):
        """ Customized rewrite pass Introduction.

            LeakyReLU can be equivalently transformed.

            .. math::
                LeakyReLU(X) = relu(X) - slope*relu(-X)
        """
        alpha = self.from_const_data(self.parsed.alpha)
        X: Transformer = self.args[0]
        out = op.nn_relu(op.negative(X))
        out = op.mul(alpha, out)
        out = op.sub(op.nn_relu(X), out)
        return out.like(self)


class FuseDivide(Transformer):
    @filter_operators(DIV)
    def __call__(self, **kw):
        """ Transform div to mul if possible. """
        A: Transformer = self.args[0]
        B: Transformer = self.args[1]
        assert B.is_param(), B
        B = B.from_np_data(1. / B.numpy())
        return op.mul(A, B).like(self)

# move to fuse constant
#  class FuseNaiveMathmatic(Transformer):
#      def __call__(self):
#          if self.is_op(BIAS_ADD):
#              X, B = self.args
#              if B.is_param() and np.abs(B.numpy()).max() == 0:
#                  return X




