import typing

import tvm
import numpy as np

OpOutputT = typing.Union[tvm.nd.NDArray, list]
OpNumpyT = typing.Union[np.ndarray, list]
ParametersT = typing.Dict[str, OpOutputT]
AttrsT = typing.Dict[str, typing.Any]

ShapeT = typing.List[int]
""" shape type, list of int, such as [1, 3, 34, 34]. """
DTypeT = str

DataLabelT = typing.Tuple[np.ndarray, typing.Any]
""" a (data, label) representation. """

def to_numpy(data: OpOutputT) -> OpNumpyT:
    return [d.numpy() for d in data] \
            if isinstance(data, list) else data.numpy()

def to_ndarray(data: OpNumpyT) -> OpOutputT:
    return [tvm.nd.array(d) for d in data] \
            if isinstance(data, list) else tvm.nd.array(data)
