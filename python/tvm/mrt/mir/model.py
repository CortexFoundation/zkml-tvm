import typing

import numpy as np

from .symbol import *
from .types import *

class Graph:
    """ Same level as the relax IRModule.  """
    mod: typing.Dict[str, Symbol] = {}
    params: ParametersT = {}

    def __setitem__(self, name: str, val: Symbol):
        self.mod[name] = val

    def merge_mod_params(self, new_params: ParametersT):
        for k, v in new_params.items():
            if k not in self.params:
                continue
            assert np.allclose(self.params[k], v), \
                f"parameter:{k} not equal, don't know use which one."
        self.params.update(new_params)
