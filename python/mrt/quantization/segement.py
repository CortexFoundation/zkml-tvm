import typing

from dataclasses import dataclass, field

from mrt.mir.symbol import *
from mrt.mir import op, opns, helper

from .scaler import WithScale
from .transform import RunOnce

_SCALE_CONSTANT_OPS = [
    opns.VAR,
    opns.WHERE, # opns.GREATER,
    opns.REPEAT, opns.SQUEEZE,
    opns.FLATTEN, opns.BATCH_FLATTEN,
    opns.RESHAPE, opns.CONCAT,
    opns.SPLIT, opns.TRANSPOSE,
    opns.EXPAND_DIMS, opns.TILE,
    opns.STRIDED_SLICE,
    opns.TUPLE, opns.TUPLE_GET_ITEM,
    opns.GET_VALID_COUNT,
    opns.NON_MAX_SUPRESSION,
    opns.CLIP, opns.AS_TYPE,
        ]

@dataclass(repr=False)
class Spliter(RunOnce):
    head: typing.Optional[dict] = None
    head_params: typing.Optional[typing.Dict[str, OpNumpyT]] = None
    seg_names: typing.List[str] = field(default_factory=list)

    def __call__(self, **kwargs):
        """ Auto split the model. """
        refs = { self.name: 1 } # add refs for root symbol
        def _collect_refs(sym: Spliter):
            refs.setdefault(sym.name, 0)
            if sym.is_variable():
                return
            for a in sym.args:
                refs.setdefault(a.name, 0)
                refs[a.name] += 1
        visit(self, _collect_refs)

        sym_map = {}
        sym_status = {}
        heads = [self]
        """ status code:
            1 means current symbol has been scaned and sub childs have
                been added into scan list.
            2 means current symbol matches the scan rules, but still
                exist references from other symbol. Wait for following
                scan loops to decrease reference until 0, which can
                change status to 1; Or as status3 after scan loop end.
            3 means current symbol is at the scan loop leaf, need to be
                splited for next procedure(calibrate, quantize, etc.).
        """
        while heads:
            # print([s.name for s in heads])
            new_heads = []
            for s in heads:
                sym_map.setdefault(s.name, s)
                sym_status.setdefault(s.name, 0)
                if sym_status[s.name] in [1, 3]:
                    continue

                if s.is_op(*_SCALE_CONSTANT_OPS):
                    refs[s.name] -= 1
                    if refs[s.name] == 0:
                        sym_status[s.name] = 1
                        new_heads.extend(s.args)
                    else:
                        sym_status[s.name] = 2
                else:
                    sym_status[s.name] = 3
            heads = new_heads

        out_names = [n for n, s in sym_status.items() if s in [2, 3]]
        outs = [sym_map[n] for n in out_names]

        # for n, s in sym_status.items():
        #     assert s in [1, 2, 3], (n, s)

        # scans = [ self, ]
        # scaned = []
        # outs = []
        # while scans:
        #     print([s.name for s in scans])
        #     new_scans = []
        #     for s in scans:
        #         if s.name in scaned:
        #             continue
        #         scaned.append(s.name)
        #         if s.is_op(*_SCALE_CONSTANT_OPS):
        #             new_scans.extend(s.args)
        #         else:
        #             outs.append(s)
        #     scans = new_scans

        self.seg_names = [o.name for o in outs]
        # print(sorted(self.seg_names))

        def _split(sym: Spliter):
            return op.as_variable(sym) \
                    if sym.name in self.seg_names else sym
        head = transform(self, _split)
        self.head = dump_json(head)

        self.head_params = {}
        def _update_params(sym: Symbol):
            if op.is_param(sym, self.params):
                self.head_params[sym.name] = to_numpy(
                        self.params[sym.name])
        visit(head, _update_params)

        # helper.format_print(head, self.head_params)

        return op.Tuple(*outs).like(self)

@dataclass(repr=False)
class Merger(WithScale, RunOnce):
    def __call__(self, spliter: Spliter, **kw):
        assert self.op_name == opns.TUPLE
        tail_outs = dict(zip(spliter.seg_names, self.args))

        # print(spliter.seg_names)

        assert spliter.head is not None
        head_params = {k: to_ndarray(v) \
                for k, v in spliter.head_params.items()}
        # head_params.update(self.params)
        head = load_json(spliter.head, params=head_params)

        # helper.format_print(head, head_params)

        def _merge(sym: Symbol):
            if op.is_input(sym, head_params):
                assert sym.name in tail_outs
                return tail_outs[sym.name]
            return sym
        out = transform(head, _merge)

        return out.like(self, params={ **head_params, **self.params })


