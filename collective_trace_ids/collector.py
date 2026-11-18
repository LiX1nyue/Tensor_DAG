"""Record deterministic IDs at Python torch.distributed collective boundaries."""

import functools
import hashlib
import json
import os
import threading
import time


_COLLECTIVES = (
    "all_reduce",
    "all_gather",
    "all_gather_into_tensor",
    "reduce_scatter",
    "reduce_scatter_tensor",
    "broadcast",
    "all_to_all",
    "all_to_all_single",
)


class _Collector:
    def __init__(self, path):
        self.path = path
        self.lock = threading.Lock()
        self.seq = {}
        self.fp = None

    def open(self):
        parent = os.path.dirname(os.path.abspath(self.path))
        os.makedirs(parent, exist_ok=True)
        self.fp = open(self.path, "a", encoding="utf-8", buffering=1)

    @staticmethod
    def rank_and_group(group, dist):
        rank = dist.get_rank(group)
        if hasattr(dist, "get_process_group_ranks"):
            ranks = tuple(sorted(dist.get_process_group_ranks(group)))
        elif group is dist.group.WORLD:
            ranks = tuple(range(dist.get_world_size()))
        else:
            raise RuntimeError(
                "This PyTorch version cannot enumerate subgroup ranks; "
                "refusing to generate a misleading group ID"
            )
        group_key = "ranks=" + ",".join(map(str, ranks))
        return rank, group_key

    def record(self, op, group, dist, start_ns):
        rank, group_key = self.rank_and_group(group, dist)
        with self.lock:
            seq = self.seq.get(group_key, 0)
            self.seq[group_key] = seq + 1
            digest = hashlib.sha1(group_key.encode("ascii")).hexdigest()[:12]
            rec = {
                "trace_id": f"coll:{digest}:{seq:08d}",
                "group_key": group_key,
                "group_seq": seq,
                "op": op,
                "rank": rank,
                "world_size": dist.get_world_size(group),
                "pid": os.getpid(),
                "thread_id": threading.get_ident(),
                "call_start_ns": start_ns,
                "call_start_us": start_ns / 1000.0,
            }
            self.fp.write(json.dumps(rec, separators=(",", ":")) + "\n")


def install_collective_trace_ids(path=None):
    """Install wrappers once and return the collector.

    Import this after torch.distributed has been imported. The wrappers are
    intentionally narrow: they only observe public collective entry points.
    """
    import torch.distributed as dist

    existing = getattr(dist, "_collective_trace_id_collector", None)
    if existing is not None:
        return existing
    path = path or os.environ.get("COLLECTIVE_TRACE_LOG", "collective_calls.jsonl")
    collector = _Collector(path)
    collector.open()

    for op in _COLLECTIVES:
        original = getattr(dist, op, None)
        if original is None or getattr(original, "_collective_trace_wrapped", False):
            continue

        @functools.wraps(original)
        def wrapped(*args, __original=original, __op=op, **kwargs):
            start_ns = time.time_ns()
            group = kwargs.get("group")
            if group is None:
                group = dist.group.WORLD
            result = __original(*args, **kwargs)
            return result

        wrapped._collective_trace_wrapped = True
        setattr(dist, op, wrapped)

    dist._collective_trace_id_collector = collector
    return collector
