# Collective trace IDs

This directory contains a Python-only experiment for the MindSpeed-LLM/CANN 9.0.0
environment. It records one ID at each Python collective call and, after profiling,
adds that ID to the matching existing event in `trace_view.json` under `args`.

It does not modify CANN, HCCL, torch_npu, or the communication implementation.

## What is added

Each participating rank writes a JSONL record such as:

```json
{"trace_id":"coll:g8f...:000017","group_key":"ranks=0,1,2,3","group_seq":17,"op":"all_reduce","rank":0}
```

The postprocessor adds these fields only to a matching trace event whose `cat` is
`cpu_op` and `ph` is `X`. Other categories and phases, including lower-level
`hcom_*` events, are ignored:

```json
"args": {
  "Sequence number": -1,
  "collective_trace_id": "coll:g8f...:000017",
  "collective_group": "ranks=0,1,2,3",
  "collective_seq": 17,
  "collective_op": "all_reduce",
  "collective_rank": 0
}
```

`Sequence number` is preserved. The ID is unique within one run and the sequence
counter is independent for each deterministic group. The group key is derived from
the sorted global ranks in the process group, so it is the same on every rank. If
one job creates distinct process groups with identical membership, this key merges
them; check for that case during validation and add a stable logical group name if
needed.

## Installation in the training process

Copy this directory into the container and add its parent to `PYTHONPATH`. Import
the installer immediately after `torch.distributed` and `torch_npu` are imported,
before the training loop or model initialization creates collective calls:

```python
from collective_trace_ids import install_collective_trace_ids

install_collective_trace_ids("/path/to/collective_calls.jsonl")
```

For a distributed job, use a rank-specific path, for example:

```python
import os
install_collective_trace_ids(
    f"/data/profile/collective_calls_rank{os.environ.get('RANK', 'unknown')}.jsonl"
)
```

The installer wraps the public `torch.distributed` collective functions listed in
`collector.py`. It preserves positional and keyword arguments, return values,
exceptions, and `async_op` behavior. The record is written at the Python call
boundary after the original call succeeds; it is not a second collective and does
not add an MSTX event. Calls bypassing these public Python APIs, such as lower-
level fused extensions, will not be recorded.

## Postprocess a rank trace

After the profiler has generated a trace, run this on each rank's output:

```bash
python3 annotate_trace.py \
  --trace /data/profile/trace_view.json \
  --calls /data/profile/collective_calls_rank0.jsonl \
  --output /data/profile/trace_view.collective_ids.json
```

The original trace is never overwritten. The script reports unmatched calls and
the number of annotated events. It matches Python call entry time to the nearest
same-operation `cpu_op` complete event in a configurable time window. It never
annotates `hcom_*`, enqueue/dequeue, or device events. This heuristic must
be checked against a server trace: CANN may timestamp HCCL events after Python
enters the call, and fused operations may produce multiple lower-level events.
Postprocessing is required because Python cannot append fields to an already-
created CANN event's `args` object at runtime.

## Server validation requested

Run a short 2-rank or 8-rank test with profiling enabled and confirm:

1. every participating rank creates its JSONL call log;
2. the same `collective_group` and `collective_seq` produce the same
   `collective_trace_id` on all ranks;
3. `trace_view.collective_ids.json` contains those fields under `args` on the
   HCCL/C10D event;
4. the annotated event's original `Sequence number` remains present.
5. no distinct process groups with identical rank membership need disambiguation.

This first version intentionally reports a warning for calls that cannot be mapped
to a trace event. It does not silently claim complete coverage for collectives
that bypass `torch.distributed` or are launched by a lower-level fused extension.
