#!/usr/bin/env python3
"""Add Python collective IDs to existing CANN/C10D trace events."""

import argparse
import bisect
import json
import sys


_EVENT_NAMES = {
    "all_reduce": ("HcclAllreduce", "c10d::allreduce_", "hcom_allReduce"),
    "all_gather": ("HcclAllGather", "HcclAllgatherBase", "hcom_allGather"),
    "all_gather_into_tensor": ("HcclAllGather", "HcclAllgatherBase", "hcom_allGather"),
    "reduce_scatter": ("HcclReduceScatter", "HcclReduceScatterBase", "c10d::_reduce_scatter_base_", "hcom_reduceScatter"),
    "reduce_scatter_tensor": ("HcclReduceScatter", "HcclReduceScatterBase", "c10d::_reduce_scatter_base_", "hcom_reduceScatter"),
    "broadcast": ("HcclBroadcast", "hcom_broadcast"),
    "all_to_all": ("HcclAlltoAll", "hcom_allToAll"),
    "all_to_all_single": ("HcclAlltoAll", "hcom_allToAll"),
    "barrier": ("HcclBarrier", "hcom_barrier"),
}


def load_calls(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def event_ts(event):
    try:
        return float(event.get("ts"))
    except (TypeError, ValueError):
        return None


def annotate(trace, calls, tolerance_us):
    indexed = []
    for index, event in enumerate(trace):
        ts = event_ts(event)
        if ts is not None:
            indexed.append((ts, index, event))
    indexed.sort(key=lambda item: item[0])
    timestamps = [item[0] for item in indexed]
    used = set()
    matched = 0
    for call in calls:
        names = _EVENT_NAMES.get(call["op"], ())
        start = float(call["call_start_us"])
        candidates = []
        left = bisect.bisect_left(timestamps, start - tolerance_us)
        right = bisect.bisect_right(timestamps, start + tolerance_us)
        for ts, index, event in indexed[left:right]:
            if (index in used
                    or event.get("cat") != "cpu_op"
                    or event.get("ph") != "X"
                    or event.get("name") not in names):
                continue
            delta = ts - start
            # Prefer canonical Python C10D events and then HCCL API events over
            # lower-level hcom implementation events.
            priority = names.index(event.get("name"))
            candidates.append((priority, abs(delta), index))
        if not candidates:
            continue
        _, _, index = min(candidates)
        event = trace[index]
        args = event.setdefault("args", {})
        args.update({
            "collective_trace_id": call["trace_id"],
            "collective_group": call["group_key"],
            "collective_seq": call["group_seq"],
            "collective_op": call["op"],
            "collective_rank": call["rank"],
        })
        used.add(index)
        matched += 1
    return matched


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", required=True)
    parser.add_argument("--calls", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tolerance-us", type=float, default=5000.0)
    args = parser.parse_args()
    with open(args.trace, encoding="utf-8") as f:
        trace = json.load(f)
    calls = load_calls(args.calls)
    matched = annotate(trace, calls, args.tolerance_us)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(trace, f, ensure_ascii=False, separators=(",", ":"))
    print(f"calls={len(calls)} matched={matched} unmatched={len(calls) - matched}")
    return 0 if matched == len(calls) else 2


if __name__ == "__main__":
    sys.exit(main())
