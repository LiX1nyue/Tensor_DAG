import json

from annotate_trace import annotate


def test_annotation_preserves_sequence_number():
    trace = [{
        "name": "HcclAllreduce",
        "ts": "1000.0",
        "args": {"Sequence number": -1},
    }]
    calls = [{
        "trace_id": "coll:abc:00000000",
        "group_key": "ranks=0,1",
        "group_seq": 0,
        "op": "all_reduce",
        "rank": 0,
        "call_start_us": 1000.2,
    }]
    assert annotate(trace, calls, 5) == 1
    assert trace[0]["args"]["Sequence number"] == -1
    assert trace[0]["args"]["collective_trace_id"] == "coll:abc:00000000"
