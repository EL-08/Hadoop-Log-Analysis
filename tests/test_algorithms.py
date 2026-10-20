import os
import random
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)  # Lets pytest import the project modules

from burst_detection import detect_bursts, merge_overlapping
from log_merger import k_way_merge, list_stream, merge_log_files, parse_timestamp
from strassen_dependency import count_transitions, naive_multiply, strassen_multiply, to_binary

DATA_DIR = os.path.join(ROOT, "data")


# Part 1: K-way merge
def test_merge_matches_sorted_for_random_streams():
    random.seed(7)
    for k in range(1, 10):  # Odd and even numbers of streams
        streams = [sorted(random.randint(0, 100) for _ in range(random.randint(0, 15))) for _ in range(k)]
        merged = [entry[0] for entry in k_way_merge([list_stream(f"S{i}", s) for i, s in enumerate(streams)])]
        assert merged == sorted(value for stream in streams for value in stream)


def test_merge_is_stable_on_equal_timestamps():
    merged = k_way_merge([list_stream("A", [5, 5]), list_stream("B", [5]), list_stream("C", [5])])
    assert [entry[1] for entry in merged] == ["A", "A", "B", "C"]  # Earlier streams win ties


def test_merge_handles_empty_inputs():
    assert list(k_way_merge([])) == []
    assert [entry[0] for entry in k_way_merge([list_stream("A", []), list_stream("B", [1])])] == [1]


def test_merge_is_lazy():
    pulled = {"A": 0, "B": 0, "C": 0}

    def counted(name, values):
        for entry in list_stream(name, values):
            pulled[name] += 1
            yield entry

    merged = k_way_merge([counted("A", [1, 4]), counted("B", [2, 5]), counted("C", [3, 6])])
    assert pulled == {"A": 0, "B": 0, "C": 0}  # Building the merge tree reads nothing
    next(merged)
    assert pulled == {"A": 1, "B": 1, "C": 1}  # One entry per stream is enough to produce the first output


def test_merge_log_files_on_dataset(tmp_path):
    paths = sorted(os.path.join(DATA_DIR, name) for name in os.listdir(DATA_DIR) if name.endswith(".csv"))
    entryCount, entriesPerSource, peakInMemory, firstOutputReads = merge_log_files(paths, str(tmp_path / "merged.csv"))
    assert entryCount == 2000
    assert peakInMemory <= len(paths)  # Never more than one buffered entry per stream
    assert all(count == 1 for count in firstOutputReads.values())


def test_parse_timestamp_keeps_milliseconds():
    assert parse_timestamp("2015-10-18", "18:01:51,775") - parse_timestamp("2015-10-18", "18:01:51,000") == 775


# Part 2: Strassen and dependencies
@pytest.mark.parametrize("n", [1, 2, 3, 5, 7, 8, 9, 12, 17])
def test_strassen_matches_naive_for_any_size(n):
    random.seed(n)
    a = [[random.randint(-5, 5) for _ in range(n)] for _ in range(n)]
    b = [[random.randint(-5, 5) for _ in range(n)] for _ in range(n)]
    assert strassen_multiply(a, b) == naive_multiply(a, b)
    assert strassen_multiply(a, b, leafSize=4) == naive_multiply(a, b)


def test_strassen_rejects_non_square():
    with pytest.raises(ValueError):
        strassen_multiply([[1, 2]], [[1], [2]])


def test_dependency_matrix_and_hops():
    a = to_binary(count_transitions(["X", "Y", "Y", "Z", "X"], ["X", "Y", "Z"]))
    assert a == [[0, 1, 0], [0, 1, 1], [1, 0, 0]]
    assert strassen_multiply(a, a) == [[0, 1, 1], [1, 1, 1], [0, 1, 0]]


# Part 3: burst detection
def test_burst_example():
    count, bursts = detect_bursts([0, 10, 20, 500, 900, 960, 1010, 1040, 3000, 3500, 4000, 4500], 100, 3, 4)
    assert count == 2
    assert bursts == [(0, 20, 3, "local"), (960, 1040, 3, "cross-boundary")]


def test_burst_requires_span_strictly_less_than_t():
    assert detect_bursts([0, 50, 100], 100, 3, 4)[0] == 0  # span == T is not a burst
    assert detect_bursts([0, 50, 99], 100, 3, 4)[0] == 1


def test_no_bursts_in_sparse_or_empty_input():
    assert detect_bursts([], 100, 3, 4) == (0, [])
    assert detect_bursts(list(range(0, 10000, 500)), 100, 3, 4) == (0, [])


def test_count_equals_length_of_burst_list():
    random.seed(3)
    timestamps = sorted(random.randint(0, 20000) for _ in range(500))
    count, bursts = detect_bursts(timestamps, 200, 5, 16)
    assert count == len(bursts)
    for start, end, eventCount, kind in bursts:
        assert end - start < 200 and eventCount >= 5


def test_bursts_do_not_overlap():
    random.seed(3)
    timestamps = sorted(random.randint(0, 20000) for _ in range(500))
    for spanLimit, minEvents, baseSize in [(200, 5, 16), (1000, 10, 32), (2000, 5, 8)]:
        count, bursts = detect_bursts(timestamps, spanLimit, minEvents, baseSize)
        assert sum(burst[2] for burst in bursts) <= len(timestamps)
        for previous, current in zip(bursts, bursts[1:]):
            assert current[0] >= previous[1]


def test_merge_overlapping_regions():
    assert merge_overlapping([(0, 10, 3, "local"), (5, 20, 4, "cross-boundary"), (30, 40, 3, "local")]) == \
        [(0, 20), (30, 40)]
