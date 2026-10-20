import os
import random
import time
from collections import Counter

from burst_detection import detect_bursts, merge_overlapping
from log_merger import read_log_stream
from strassen_dependency import naive_multiply, next_power_of_two, strassen_multiply


def profile_components(paths):
    # Summarizes what each log file contains
    profiles = {}
    for path in paths:
        source = os.path.splitext(os.path.basename(path))[0]
        classes = Counter()
        levels = Counter()
        count = 0
        for timestampMs, _, row in read_log_stream(path):
            if count == 0:
                firstMs = timestampMs
            lastMs = timestampMs
            levels[row[4]] += 1  # Level column
            classes[row[6].split(".")[-1]] += 1  # Component column, org.apache.hadoop.ipc.Client -> Client
            count += 1
        mainComponent, mainCount = classes.most_common(1)[0]
        profiles[source] = (count, firstMs, lastMs, mainComponent, mainCount / count, dict(levels))
    return profiles


def top_transitions(counts, components, limit):
    # Ranks the X -> Y dependencies by how often Y came right after X
    ranked = []
    for i, source in enumerate(components):
        outgoing = sum(counts[i])
        for j, target in enumerate(components):
            if counts[i][j] > 0:
                ranked.append((source, target, counts[i][j], counts[i][j] / outgoing))  # Share of X's outgoing events
    ranked.sort(key=lambda item: item[2], reverse=True)
    return ranked[:limit]


def hop_distances(a, a2, a3, components):
    # Groups every pair (X, Y) by the fewest hops needed to get from X to Y
    groups = {"1-hop": [], "2-hop only": [], "3-hop only": [], "unreachable": []}
    for i, source in enumerate(components):
        for j, target in enumerate(components):
            if a[i][j] > 0:
                groups["1-hop"].append((source, target))
            elif a2[i][j] > 0:
                groups["2-hop only"].append((source, target))
            elif a3[i][j] > 0:
                groups["3-hop only"].append((source, target))
            else:
                groups["unreachable"].append((source, target))
    return groups


def burst_composition(bursts, mergedPath):
    # Counts which components produced the events in each burst's time range
    compositions = [Counter() for _ in bursts]
    for timestampMs, _, row in read_log_stream(mergedPath):
        for index, (start, end, _, _) in enumerate(bursts):
            if start <= timestampMs <= end:
                compositions[index][row[0]] += 1
    return compositions


def burst_sensitivity(timestamps, spanLimits, minEventsList, baseSizes):
    # Runs burst detection for every combination of T, M and B
    rows = []
    for spanLimit in spanLimits:
        for minEvents in minEventsList:
            for baseSize in baseSizes:
                total, bursts = detect_bursts(timestamps, spanLimit, minEvents, baseSize)
                local = sum(1 for burst in bursts if burst[3] == "local")
                regions = len(merge_overlapping(bursts))  # Bursty periods left after merging overlapping bursts
                rows.append((spanLimit, minEvents, baseSize, total, local, total - local, regions))
    return rows


def strassen_benchmark(sizes, leafSize):
    # Times Strassen against naive multiplication on random 0/1 matrices
    random.seed(1)
    rows = []
    for n in sizes:
        a = [[random.randint(0, 1) for _ in range(n)] for _ in range(n)]
        b = [[random.randint(0, 1) for _ in range(n)] for _ in range(n)]

        strassenCounter = [0]
        start = time.perf_counter()
        strassenResult = strassen_multiply(a, b, leafSize, strassenCounter)
        strassenSeconds = time.perf_counter() - start

        naiveCounter = [0]
        start = time.perf_counter()
        naiveResult = naive_multiply(a, b, naiveCounter)
        naiveSeconds = time.perf_counter() - start

        rows.append((n, next_power_of_two(n), strassenSeconds, naiveSeconds, strassenCounter[0], naiveCounter[0],
                     strassenResult == naiveResult))
    return rows
