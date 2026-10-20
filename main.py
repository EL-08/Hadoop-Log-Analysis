import csv
import glob
import os
import time

from analysis import (burst_composition, burst_sensitivity, hop_distances, profile_components, strassen_benchmark,
                      top_transitions)
from burst_detection import detect_bursts, load_timestamps, merge_overlapping
from log_merger import format_timestamp, merge_log_files
from plots import plot_burst_timeline, plot_dependency_heatmaps
from strassen_dependency import (build_transition_counts, format_matrix, naive_multiply, next_power_of_two, save_matrix,
                                 strassen_multiply, to_binary)

# Runs Parts 1-3 on the data/ folder and saves everything printed to outputs/results_summary.txt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
MERGED_PATH = os.path.join(OUTPUT_DIR, "merged_log.csv")

SPAN_LIMIT_MS = 1000  # T, a burst spans less than 1 second
MIN_EVENTS = 10  # M, a burst has at least 10 events
BASE_SIZE = 32  # B, partitions with at most 32 events are base cases

# Values tried in the sensitivity analysis
SENSITIVITY_T = [250, 500, 1000, 2000]
SENSITIVITY_M = [5, 10, 20]
SENSITIVITY_B = [16, 32, 64]

os.makedirs(OUTPUT_DIR, exist_ok=True)
summaryLines = []


def report(line=""):
    # Prints a line and keeps a copy for results_summary.txt
    print(line)
    summaryLines.append(line)


logPaths = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")))
components = [os.path.splitext(os.path.basename(path))[0] for path in logPaths]
n = len(components)

# Dataset overview
report("=== Dataset: one log file per system component ===")
profiles = profile_components(logPaths)
report(f"{'File':<5} {'Entries':>7}  {'First timestamp':<23} {'Last timestamp':<23} {'Main Hadoop class (share)':<40} Levels")
for component in components:
    count, firstMs, lastMs, mainComponent, mainShare, levels = profiles[component]
    mainText = f"{mainComponent} ({mainShare:.0%})"
    report(f"{component:<5} {count:>7}  {format_timestamp(firstMs):<23} {format_timestamp(lastMs):<23} "
           f"{mainText:<40} {levels}")

# Part 1
report("\n=== Part 1: Divide-and-Conquer K-Way Merge (lazy loading) ===")
start = time.perf_counter()
entryCount, entriesPerSource, peakInMemory, firstOutputReads = merge_log_files(logPaths, MERGED_PATH)
mergeSeconds = time.perf_counter() - start
inputTotal = sum(profile[0] for profile in profiles.values())

report(f"Merged {n} streams ({', '.join(components)}) into {entryCount} entries in {mergeSeconds * 1000:.1f} ms")
report(f"Input entries: {inputTotal}  Output entries: {entryCount}  No entries lost: {inputTotal == entryCount}")
report(f"Entries per source in output: {entriesPerSource}")
report(f"Rows read from each file before the first merged entry was written: {firstOutputReads}")
report(f"Peak number of entries held in memory during the merge: {peakInMemory} "
       f"(K = {n} streams, total entries = {entryCount})")

timestamps = load_timestamps(MERGED_PATH)
report(f"Output is chronologically sorted: {timestamps == sorted(timestamps)}")
report(f"Output: {MERGED_PATH}")

# Part 2
report("\n=== Part 2: Fault Dependency Analysis with Strassen's Algorithm ===")
transitionCounts = build_transition_counts(MERGED_PATH, components)
matrixA = to_binary(transitionCounts)
strassenCounter = [0]
matrixA2 = strassen_multiply(matrixA, matrixA, 1, strassenCounter)
matrixA3 = strassen_multiply(matrixA2, matrixA, 1, strassenCounter)
naiveA2 = naive_multiply(matrixA, matrixA)
naiveA3 = naive_multiply(naiveA2, matrixA)

padded = next_power_of_two(n)
report(f"Matrix size: {n} x {n} (not a power of 2, padded to {padded} x {padded} inside Strassen)")
for title, matrix in [("Transition counts (how many times Y immediately followed X):", transitionCounts),
                      ("Dependency matrix A (1 = direct dependency X -> Y):", matrixA),
                      ("A^2 (number of two-hop walks X -> ? -> Y):", matrixA2),
                      ("A^3 (number of three-hop walks X -> ? -> ? -> Y):", matrixA3)]:
    report("\n" + title)
    report(format_matrix(matrix, components))
report(f"\nVerification: Strassen A^2 == naive A^2: {matrixA2 == naiveA2}, "
       f"Strassen A^3 == naive A^3: {matrixA3 == naiveA3}")
report(f"Scalar multiplications used by Strassen for A^2 and A^3 together: {strassenCounter[0]} "
       f"(naive on {n} x {n} would use {2 * n ** 3})")

report("\nStrongest direct dependencies (by number of transitions):")
for source, target, count, share in top_transitions(transitionCounts, components, 8):
    report(f"  {source} -> {target}: {count} transitions ({share:.0%} of events after {source})")

report("\nShortest dependency distance between ordered component pairs:")
for group, pairs in hop_distances(matrixA, matrixA2, matrixA3, components).items():
    pairText = "(see A)" if group == "1-hop" else ", ".join(f"{x}->{y}" for x, y in pairs)
    report(f"  {group}: {len(pairs)} pairs {pairText}")

for name, matrix in [("dependency_matrix_A", matrixA), ("dependency_matrix_A2", matrixA2),
                     ("dependency_matrix_A3", matrixA3), ("transition_counts", transitionCounts)]:
    save_matrix(matrix, components, os.path.join(OUTPUT_DIR, name + ".csv"))
plot_dependency_heatmaps([matrixA, matrixA2, matrixA3], ["A (direct)", "A² (two-hop)", "A³ (three-hop)"], components,
                         os.path.join(OUTPUT_DIR, "dependency_heatmaps.png"))

report("\nStrassen vs naive multiplication on random 0/1 matrices (leaf size 8):")
report(f"  {'n':>4} {'padded':>6} {'Strassen (s)':>13} {'naive (s)':>10} {'Strassen mults':>15} {'naive mults':>12} match")
for row in strassen_benchmark([7, 16, 33, 64, 100, 128], 8):
    size, paddedSize, strassenSeconds, naiveSeconds, strassenMults, naiveMults, matches = row
    report(f"  {size:>4} {paddedSize:>6} {strassenSeconds:>13.4f} {naiveSeconds:>10.4f} {strassenMults:>15} "
           f"{naiveMults:>12} {matches}")

# Part 3
report("\n=== Part 3: Divide-and-Conquer Burst Detection ===")
report(f"Parameters: T = {SPAN_LIMIT_MS} ms, M = {MIN_EVENTS} events, B = {BASE_SIZE} entries")
burstCount, bursts = detect_bursts(timestamps, SPAN_LIMIT_MS, MIN_EVENTS, BASE_SIZE)
localCount = sum(1 for burst in bursts if burst[3] == "local")
regions = merge_overlapping(bursts)
insideCount = sum(1 for t in timestamps if any(s <= t <= e for s, e in regions))

report(f"Total bursts detected: {burstCount} ({localCount} local, {burstCount - localCount} cross-boundary)")
report(f"Distinct bursty periods after merging overlapping bursts: {len(regions)}")
report(f"Events inside at least one burst: {insideCount} of {len(timestamps)}")
report("\nList of bursts:")
report(f"  {'#':>2} {'Start':<23} {'End':<23} {'Span ms':>7} {'Events':>6} {'Kind':<15} Components in burst")
with open(os.path.join(OUTPUT_DIR, "bursts.csv"), "w", newline="", encoding="utf-8") as burstFile:
    writer = csv.writer(burstFile)
    writer.writerow(["Start", "End", "SpanMs", "EventCount", "Kind", "Components"])
    compositions = burst_composition(bursts, MERGED_PATH)
    for index, (startMs, endMs, eventCount, kind) in enumerate(bursts):
        componentText = " ".join(f"{name}:{count}" for name, count in compositions[index].most_common())
        startText = format_timestamp(startMs)
        endText = format_timestamp(endMs)
        writer.writerow([startText, endText, endMs - startMs, eventCount, kind, componentText])
        report(f"  {index + 1:>2} {startText:<23} {endText:<23} {endMs - startMs:>7} {eventCount:>6} {kind:<15} "
               f"{componentText}")

report("\nParameter sensitivity (total = local + cross; regions = distinct bursty periods):")
report(f"  {'T ms':>5} {'M':>3} {'B':>3} {'total':>5} {'local':>5} {'cross':>5} {'regions':>7}")
for row in burst_sensitivity(timestamps, SENSITIVITY_T, SENSITIVITY_M, SENSITIVITY_B):
    spanLimit, minEvents, baseSize, total, local, cross, regionCount = row
    report(f"  {spanLimit:>5} {minEvents:>3} {baseSize:>3} {total:>5} {local:>5} {cross:>5} {regionCount:>7}")

zoomStartMs = bursts[0][0] - 1500
plot_burst_timeline(timestamps, bursts, zoomStartMs, zoomStartMs + 9000,
                    os.path.join(OUTPUT_DIR, "burst_timeline.png"))

report(f"\nOutputs written to {OUTPUT_DIR}: merged_log.csv, dependency_matrix_A*.csv, transition_counts.csv, "
       f"bursts.csv, dependency_heatmaps.png, burst_timeline.png, results_summary.txt")

with open(os.path.join(OUTPUT_DIR, "results_summary.txt"), "w", encoding="utf-8") as summaryFile:
    summaryFile.write("\n".join(summaryLines) + "\n")
