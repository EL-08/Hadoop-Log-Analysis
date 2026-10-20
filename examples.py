from burst_detection import detect_bursts
from log_merger import k_way_merge, list_stream
from strassen_dependency import (count_transitions, format_matrix, naive_multiply, next_power_of_two, pad_matrix,
                                 strassen_multiply, to_binary)

# Small examples for every algorithm that print the trace, the answer worked out by hand and the actual answer


def show_check(label, expected, actual):
    print(f"  Expected {label}: {expected}")
    print(f"  Actual   {label}: {actual}")
    print("  Result: " + ("PASS" if expected == actual else "FAIL"))


def counted_stream(name, values):
    # Counts how many entries the merge has pulled from each stream to check that it is lazy
    for entry in list_stream(name, values):
        readCounter[name] = readCounter.get(name, 0) + 1
        yield entry


# Example 1: K-way merge of four tiny sorted streams
print("=== Example 1: Divide-and-Conquer K-Way Merge ===")
exampleStreams = {"A": [1, 5, 9], "B": [2, 6], "C": [3, 4, 10], "D": [7, 8]}
for name, values in exampleStreams.items():
    print(f"  Stream {name}: {values}")

readCounter = {}
mergeTrace = []
streams = [counted_stream(name, values) for name, values in exampleStreams.items()]
merged = k_way_merge(streams, list(exampleStreams), mergeTrace)
print("  Recursion tree:")
for line in mergeTrace:
    print("    " + line)

firstEntry = next(merged)
print(f"  Entries read from each stream when the first output ({firstEntry[0]} from {firstEntry[1]}) is produced: "
      f"{readCounter}")
mergedOrder = [firstEntry] + list(merged)
show_check("merged timestamps", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [entry[0] for entry in mergedOrder])
show_check("source order", list("ABCCABDDAC"), [entry[1] for entry in mergedOrder])

# Example 2: Strassen on 2x2 to show the seven products and on 3x3 to show the padding
print("\n=== Example 2: Strassen's Matrix Multiplication ===")
a2 = [[1, 2], [3, 4]]
b2 = [[5, 6], [7, 8]]
m1 = (1 + 4) * (5 + 8)
m2 = (3 + 4) * 5
m3 = 1 * (6 - 8)
m4 = 4 * (7 - 5)
m5 = (1 + 2) * 8
m6 = (3 - 1) * (5 + 6)
m7 = (2 - 4) * (7 + 8)
print(f"  A = {a2}, B = {b2}")
print(f"  M1..M7 = {m1}, {m2}, {m3}, {m4}, {m5}, {m6}, {m7}")
print(f"  C11 = M1+M4-M5+M7 = {m1 + m4 - m5 + m7}, C12 = M3+M5 = {m3 + m5}, "
      f"C21 = M2+M4 = {m2 + m4}, C22 = M1-M2+M3+M6 = {m1 - m2 + m3 + m6}")
show_check("2x2 product", [[19, 22], [43, 50]], strassen_multiply(a2, b2))

a3 = [[1, 2, 0], [0, 1, 3], [4, 0, 1]]
b3 = [[2, 0, 1], [1, 3, 0], [0, 1, 2]]
paddedSize = next_power_of_two(3)
print(f"\n  A = {a3}, B = {b3}  (n = 3, padded to {paddedSize} x {paddedSize})")
print("  Padded A:")
for row in pad_matrix(a3, paddedSize):
    print(f"    {row}")
show_check("3x3 product (hand computed)", [[4, 6, 1], [1, 6, 6], [8, 1, 6]], strassen_multiply(a3, b3))
show_check("3x3 product (naive)", naive_multiply(a3, b3), strassen_multiply(a3, b3))

# Example 3: dependency matrix, A^2 and A^3 from a tiny merged timeline
print("\n=== Example 3: Dependency Matrix, A^2 and A^3 ===")
exampleComponents = ["X", "Y", "Z"]
exampleTimeline = ["X", "Y", "Y", "Z", "X"]
print(f"  Merged timeline sources: {' -> '.join(exampleTimeline)}")
print("  Transitions: X->Y, Y->Y, Y->Z, Z->X")
exampleA = to_binary(count_transitions(exampleTimeline, exampleComponents))
exampleA2 = strassen_multiply(exampleA, exampleA)
exampleA3 = strassen_multiply(exampleA2, exampleA)
for title, matrix in [("A", exampleA), ("A^2 (number of 2-hop paths)", exampleA2),
                      ("A^3 (number of 3-hop paths)", exampleA3)]:
    print(f"  {title}:")
    print("    " + format_matrix(matrix, exampleComponents).replace("\n", "\n    "))
show_check("A", [[0, 1, 0], [0, 1, 1], [1, 0, 0]], exampleA)
show_check("A^2", [[0, 1, 1], [1, 1, 1], [0, 1, 0]], exampleA2)
show_check("A^3", [[1, 1, 1], [1, 2, 1], [0, 1, 1]], exampleA3)

# Example 4: burst detection with one local burst and one burst only the combine step can find
print("\n=== Example 4: Divide-and-Conquer Burst Detection ===")
exampleTimestamps = [0, 10, 20, 500, 900, 960, 1010, 1040, 3000, 3500, 4000, 4500]
print(f"  Timestamps (ms): {exampleTimestamps}")
print("  Parameters: T = 100 ms, M = 3 events, B = 4 events")
burstTrace = []
exampleCount, exampleBursts = detect_bursts(exampleTimestamps, 100, 3, 4, burstTrace)
print("  Recursion trace:")
for line in burstTrace:
    print("    " + line)
show_check("burst count", 2, exampleCount)
show_check("bursts", [(0, 20, 3, "local"), (960, 1040, 3, "cross-boundary")], exampleBursts)
