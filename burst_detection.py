from bisect import bisect_left, bisect_right

from log_merger import read_log_stream


def load_timestamps(mergedPath):
    # Returns the timestamp in ms of every event in the merged log
    return [timestampMs for timestampMs, _, _ in read_log_stream(mergedPath)]


def detect_bursts(timestamps, spanLimit, minEvents, baseSize, trace=None):
    # spanLimit is T, minEvents is M and baseSize is B, returns the number of bursts and the list of bursts

    def local_burst(left, right):
        # The whole partition [left, right) is a burst if it has at least M events and spans less than T
        span = timestamps[right - 1] - timestamps[left]  # Sorted, so first and last are min and max
        if right - left >= minEvents and span < spanLimit:
            return timestamps[left], timestamps[right - 1], right - left, "local"
        return None

    def cross_boundary_burst(left, mid, right, freeStart, freeEnd):
        # Only events within T of the boundary can form a burst across it, like the strip in closest pair
        boundary = timestamps[mid]
        stripStart = max(bisect_left(timestamps, boundary - spanLimit, left, mid), freeStart)
        stripEnd = min(bisect_right(timestamps, boundary + spanLimit, mid, right), freeEnd)

        # Sliding window over the strip to find the largest group with span < T that has events from both halves
        bestCount = 0
        bestStart = bestEnd = 0
        i = stripStart
        for j in range(mid, stripEnd):
            while timestamps[j] - timestamps[i] >= spanLimit:  # Shrink the window until its span is less than T
                i += 1
            if i < mid and j - i + 1 > bestCount:  # i < mid means the window still has left half events
                bestCount = j - i + 1
                bestStart, bestEnd = i, j

        if bestCount >= minEvents:
            return (timestamps[bestStart], timestamps[bestEnd], bestCount, "cross-boundary"), bestStart, bestEnd
        return None

    def solve(left, right, depth):
        indent = "  " * depth

        if right - left <= baseSize:  # Base case, test the whole partition for a local burst
            burst = local_burst(left, right)
            if trace is not None:
                verdict = "LOCAL BURST" if burst is not None else "no burst"
                span = timestamps[right - 1] - timestamps[left]
                trace.append(f"{indent}base {timestamps[left:right]} n={right - left} span={span} -> {verdict}")
            if burst is None:
                return 0, [], None, None
            return 1, [burst], left, right - 1

        # Split at the median timestamp and find the bursts in each half recursively
        mid = (left + right) // 2
        if trace is not None:
            trace.append(f"{indent}divide {timestamps[left:right]} at median {timestamps[mid]}")
        leftCount, leftBursts, leftFirst, leftLast = solve(left, mid, depth + 1)
        rightCount, rightBursts, rightFirst, rightLast = solve(mid, right, depth + 1)
        burstCount = leftCount + rightCount
        burstList = leftBursts + rightBursts

        # Add one more burst if events on both sides of the boundary form a burst together
        freeStart = left if leftLast is None else leftLast + 1
        freeEnd = right if rightFirst is None else rightFirst
        found = cross_boundary_burst(left, mid, right, freeStart, freeEnd)
        crossBurst = None
        if found is not None:
            crossBurst, crossFirst, crossLast = found
            burstCount += 1
            burstList.append(crossBurst)
            leftFirst = crossFirst if leftFirst is None else leftFirst
            rightLast = crossLast if rightLast is None else rightLast
        if trace is not None:
            verdict = f"CROSS BURST {crossBurst[:3]}" if crossBurst is not None else "no cross burst"
            trace.append(f"{indent}combine at {timestamps[mid]}: {verdict} -> total {burstCount}")
        first = leftFirst if leftFirst is not None else rightFirst
        last = rightLast if rightLast is not None else leftLast
        return burstCount, burstList, first, last

    timestamps = sorted(timestamps)  # The merged log is already sorted so this is O(n)
    if len(timestamps) == 0:
        return 0, []
    totalCount, bursts, _, _ = solve(0, len(timestamps), 0)
    bursts.sort(key=lambda burst: (burst[0], burst[1]))
    return totalCount, bursts


def merge_overlapping(bursts):
    # Merges overlapping bursts into separate time regions to show how many bursty periods there really are
    regions = []
    for start, end, eventCount, kind in sorted(bursts):
        if regions and start <= regions[-1][1]:
            regions[-1] = (regions[-1][0], max(regions[-1][1], end))
        else:
            regions.append((start, end))
    return regions
