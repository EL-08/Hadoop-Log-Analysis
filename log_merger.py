import csv
import os
from datetime import datetime, timedelta

EPOCH = datetime(1970, 1, 1)


def parse_timestamp(date, time):
    # Turns the Date and Time columns (e.g. 2015-10-18 and 18:01:51,775) into milliseconds
    moment = datetime.strptime(date + " " + time, "%Y-%m-%d %H:%M:%S,%f")
    return (moment - EPOCH) // timedelta(milliseconds=1)


def format_timestamp(timestampMs):
    # Turns milliseconds back into the dataset's timestamp format for printing
    moment = EPOCH + timedelta(milliseconds=timestampMs)
    return moment.strftime("%Y-%m-%d %H:%M:%S,") + f"{timestampMs % 1000:03d}"


def read_log_stream(path, readCounter=None):
    # Generator that yields one entry at a time so a file is never fully loaded, readCounter tracks rows read per file
    source = os.path.splitext(os.path.basename(path))[0]  # p0.csv -> p0
    with open(path, newline="", encoding="utf-8") as logFile:
        reader = csv.reader(logFile)
        header = next(reader)
        dateIndex = header.index("Date")
        timeIndex = header.index("Time")
        for row in reader:
            if readCounter is not None:
                readCounter[source] = readCounter.get(source, 0) + 1
            yield parse_timestamp(row[dateIndex], row[timeIndex]), source, row


def list_stream(source, timestamps):
    # Wraps a list as a stream so the examples and tests use the same merge code
    for timestamp in timestamps:
        yield timestamp, source, [str(timestamp)]


def two_way_merge(left, right):
    # Lazily merges two sorted streams, only holding the current entry of each one
    leftEntry = next(left, None)
    rightEntry = next(right, None)

    while leftEntry is not None and rightEntry is not None:
        if leftEntry[0] <= rightEntry[0]:  # Left wins ties so the merge is stable
            yield leftEntry
            leftEntry = next(left, None)
        else:
            yield rightEntry
            rightEntry = next(right, None)

    # Append whatever is left in the other stream
    while leftEntry is not None:
        yield leftEntry
        leftEntry = next(left, None)
    while rightEntry is not None:
        yield rightEntry
        rightEntry = next(right, None)


def k_way_merge(streams, labels=None, trace=None, depth=0):
    # Split the streams into two halves, merge each half recursively and then two-way merge the results
    if len(streams) == 0:
        return iter(())
    if labels is None:
        labels = [f"S{i}" for i in range(len(streams))]
    indent = "  " * depth

    if len(streams) == 1:  # A single stream is already sorted
        if trace is not None:
            trace.append(f"{indent}base case: {labels[0]}")
        return streams[0]

    mid = len(streams) // 2
    if trace is not None:
        trace.append(f"{indent}divide {labels} -> {labels[:mid]} | {labels[mid:]}")
    leftStream = k_way_merge(streams[:mid], labels[:mid], trace, depth + 1)
    rightStream = k_way_merge(streams[mid:], labels[mid:], trace, depth + 1)
    if trace is not None:
        trace.append(f"{indent}two-way merge {labels[:mid]} + {labels[mid:]}")
    return two_way_merge(leftStream, rightStream)


def merge_log_files(paths, outputPath):
    # Merges every log file into merged_log.csv one row at a time and tracks how many entries are in memory
    with open(paths[0], newline="", encoding="utf-8") as firstFile:
        header = next(csv.reader(firstFile))
    header[0] = "RowIndex"  # The first column has no name in the original files

    readCounter = {}
    streams = [read_log_stream(path, readCounter) for path in paths]
    entryCount = 0
    peakInMemory = 0
    firstOutputReads = {}
    entriesPerSource = {}

    with open(outputPath, "w", newline="", encoding="utf-8") as outputFile:
        writer = csv.writer(outputFile)
        writer.writerow(["Source"] + header)
        for _, source, row in k_way_merge(streams):
            if entryCount == 0:
                firstOutputReads = dict(readCounter)
            inMemory = sum(readCounter.values()) - entryCount  # Entries read but not written yet
            peakInMemory = max(peakInMemory, inMemory)
            writer.writerow([source] + row)
            entryCount += 1
            entriesPerSource[source] = entriesPerSource.get(source, 0) + 1

    return entryCount, entriesPerSource, peakInMemory, firstOutputReads
