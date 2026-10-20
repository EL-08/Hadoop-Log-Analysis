# Divide-and-Conquer Analysis of Distributed Hadoop Logs

This project applies divide-and-conquer algorithms to the Hadoop logs of seven system components. A lazy K-way merge builds one global timeline from the seven sorted files (Part 1), Strassen's algorithm computes the two-hop and three-hop dependencies between components (Part 2), and a median-split recursion detects bursts of log activity (Part 3).

## Report
The full write-up is in [Report.md](Report.md), which covers the code structure, the algorithms, the verification examples, the execution results and an analysis of the job's failure.

## Project Structure
```
data/                     p0.csv ... p6.csv, one sorted log file per component
log_merger.py             Part 1: lazy loading + divide-and-conquer K-way merge
strassen_dependency.py    Part 2: dependency matrix + Strassen multiplication (any n)
burst_detection.py        Part 3: divide-and-conquer burst detection
analysis.py, plots.py     Result tables and figures
main.py                   Runs Parts 1-3 on the dataset and writes outputs/
examples.py               Hand-checkable examples with step-by-step traces
tests/test_algorithms.py  23 automated tests
outputs/                  merged_log.csv, dependency matrices, bursts.csv, figures, results_summary.txt
```

## How to Run
```
pip install -r requirements.txt
python main.py               # full pipeline on the dataset
python examples.py           # traced examples with PASS/FAIL
python -m pytest tests       # automated tests
```

The burst parameters T, M and B are set near the top of `main.py` as `SPAN_LIMIT_MS`, `MIN_EVENTS` and `BASE_SIZE`.
