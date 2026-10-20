import csv


def count_transitions(sources, components):
    # counts[x][y] is how many times an event from Y came right after an event from X
    position = {}
    for index, component in enumerate(components):
        position[component] = index

    counts = [[0] * len(components) for _ in components]
    previousSource = None
    for source in sources:
        if previousSource is not None:
            counts[position[previousSource]][position[source]] += 1
        previousSource = source
    return counts


def build_transition_counts(mergedPath, components):
    with open(mergedPath, newline="", encoding="utf-8") as mergedFile:
        rows = list(csv.reader(mergedFile))
    return count_transitions([row[0] for row in rows[1:]], components)


def to_binary(m):
    return [[1 if value != 0 else 0 for value in row] for row in m]


def build_dependency_matrix(mergedPath, components):
    # A[x][y] is 1 if Y ever came right after X
    return to_binary(build_transition_counts(mergedPath, components))


def add_matrix(a, b):
    return [[a[i][j] + b[i][j] for j in range(len(a))] for i in range(len(a))]


def subtract_matrix(a, b):
    return [[a[i][j] - b[i][j] for j in range(len(a))] for i in range(len(a))]


def naive_multiply(a, b, counter=None):
    # Regular O(n^3) multiplication used as Strassen's base case and to check its answers
    n = len(a)
    result = [[0] * n for _ in range(n)]
    for i in range(n):
        for k in range(n):
            for j in range(n):
                result[i][j] += a[i][k] * b[k][j]
    if counter is not None:
        counter[0] += n * n * n
    return result


def split_matrix(m):
    # Splits an n x n matrix into its four quadrants
    mid = len(m) // 2
    topLeft = [row[:mid] for row in m[:mid]]
    topRight = [row[mid:] for row in m[:mid]]
    bottomLeft = [row[:mid] for row in m[mid:]]
    bottomRight = [row[mid:] for row in m[mid:]]
    return topLeft, topRight, bottomLeft, bottomRight


def join_matrix(c11, c12, c21, c22):
    topRows = [c11[i] + c12[i] for i in range(len(c11))]
    bottomRows = [c21[i] + c22[i] for i in range(len(c21))]
    return topRows + bottomRows


def strassen_recursive(a, b, leafSize, counter=None):
    # Uses 7 recursive products instead of 8, so T(n) = 7T(n/2) + O(n^2) = O(n^2.81), n must be a power of two
    if len(a) <= leafSize:
        return naive_multiply(a, b, counter)

    a11, a12, a21, a22 = split_matrix(a)
    b11, b12, b21, b22 = split_matrix(b)

    m1 = strassen_recursive(add_matrix(a11, a22), add_matrix(b11, b22), leafSize, counter)  # (A11 + A22)(B11 + B22)
    m2 = strassen_recursive(add_matrix(a21, a22), b11, leafSize, counter)  # (A21 + A22)B11
    m3 = strassen_recursive(a11, subtract_matrix(b12, b22), leafSize, counter)  # A11(B12 - B22)
    m4 = strassen_recursive(a22, subtract_matrix(b21, b11), leafSize, counter)  # A22(B21 - B11)
    m5 = strassen_recursive(add_matrix(a11, a12), b22, leafSize, counter)  # (A11 + A12)B22
    m6 = strassen_recursive(subtract_matrix(a21, a11), add_matrix(b11, b12), leafSize, counter)  # (A21 - A11)(B11 + B12)
    m7 = strassen_recursive(subtract_matrix(a12, a22), add_matrix(b21, b22), leafSize, counter)  # (A12 - A22)(B21 + B22)

    # Build each quadrant of the answer from the seven products
    c11 = add_matrix(subtract_matrix(add_matrix(m1, m4), m5), m7)  # M1 + M4 - M5 + M7
    c12 = add_matrix(m3, m5)  # M3 + M5
    c21 = add_matrix(m2, m4)  # M2 + M4
    c22 = add_matrix(subtract_matrix(add_matrix(m1, m3), m2), m6)  # M1 - M2 + M3 + M6
    return join_matrix(c11, c12, c21, c22)


def next_power_of_two(n):
    power = 1
    while power < n:
        power *= 2
    return power


def pad_matrix(m, size):
    # Puts the matrix in the top left corner of a size x size zero matrix
    padded = [row + [0] * (size - len(row)) for row in m]
    padded.extend([[0] * size for _ in range(size - len(m))])
    return padded


def strassen_multiply(a, b, leafSize=1, counter=None):
    # Pads both matrices with zeros up to a power of two, multiplies them and crops the answer back to n x n
    n = len(a)
    if len(b) != n or any(len(row) != n for row in a + b):
        raise ValueError("Strassen multiplication requires two square matrices of the same size.")

    size = next_power_of_two(n)
    product = strassen_recursive(pad_matrix(a, size), pad_matrix(b, size), leafSize, counter)
    return [row[:n] for row in product[:n]]


def format_matrix(m, labels):
    width = max([len(label) for label in labels] + [len(str(value)) for row in m for value in row])
    lines = [" " * width + " " + " ".join(label.rjust(width) for label in labels)]
    for label, row in zip(labels, m):
        lines.append(label.rjust(width) + " " + " ".join(str(value).rjust(width) for value in row))
    return "\n".join(lines)


def save_matrix(m, labels, path):
    with open(path, "w", newline="", encoding="utf-8") as matrixFile:
        writer = csv.writer(matrixFile)
        writer.writerow([""] + labels)
        for label, row in zip(labels, m):
            writer.writerow([label] + row)
