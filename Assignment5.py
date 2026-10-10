import random
import math
import matplotlib.pyplot as plt

CLASSES = [1, 2, 3]

def mean(values):
    return sum(values) / len(values)

def invert_matrix(matrix):
    n = len(matrix)
    A = [[float(x) for x in row] for row in matrix]
    I = [[1.0 if i == j else 0.0 for j in range(n)]
         for i in range(n)]

    for i in range(n):
        pivot_row = max(range(i, n),
                        key=lambda r: abs(A[r][i]))

        if abs(A[pivot_row][i]) < 1e-12:
            raise ValueError("Singular covariance matrix")

        A[i], A[pivot_row] = A[pivot_row], A[i]
        I[i], I[pivot_row] = I[pivot_row], I[i]

        pivot = A[i][i]

        for j in range(n):
            A[i][j] /= pivot
            I[i][j] /= pivot

        for r in range(n):
            if r != i:
                factor = A[r][i]

                for j in range(n):
                    A[r][j] -= factor * A[i][j]
                    I[r][j] -= factor * I[i][j]

    return I


def determinant(matrix):
    return (
        matrix[0][0] * matrix[1][1]
        - matrix[0][1] * matrix[1][0]
    )

def covariance_matrix(points, mu):
    n = len(points)

    c11 = c12 = c22 = 0.0

    for x, y in points:
        dx = x - mu[0]
        dy = y - mu[1]

        c11 += dx * dx
        c12 += dx * dy
        c22 += dy * dy

    return [
        [c11 / n, c12 / n],
        [c12 / n, c22 / n]
    ]


def estimate_parameters(points):
    mu = [
        mean([p[0] for p in points]),
        mean([p[1] for p in points])
    ]

    cov = covariance_matrix(points, mu)

    cov[0][0] += 1e-6
    cov[1][1] += 1e-6

    det = determinant(cov)

    if det <= 0:
        raise ValueError("Covariance determinant must be positive")

    inv = invert_matrix(cov)

    return {
        "mean": mu,
        "cov": cov,
        "inverse": inv,
        "det": det
    }


def fit_models(points, labels):
    models = {}

    for c in CLASSES:
        class_points = [
            p for p, label in zip(points, labels)
            if label == c
        ]

        if not class_points:
            raise ValueError("No training points for class " + str(c))

        models[c] = estimate_parameters(class_points)

        print("\nClass", c)
        print("Number of points:", len(class_points))
        print("Mean:", models[c]["mean"])
        print("Covariance:")

        for row in models[c]["cov"]:
            print(row)

    return models

def discriminant(point, model):
    dx = point[0] - model["mean"][0]
    dy = point[1] - model["mean"][1]

    inv = model["inverse"]

    distance = dx * (inv[0][0] * dx + inv[0][1] * dy)
    distance += dy * (inv[1][0] * dx + inv[1][1] * dy)
    return -0.5 * distance - 0.5 * math.log(model["det"])


def classify_points(points, models):
    predictions = []
    all_scores = []

    for point in points:
        scores = [
            discriminant(point, models[c])
            for c in CLASSES
        ]

        predictions.append(CLASSES[scores.index(max(scores))])
        all_scores.append(scores)

    return predictions, all_scores

def confusion_matrix(actual, predicted):
    matrix = [[0 for _ in CLASSES] for _ in CLASSES]

    for a, p in zip(actual, predicted):
        matrix[CLASSES.index(a)][CLASSES.index(p)] += 1
    return matrix

def print_confusion_matrix(actual, predicted):
    matrix = confusion_matrix(actual, predicted)
    print("\nConfusion Matrix")
    print("Rows = Actual, Columns = Predicted")
    print("          Pred 1  Pred 2  Pred 3")

    for i, row in enumerate(matrix):
        print("Actual {}: {}".format(CLASSES[i], row))

    correct = sum(matrix[i][i] for i in range(3))
    total = sum(sum(row) for row in matrix)

    print("Accuracy: {:.2f}%".format(100 * correct / total))

def roc_points(binary_actual, scores):
    positives = sum(binary_actual)
    negatives = len(binary_actual) - positives

    if positives == 0 or negatives == 0:
        return [0, 1], [0, 1]
    thresholds = [max(scores) + 1]
    thresholds += sorted(set(scores), reverse=True)
    thresholds += [min(scores) - 1]

    fprs = []
    tprs = []

    for threshold in thresholds:
        tp = fp = 0

        for actual, score in zip(binary_actual, scores):
            predicted_positive = score >= threshold

            if predicted_positive and actual == 1:
                tp += 1
            elif predicted_positive and actual == 0:
                fp += 1

        tprs.append(tp / positives)
        fprs.append(fp / negatives)

    return fprs, tprs


def plot_roc_curves(actual, scores, title):
    plt.figure(figsize=(8, 6))

    for index, c in enumerate(CLASSES):
        binary_actual = [
            1 if label == c else 0 for label in actual
        ]

        class_scores = [row[index] for row in scores]

        fpr, tpr = roc_points(binary_actual, class_scores)

        plt.step(
            fpr, tpr, where="post",
            label="Class {} vs Rest".format(c)
        )

    plt.plot(
        [0, 1], [0, 1], "k--",
        label="Random classifier"
    )

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.xlim(0, 1)
    plt.ylim(0, 1.02)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_decision_boundary(points, labels, models, title):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]

    margin_x = max(1, (max(xs) - min(xs)) * 0.08)
    margin_y = max(1, (max(ys) - min(ys)) * 0.08)

    min_x, max_x = min(xs) - margin_x, max(xs) + margin_x
    min_y, max_y = min(ys) - margin_y, max(ys) + margin_y
    steps = 400

    x_values = [
        min_x + (max_x - min_x) * i / (steps - 1)
        for i in range(steps)
    ]

    y_values = [
        min_y + (max_y - min_y) * i / (steps - 1)
        for i in range(steps)
    ]

    regions = []

    for y in y_values:
        row = []

        for x in x_values:
            scores = [
                discriminant((x, y), models[c])
                for c in CLASSES
            ]

            row.append(CLASSES[scores.index(max(scores))])

        regions.append(row)

    plt.figure(figsize=(9, 7))

    plt.contourf(
        x_values, y_values, regions,
        levels=[0.5, 1.5, 2.5, 3.5],
        alpha=0.18
    )

    plt.contour(
        x_values, y_values, regions,
        levels=[1.5, 2.5],
        colors="black",
        linewidths=1.2
    )

    for c in CLASSES:
        class_points = [
            p for p, label in zip(points, labels)
            if label == c
        ]

        plt.scatter(
            [p[0] for p in class_points],
            [p[1] for p in class_points],
            s=15, alpha=0.6,
            label="Class {}".format(c)
        )

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def run_evaluation(train_x, train_y, test_x, test_y, title):
    print(title)

    models = fit_models(train_x, train_y)

    predictions, scores = classify_points(test_x, models)

    print_confusion_matrix(test_y, predictions)

    plot_decision_boundary(
        test_x, test_y, models,
        title + " - Decision Boundaries"
    )

    plot_roc_curves(
        test_y, scores,
        title + " - ROC Curves"
    )

def read_dataset(filename):
    points = []
    labels = []

    with open(filename, "r") as file:
        for line in file:
            values = line.strip().replace(",", " ").split()

            if not values:
                continue
            if values[0].lower() in ("x", "feature1"):
                continue

            if len(values) != 3:
                raise ValueError(
                    "Expected x, y, class label in " + filename
                )

            x = float(values[0])
            y = float(values[1])
            label = int(float(values[2]))

            if label not in CLASSES:
                raise ValueError("Unexpected class label: " + str(label))

            points.append((x, y))
            labels.append(label)

    if not points:
        raise ValueError("No data found in " + filename)

    return points, labels

def split_80_20(points, labels, seed=42):
    random.seed(seed)

    train_x, train_y = [], []
    test_x, test_y = [], []

    for c in CLASSES:
        class_points = [
            p for p, label in zip(points, labels)
            if label == c
        ]

        random.shuffle(class_points)

        cut = int(0.8 * len(class_points))

        train_x.extend(class_points[:cut])
        train_y.extend([c] * cut)

        test_x.extend(class_points[cut:])
        test_y.extend([c] * (len(class_points) - cut))

    return train_x, train_y, test_x, test_y

def generate_gaussian_points(mu, cov, size):
    dataset = []
    L11 = math.sqrt(cov[0][0])
    L21 = cov[0][1] / L11
    L22 = math.sqrt(cov[1][1] - L21**2)

    while len(dataset) < size:

        u1 = 2 * random.random() - 1
        u2 = 2 * random.random() - 1
        s = u1 * u1 + u2 * u2
        
        if s <= 0 or s >= 1:
            continue
            
        k = math.sqrt((-2 * math.log(s)) / s)
        z1 = u1 * k
        z2 = u2 * k
        x = mu[0] + L11 * z1
        y = mu[1] + L21 * z1 + L22 * z2
        
        dataset.append((x, y))
        
    return dataset


def run_question1():
    experiments = {
        "Isotropic shared covariance": {
            1: [[1.5, 0], [0, 1.5]],
            2: [[1.5, 0], [0, 1.5]],
            3: [[1.5, 0], [0, 1.5]]
        },

        "Isotropic different covariance": {
            1: [[2.5, 0], [0, 2.5]],
            2: [[0.8, 0], [0, 0.8]],
            3: [[1.5, 0], [0, 1.5]]
        },

        "Diagonal shared covariance": {
            1: [[2.5, 0], [0, 0.8]],
            2: [[2.5, 0], [0, 0.8]],
            3: [[2.5, 0], [0, 0.8]]
        },

        "Diagonal different covariance": {
            1: [[2.5, 0], [0, 0.8]],
            2: [[0.8, 0], [0, 2.5]],
            3: [[1.5, 0], [0, 1.5]]
        },

        "Full shared covariance": {
            1: [[2.0, 0.5], [0.5, 1.0]],
            2: [[2.0, 0.5], [0.5, 1.0]],
            3: [[2.0, 0.5], [0.5, 1.0]]
        },

        "Full different covariance": {
            1: [[2.0, 0.7], [0.7, 1.0]],
            2: [[1.0, -0.4], [-0.4, 2.0]],
            3: [[1.5, 0.5], [0.5, 1.5]]
        }
    }

    means = {
        1: [0, 0],
        2: [5, 5],
        3: [0, 7]
    }

    for name, covariances in experiments.items():
        points, labels = [], []

        for c in CLASSES:
            
            class_points = generate_gaussian_points(means[c], covariances[c], size=10000)

            points.extend(class_points)
            labels.extend([c] * len(class_points))

        train_x, train_y, test_x, test_y = split_80_20(
            points, labels
        )

        run_evaluation(
            train_x, train_y, test_x, test_y,
            "Question 1 - " + name
        )

def run_question2():        
        train_x, train_y = read_dataset("train.txt")
        test_x, test_y = read_dataset("dev.txt")

        print("\nTraining points:", len(train_x))
        print("Testing points:", len(test_x))

        run_evaluation(
            train_x, train_y, test_x, test_y,
            "Question 2 - Real Dataset"
        )


if __name__ == "__main__":
        run_question1()
        run_question2()