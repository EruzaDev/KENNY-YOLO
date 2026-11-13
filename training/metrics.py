"""Small, explicit classification metrics without an extra scientific runtime."""
import numpy as np


def classification_average_precision(actual, scores, labels):
    """One-vs-rest AP, grouping tied scores at each threshold; not detection IoU mAP."""
    per_class = {}
    for label in labels:
        truth = np.asarray([item == label for item in actual], dtype=np.int64)
        positives = int(truth.sum())
        if not positives:
            per_class[label] = None
            continue
        values = np.asarray([row.get(label, 0.) for row in scores], dtype=float)
        order = np.argsort(-values, kind='stable')
        ranked, sorted_scores = truth[order], values[order]
        ends = np.r_[np.flatnonzero(np.diff(sorted_scores)), len(sorted_scores)-1]
        tp = np.cumsum(ranked)[ends]
        recall = tp / positives
        precision = tp / (ends + 1)
        per_class[label] = float(np.sum(np.diff(np.r_[0., recall]) * precision))
    available = [value for value in per_class.values() if value is not None]
    return dict(per_class=per_class, mean_average_precision=float(np.mean(available)) if available else None,
                kind='one-vs-rest classification AP; detection mAP@IoU is not applicable')


def classification_metrics(actual: list[str], predicted: list[str], labels: list[str]):
    if not actual or len(actual) != len(predicted):
        raise ValueError('Expected equally sized, nonempty truth and prediction lists.')
    index = {name: i for i, name in enumerate(labels)}
    matrix = np.zeros((len(labels), len(labels)), dtype=np.int64)
    for truth, guess in zip(actual, predicted):
        matrix[index[truth], index[guess]] += 1
    report = {}
    for i, name in enumerate(labels):
        tp = int(matrix[i, i])
        support = int(matrix[i].sum())
        positives = int(matrix[:, i].sum())
        precision = tp / positives if positives else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        report[name] = {'precision': precision, 'recall': recall,
                        'f1-score': f1, 'support': support}
    report['accuracy'] = float(matrix.trace() / matrix.sum())
    for average in ['macro avg', 'weighted avg']:
        weights = np.ones(len(labels)) if average == 'macro avg' else matrix.sum(axis=1)
        report[average] = {key:float(np.average([report[c][key] for c in labels], weights=weights))
                           for key in ['precision', 'recall', 'f1-score']}
        report[average]['support'] = len(actual)
    return matrix, report


def plot_confusion(matrix, labels, destination):
    import matplotlib.pyplot as plt
    fig, axis = plt.subplots(figsize=(6, 5))
    image = axis.imshow(matrix, cmap='Blues', vmin=0)
    fig.colorbar(image, ax=axis)
    axis.set(xticks=range(len(labels)), yticks=range(len(labels)),
             xticklabels=labels, yticklabels=labels, xlabel='Predicted class', ylabel='True class')
    for i in range(len(labels)):
        for j in range(len(labels)):
            axis.text(j, i, str(matrix[i,j]), ha='center', va='center',
                      color='white' if matrix[i,j] > matrix.max()/2 else 'black')
    fig.tight_layout(); fig.savefig(destination); plt.close(fig)
