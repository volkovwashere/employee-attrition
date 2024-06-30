from numpy.typing import NDArray
from sklearn.metrics import classification_report


def create_classification_report(
    target: NDArray,
    preds: NDArray,
    labels: list[str],
    stage: str,
) -> dict[str, float]:
    """
    This function creates and formats the classification report. It uses the sklearn clf report function
    to calculate the results and then a custom formatter function is applied on the output. In the end the
    result dict will be compatible with mlflow and can be exported.
    Args:
        target (ndarray): 1D np array that contains the targets 0, 1.
        preds (ndarray): 1D np array that contains the predicted outputs.
        labels (Listr[str]): List of the original label names.
        stage (str): ML lifecycle stage, can be train, validation, test.

    Returns:
        Dictionary with formatted results Dict[str, float]
    """
    results = classification_report(
        target,
        preds,
        target_names=labels,
        output_dict=True,
    )
    # format clf report
    results_per_label = tuple(results[label] for label in labels)
    updated_results_per_label = []
    for result, label in zip(results_per_label, labels):
        updated_results_per_label.append({f"{stage}/class_{label}/{k}": round(v, 4) for k, v in result.items()})

    formatted_results = updated_results_per_label[0] | updated_results_per_label[1] | {f"{stage}/accuracy": results["accuracy"]}
    return formatted_results
