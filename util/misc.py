from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn.utils import class_weight
from datetime import datetime
from util.plot import plot_confusion_matrix
import numpy as np


# Get a log directory for tensorboard
def get_tb_logdir(unique_name):
    timestamp = datetime.now().strftime("%m%d-%H%M")
    return f"logs/{unique_name}-{timestamp}"


# Log a message to console
def log(msg, level="INFO", header=False):
    if header:
        print(
            f"""
            ========================================
            {msg}
            ========================================
            """
        )
    else:
        print(f"{level}: {msg}")


# Computes class weights for keras with to_categorical applied to y-data
def get_class_weights(y_train):
    y_ints = categorical_to_idx(y_train)
    return class_weight.compute_class_weight(
        'balanced',
        np.unique(y_ints),
        y_ints
    )


# Returns Keras 2D categorical arrays (ex. [[0 0 1 0]] to an integer array: [3])
def categorical_to_idx(cat_labels):
    return [np.argmax(y) for y in cat_labels]


# Returns useful metrics in a dict, logs if asked
# Accuracy, F1 Micro, F1 Macro, F1 Weighted, Confusion matrix
def eval_predictions(y_true, y_pred, print_results=False):
    # Plot confusion matrix
    plot_confusion_matrix(
        y_true=y_true,
        y_pred=y_pred,
        normalize=True,
        classes=['agree', 'disagree', 'discuss']
    )
    # TODO: Precision, recall, return results in a dict
    accuracy = accuracy_score(y_true=y_true, y_pred=y_pred)
    f1_score_micro = f1_score(y_true=y_true, y_pred=y_pred, average='micro')
    f1_score_macro = f1_score(y_true=y_true, y_pred=y_pred, average='macro')
    f1_score_weighted = f1_score(y_true=y_true, y_pred=y_pred, average='weighted')
    if print_results:
        log("Prediction Evaluation", header=True)
        log(f"Accuracy: {accuracy}")
        log(f"F1 Score (Macro): {f1_score_macro}")
        log(f"F1 Score (Micro): {f1_score_micro}")
        log(f"F1 Score (Weighted): {f1_score_weighted}")
