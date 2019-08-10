from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from util import FNCData
from util.misc import log
from util.plot import plot_confusion_matrix
from sklearn.model_selection import train_test_split, StratifiedKFold


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


# K-Fold Validation - returns [train_idx], [test_idx]
# Each train/test_idx is a list of indicies from the FNCData object
# Ex. train_data = fnc_data.headlines[train_idx]
def k_fold_indicies(fnc_data, k=10):
    assert(isinstance(fnc_data, FNCData.FNCData))
    skf = StratifiedKFold(n_splits=k, shuffle=True)
    return skf.split(fnc_data.headlines, fnc_data.stances)
