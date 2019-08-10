from sklearn.utils import class_weight
from datetime import datetime

import numpy as np


# Get a log directory for tensorboard
def get_tb_logdir(unique_name):
    timestamp = datetime.now().strftime("%m%d-%H%M")
    return f"../logs/{unique_name}-{timestamp}"


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
