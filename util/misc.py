from sklearn.utils import class_weight
import numpy as np


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