from keras import Sequential, Model
from keras.callbacks import TensorBoard
from keras.layers import MaxPooling1D, Conv1D, Dropout, Concatenate, Dense, Flatten, AveragePooling1D
from keras.utils import to_categorical
from keras_preprocessing.sequence import pad_sequences
from keras.optimizers import Adam
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from util.GoogleVectorizer import GoogleVectorizer
from util.FNCData import FNCData
from util.eval import eval_predictions
from util.misc import get_class_weights, log, get_tb_logdir
from util.plot import plot_keras_history, plot_confusion_matrix


def get_1d_pool(pool_size, max_pool=True):
    return MaxPooling1D(pool_size=pool_size) if max_pool else AveragePooling1D(pool_size=pool_size)


def get_input_cnn(input_shape, dropout, conv_num_hidden, conv_kernel_size, max_pool, pool_size):
    cnn = Sequential()

    cnn.add(
        Conv1D(
            filters=conv_num_hidden,
            kernel_size=conv_kernel_size,
            activation='relu',
            input_shape=input_shape
        )
    )
    cnn.add(Dropout(dropout))
    cnn.add(get_1d_pool(pool_size=pool_size, max_pool=max_pool))

    cnn.add(Conv1D(filters=conv_num_hidden, kernel_size=conv_kernel_size, activation='relu'))
    cnn.add(Dropout(dropout))
    cnn.add(get_1d_pool(pool_size=pool_size, max_pool=max_pool))

    cnn.add(Conv1D(filters=conv_num_hidden * 2, kernel_size=conv_kernel_size, activation='relu'))
    cnn.add(Dropout(dropout))
    cnn.add(get_1d_pool(pool_size=pool_size, max_pool=max_pool))

    cnn.add(Conv1D(filters=conv_num_hidden * 2, kernel_size=conv_kernel_size, activation='relu'))
    cnn.add(Dropout(dropout))

    cnn.add(Conv1D(filters=conv_num_hidden * 3, kernel_size=conv_kernel_size, activation='relu'))
    cnn.add(Dropout(dropout))

    return cnn


class CiscoCNN(object):

    def __init__(self, input_shape, dropout=0.5, conv_num_hidden=256, conv_kernel_size=3, max_pool=True,
                 pool_size=2, dense_num_hidden=1024, lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8):
        claim_cnn = get_input_cnn(
            input_shape=input_shape,
            dropout=dropout,
            conv_num_hidden=conv_num_hidden,
            conv_kernel_size=conv_kernel_size,
            max_pool=max_pool,
            pool_size=pool_size
        )
        body_cnn = get_input_cnn(
            input_shape=input_shape,
            dropout=dropout,
            conv_num_hidden=conv_num_hidden,
            conv_kernel_size=conv_kernel_size,
            max_pool=max_pool,
            pool_size=pool_size
        )

        merged_mlp = Concatenate()([claim_cnn.output, body_cnn.output])
        merged_mlp = Flatten()(merged_mlp)
        merged_mlp = Dense(dense_num_hidden, activation='relu')(merged_mlp)
        merged_mlp = Dense(dense_num_hidden, activation='relu')(merged_mlp)
        merged_mlp = Dense(dense_num_hidden, activation='relu')(merged_mlp)
        merged_mlp = Dense(3, activation='softmax')(merged_mlp)

        complete_model = Model([claim_cnn.input, body_cnn.input], merged_mlp)

        # Create the optimizer
        optimizer = Adam(lr=lr, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon)

        complete_model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        complete_model.summary()
        self.model = complete_model

    def train(self, titles, bodies, labels, epochs, seq_len, num_classes=3,
              batch_size=32, val_split=0.2, verbose=1, logs_name=None):
        # Do sequence padding
        titles = pad_sequences(titles, maxlen=seq_len, dtype='float32')
        bodies = pad_sequences(bodies, maxlen=seq_len, dtype='float32')
        labels = to_categorical(labels, num_classes=num_classes)
        class_weights = get_class_weights(labels)
        log("Calculated class weights")
        log(class_weights)
        # Init tensorboard
        callbacks = []
        if logs_name is not None:
            callbacks.append(TensorBoard(log_dir=get_tb_logdir(f"CiscoCNN_{logs_name}")))
        return self.model.fit(
            [titles, bodies],
            labels,
            batch_size=batch_size,
            epochs=epochs,
            verbose=verbose,
            validation_split=val_split,
            class_weight=class_weights,
            callbacks=callbacks
        )

    def predict(self, titles, bodies, seq_len, batch_size=32, verbose=1):
        titles = pad_sequences(titles, maxlen=seq_len, dtype='float32')
        bodies = pad_sequences(bodies, maxlen=seq_len, dtype='float32')
        return self.model.predict(
            [titles, bodies],
            batch_size=batch_size,
            verbose=verbose
        )


if __name__ == '__main__':
    # Model Params
    NUM_CLASSES = 3
    SEQ_LEN = 500
    EMB_DIM = 300
    INPUT_SHAPE = (SEQ_LEN, EMB_DIM)
    DROPOUT = 0.5
    NUM_CONV_HIDDEN = 256
    KERNEL_SIZE_CONV = 3
    USE_MAXPOOL = False
    POOL_SIZE = 2
    NUM_DENSE_HIDDEN = 1024
    # Optimizer
    ADAM_LR = 0.0002
    ADAM_B1 = 0.1
    ADAM_B2 = 0.001
    ADAM_EPSILON = 1e-08

    # Training Params
    NUM_EPOCHS = 30
    BATCH_SIZE = 64
    TRAIN_VAL_SPLIT = 0.2

    # Vectorize Data
    # v = GoogleVectorizer(path='../util/GoogleNews-vectors-negative300.bin.gz')
    data = FNCData(
        # stance_f='../data/train_stances.csv',
        # body_f='../data/train_bodies.csv',
        # max_seq_len=SEQ_LEN, vectorizer=v,
        # pkl_to='../data/vectorized_data_balanced.pkl',
        # bal_stances=True,
        pkl_from='../data/vectorized_data.pkl',
    )

    # Create model
    model = CiscoCNN(
        input_shape=INPUT_SHAPE,
        dropout=DROPOUT,
        conv_num_hidden=NUM_CONV_HIDDEN,
        conv_kernel_size=KERNEL_SIZE_CONV,
        max_pool=USE_MAXPOOL,
        pool_size=POOL_SIZE,
        dense_num_hidden=NUM_DENSE_HIDDEN,
        lr=ADAM_LR,
        beta_1=ADAM_B1,
        beta_2=ADAM_B2,
        epsilon=ADAM_EPSILON
    )

    # Train the model
    history = model.train(
        titles=data.headlines,
        bodies=data.bodies,
        labels=data.stances,
        epochs=NUM_EPOCHS,
        seq_len=SEQ_LEN,
        batch_size=BATCH_SIZE,
        num_classes=NUM_CLASSES,
        val_split=TRAIN_VAL_SPLIT,
        logs_name=f"{NUM_CONV_HIDDEN}CONV-{NUM_DENSE_HIDDEN}DENSE-{DROPOUT}DOUT-{USE_MAXPOOL}MXPL-{NUM_EPOCHS}EPOCHS"
    )

    # Plot training history
    plot_keras_history(history, True)

    # Evaluate model
    test_data = FNCData(
        # max_seq_len=500,
        # vectorizer=v,
        # stance_f='../data/competition_test_stances.csv',
        # body_f='../data/competition_test_bodies.csv',
        # pkl_to='../data/vectorized_data_test.pkl',
        # bal_stances=False,
        pkl_from='../data/vectorized_data_test.pkl'
    )
    y_true = test_data.stances
    y_pred = model.predict(
        titles=test_data.headlines,
        bodies=test_data.bodies,
        seq_len=SEQ_LEN,
        batch_size=BATCH_SIZE
    )
    y_pred = [np.argmax(i) for i in y_pred]

    eval_predictions(y_true=y_true, y_pred=y_pred, print_results=True)
