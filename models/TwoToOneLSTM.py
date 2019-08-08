from keras import Sequential, Model
from keras.callbacks import TensorBoard
from keras.layers import Concatenate, Dense, LSTM, BatchNormalization, Bidirectional
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras_preprocessing.sequence import pad_sequences
import numpy as np

from util.GoogleVectorizer import GoogleVectorizer
from util.FNCData import FNCData
from util.misc import get_class_weights, get_tb_logdir, eval_predictions, log
from util.plot import plot_keras_history, plot_confusion_matrix


def get_input_lstm(input_shape, dropout, num_units, bi_directional):
    input_lstm = Sequential()

    # The sequence processing LSTM layer
    if bi_directional:
        lstm = Bidirectional(
            LSTM(units=num_units, dropout=dropout, recurrent_dropout=dropout),
            input_shape=input_shape
        )  # TODO: Play with merge methods
    else:
        lstm = LSTM(
            units=num_units,
            input_shape=input_shape,
            dropout=dropout,
            recurrent_dropout=dropout
        )
    input_lstm.add(lstm)

    return input_lstm


class TwoToOneLSTM(object):

    def __init__(self, input_shape, dropout=0.5, lstm_num_units=32, lstm_bidirectional=False, dense_num_hidden=512,
                 lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8):
        claim_lstm = get_input_lstm(
            input_shape=input_shape,
            dropout=dropout,
            num_units=lstm_num_units,
            bi_directional=lstm_bidirectional
        )
        body_lstm = get_input_lstm(
            input_shape=input_shape,
            dropout=dropout,
            num_units=lstm_num_units,
            bi_directional=lstm_bidirectional
        )

        merged_mlp = Concatenate()([claim_lstm.output, body_lstm.output])
        merged_mlp = BatchNormalization()(merged_mlp)
        merged_mlp = Dense(dense_num_hidden, activation='relu')(merged_mlp)
        merged_mlp = Dense(dense_num_hidden, activation='relu')(merged_mlp)
        merged_mlp = Dense(3, activation='softmax')(merged_mlp)

        complete_model = Model([claim_lstm.input, body_lstm.input], merged_mlp)

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
            callbacks.append(TensorBoard(log_dir=get_tb_logdir(f"Two2OneLSTM_{logs_name}")))
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
    SEQ_LEN = 500
    EMB_DIM = 300
    INPUT_SHAPE = (SEQ_LEN, EMB_DIM)
    DROPOUT = 0.5
    NUM_LSTM_UNITS = 64
    LSTM_BIDIRECTIONAL = True
    NUM_DENSE_HIDDEN = 512
    NUM_CLASSES = 3
    # Optimizer
    ADAM_LR = 0.001
    ADAM_B1 = 0.9
    ADAM_B2 = 0.999
    ADAM_EPSILON = 1e-08

    # Training Params
    NUM_EPOCHS = 30
    BATCH_SIZE = 64
    TRAIN_VAL_SPLIT = 0.2

    # Vectorize Data
    # v = GoogleVectorizer(path='../util/GoogleNews-vectors-negative300.bin.gz')
    data = FNCData(
        # max_seq_len=SEQ_LEN,
        # vectorizer=v,
        # stance_f='../data/train_stances.csv',
        # body_f='../data/train_bodies.csv',
        # pkl_to='../data/vectorized_data.pkl'
        pkl_from='../data/vectorized_data.pkl'
    )

    # Create model
    model = TwoToOneLSTM(
        input_shape=INPUT_SHAPE,
        dropout=DROPOUT,
        lstm_num_units=NUM_LSTM_UNITS,
        lstm_bidirectional=LSTM_BIDIRECTIONAL,
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
        val_split=TRAIN_VAL_SPLIT,
        num_classes=NUM_CLASSES,
        logs_name=f"{Bidirectional}BIDIR){NUM_LSTM_UNITS}LSTM-{NUM_DENSE_HIDDEN}DENSE-{DROPOUT}DOUT-{NUM_EPOCHS}EPOCHS"
    )

    # Plot training history
    plot_keras_history(history, True)

    # Evaluate model
    test_data = FNCData(
        # max_seq_len=500,
        # vectorizer=v,
        # stance_f='../data/competition_test_stances.csv',
        # body_f='../data/competition_test_bodies.csv',
        # pkl_to='../data/vectorized_data_test.pkl'
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
