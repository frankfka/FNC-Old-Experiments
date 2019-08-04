from keras import Sequential, Model
from keras.layers import MaxPooling1D, Conv1D, Dropout, Concatenate, Dense, Flatten, AveragePooling1D, LSTM, \
    BatchNormalization
from keras.utils import to_categorical
from keras_preprocessing.sequence import pad_sequences
import numpy as np

from util.GoogleVectorizer import GoogleVectorizer
from util.FNCData import FNCData
from util.plot import plot_keras_history, plot_confusion_matrix


def get_input_lstm(input_shape, dropout, num_units):
    lstm = Sequential()
    lstm.add(
        LSTM(
            units=num_units,
            input_shape=input_shape,
            dropout=dropout,
            recurrent_dropout=dropout
        )
    )

    return lstm


class TwoToOneLSTM(object):

    def __init__(self, input_shape, dropout=0.5, lstm_num_units=32, dense_num_hidden=512):
        claim_lstm = get_input_lstm(
            input_shape=input_shape,
            dropout=dropout,
            num_units=lstm_num_units
        )
        body_lstm = get_input_lstm(
            input_shape=input_shape,
            dropout=dropout,
            num_units=lstm_num_units
        )

        merged_mlp = Concatenate()([claim_lstm.output, body_lstm.output])
        merged_mlp = BatchNormalization()(merged_mlp)
        merged_mlp = Dense(dense_num_hidden, activation='relu')(merged_mlp)
        merged_mlp = Dense(dense_num_hidden, activation='relu')(merged_mlp)
        merged_mlp = Dense(4, activation='softmax')(merged_mlp)

        complete_model = Model([claim_lstm.input, body_lstm.input], merged_mlp)
        complete_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        complete_model.summary()
        self.model = complete_model

    def train(self, titles, bodies, labels, epochs, seq_len, num_classes=4,
              batch_size=32, val_split=0.2, verbose=1):
        # Do sequence padding
        titles = pad_sequences(titles, maxlen=seq_len, dtype='float32')
        bodies = pad_sequences(bodies, maxlen=seq_len, dtype='float32')
        labels = to_categorical(labels, num_classes=num_classes)
        return self.model.fit(
            [titles, bodies],
            labels,
            batch_size=batch_size,
            epochs=epochs,
            verbose=verbose,
            validation_split=val_split
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
    # Define parameters
    SEQ_LEN = 500
    EMB_DIM = 300
    INPUT_SHAPE = (SEQ_LEN, EMB_DIM)
    DROPOUT = 0.5
    NUM_LSTM_UNITS = 64
    NUM_DENSE_HIDDEN = 512

    NUM_EPOCHS = 20
    BATCH_SIZE = 32
    TRAIN_VAL_SPLIT = 0.2

    # Vectorize Data
    v = GoogleVectorizer(path='../util/GoogleNews-vectors-negative300.bin.gz')
    data = FNCData(stance_f='../data/train_stances.csv',
                   body_f='../data/train_bodies.csv',
                   max_seq_len=SEQ_LEN, vectorizer=v,
                   pkl_to='../data/vectorized_data.pkl')

    # Create model
    model = TwoToOneLSTM(
        input_shape=INPUT_SHAPE,
        dropout=DROPOUT,
        lstm_num_units=NUM_LSTM_UNITS,
        dense_num_hidden=NUM_DENSE_HIDDEN
    )

    # Train the model
    history = model.train(
        titles=data.headlines,
        bodies=data.bodies,
        labels=data.stances,
        epochs=NUM_EPOCHS,
        seq_len=SEQ_LEN,
        batch_size=BATCH_SIZE,
        val_split=TRAIN_VAL_SPLIT
    )

    # Plot training history
    plot_keras_history(history, True)

    # Evaluate model
    num_to_eval = 100
    y_true = data.stances[0:num_to_eval]
    y_pred = model.predict(
        titles=data.headlines[0:num_to_eval],
        bodies=data.bodies[0:num_to_eval],
        seq_len=SEQ_LEN,
        batch_size=BATCH_SIZE
    )
    y_pred = [np.argmax(i) for i in y_pred]

    # Plot confusion matrix
    plot_confusion_matrix(
        y_true=y_true,
        y_pred=y_pred,
        normalize=True,
        classes=['agree', 'disagree', 'discuss', 'unrelated']
    )
