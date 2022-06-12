
import unittest

import numpy as np
from tensorflow.keras.models import Model
from atten_lstm import AttentionLSTM, SelfAttention
from tensorflow.keras.layers import Input, LSTM, Dense

class TestSelfAttention(unittest.TestCase):

    def test_selfatten(self):
        inp = Input(shape=(10, 1))
        lstm = LSTM(2, return_sequences=True)(inp)
        sa, _ = SelfAttention()(lstm)
        out = Dense(1)(sa)

        model = Model(inputs=inp, outputs=out)
        model.compile(loss="mse")

        x = np.random.random((100, 10, 1))
        y = np.random.random((100, 1))
        h = model.fit(x=x, y=y)
        return


class TestAttentionLSTM(unittest.TestCase):

    def test_attenlstm(self):
        seq_len = 20
        num_inputs = 2
        inp = Input(shape=(seq_len, num_inputs))
        outs = AttentionLSTM(num_inputs, 16)(inp)
        outs = Dense(1)(outs)

        model = Model(inputs=inp, outputs=outs)
        model.compile(loss="mse")

        # define input
        x = np.random.random((100, seq_len, num_inputs))
        y = np.random.random((100, 1))
        h = model.fit(x=x, y=y)
        return


if __name__ == "__main__":
    unittest.main()