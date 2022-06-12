
__all__ = ["SelfAttention", "AttentionLSTM"]

import tensorflow as tf

initializers = tf.keras.initializers
regularizers = tf.keras.regularizers
constraints = tf.keras.constraints
Layer = tf.keras.layers.Layer
layers = tf.keras.layers
K = tf.keras.backend
Dense = tf.keras.layers.Dense
Lambda = tf.keras.layers.Lambda
Activation = tf.keras.layers.Activation
Softmax = tf.keras.layers.Softmax
dot = tf.keras.layers.dot
concatenate = tf.keras.layers.concatenate


class SelfAttention(Layer):
    """
    SelfAttention is originally proposed by Cheng et al., 2016 [1]_
    Here using the implementation of Philipperemy from
    [2]_ with modification
    that `attn_units` and `attn_activation` attributes can be changed.
    The default values of these attributes are same as used by the auther.
    However, there is another implementation of SelfAttention at [3]_
    but the author have cited a different paper i.e. Zheng et al., 2018 [4]_ and
    named it as additive attention.
    A useful discussion about this (in this class) implementation can be found at [5]_

    Examples
    --------
    >>> from atten_lstm import SelfAttention
    >>> from tensorflow.keras.layers import Input, LSTM, Dense
    >>> from tensorflow.keras.models import Model
    >>> import numpy as np
    >>> inp = Input(shape=(10, 1))
    >>> lstm = LSTM(2, return_sequences=True)(inp)
    >>> sa, _ = SelfAttention()(lstm)
    >>> out = Dense(1)(sa)
    ...
    >>> model = Model(inputs=inp, outputs=out)
    >>> model.compile(loss="mse")
    ...
    >>> print(model.summary())
    ...
    >>> x = np.random.random((100, 10, 1))
    >>> y = np.random.random((100, 1))
    >>> h = model.fit(x=x, y=y)

    References
    ----------
    .. [1] https://arxiv.org/pdf/1601.06733.pdf
    .. [2] https://github.com/philipperemy/keras-attention-mechanism/blob/master/attention/attention.py
    .. [3] https://github.com/CyberZHG/keras-self-attention/blob/master/keras_self_attention/seq_self_attention.py
    .. [4] https://arxiv.org/pdf/1806.01264.pdf
    .. [5] https://github.com/philipperemy/keras-attention-mechanism/issues/14
    """
    def __init__(
            self,
            units:int = 128,
            activation:str = 'tanh',
            return_attention_weights:bool = True,
            **kwargs
    ):
        """
        Parameters
        ----------
            units : int, optional (default=128)
                number of units for attention mechanism
            activation : str, optional (default="tanh")
                activation function to use in attention mechanism
            return_attention_weights : bool, optional (default=True)
                if True, then it returns two outputs, first is attention vector
                of shape (batch_size, units) and second is of shape (batch_size, time_steps)
                If False, then returns only attention vector.
            **kwargs :
                any additional keyword arguments for keras Layer.
        """
        self.units = units
        self.attn_activation = activation
        self.return_attention_weights = return_attention_weights
        super().__init__(**kwargs)

    def build(self, input_shape):
        hidden_size = int(input_shape[-1])
        self.d1 = Dense(hidden_size, use_bias=False)
        self.act = Activation('softmax')
        self.d2 = Dense(self.units, use_bias=False, activation=self.attn_activation)
        return

    def call(self, hidden_states, *args, **kwargs):
        """
        Many-to-one attention mechanism for Keras.
        @param hidden_states: 3D tensor with shape (batch_size, time_steps, input_dim).
        @return: 2D tensor with shape (batch_size, 128)
        @author: felixhao28.
        The original code which has here been modified had Apache Licence 2.0.
        """
        hidden_size = int(hidden_states.shape[2])
        # Inside dense layer
        #              hidden_states            dot               W            =>           score_first_part
        # (batch_size, time_steps, hidden_size) dot (hidden_size, hidden_size) => (batch_size, time_steps, hidden_size)
        # W is the trainable weight matrix of attention Luong's multiplicative style score
        score_first_part = self.d1(hidden_states)
        #            score_first_part           dot        last_hidden_state     => attention_weights
        # (batch_size, time_steps, hidden_size) dot   (batch_size, hidden_size)  => (batch_size, time_steps)
        h_t = Lambda(lambda x: x[:, -1, :], output_shape=(hidden_size,))(hidden_states)
        score = dot([score_first_part, h_t], [2, 1])
        attention_weights =self.act(score)
        # (batch_size, time_steps, hidden_size) dot (batch_size, time_steps) => (batch_size, hidden_size)
        context_vector = dot([hidden_states, attention_weights], [1, 1])
        pre_activation = concatenate([context_vector, h_t])
        attention_vector = self.d2(pre_activation)

        if self.return_attention_weights:
            return attention_vector, attention_weights
        return attention_vector


class AttentionLSTM(Layer):
    """
    This layer combines Self Attention [7]_ mechanism with LSTM [8]_. It uses one
    separate LSTM+SelfAttention block for each input feature. The output from each
    LSTM+SelfAttention block is concatenated and returned. The layer expects
    same input dimension as by LSTM i.e. (batch_size, time_steps, input_features).
    For usage see example [9]_.

    References
    ----------
    .. [7] https://ai4water.readthedocs.io/en/dev/models/layers.html#selfattention

    .. [8] https://www.tensorflow.org/api_docs/python/tf/keras/layers/LSTM

    .. [9] https://ai4water.readthedocs.io/en/dev/auto_examples/attention_lstm.html#
    """
    def __init__(
            self,
            num_inputs: int,
            lstm_units: int,
            attn_units: int = 128,
            attn_activation: str = "tanh",
            lstm_kwargs:dict = None,
            **kwargs
    ):
        """
        Parameters
        ----------
            num_inputs: int
                number of inputs
            lstm_units : int
                number of units in LSTM layers
            attn_units : int, optional (default=128)
                number of units in SelfAttention layers
            attn_activation : str, optional (default="tanh")
                activation function in SelfAttention layers
            lstm_kwargs : dict, optional (default=None)
                any keyword arguments for LSTM layer.

        Example
        -------
        >>> import numpy as np
        >>> from tensorflow.keras.models import Model
        >>> from tensorflow.keras.layers import Input, Dense
        >>> from atten_lstm import AttentionLSTM
        >>> seq_len = 20
        >>> num_inputs = 2
        >>> inp = Input(shape=(seq_len, num_inputs))
        >>> outs = AttentionLSTM(num_inputs, 16)(inp)
        >>> outs = Dense(1)(outs)
        ...
        >>> model = Model(inputs=inp, outputs=outs)
        >>> model.compile(loss="mse")
        ...
        >>> print(model.summary())
        ... # define input
        >>> x = np.random.random((100, seq_len, num_inputs))
        >>> y = np.random.random((100, 1))
        >>> h = model.fit(x=x, y=y)

        """
        super(AttentionLSTM, self).__init__(**kwargs)
        self.num_inputs = num_inputs
        self.lstm_units = lstm_units
        self.attn_units = attn_units
        self.attn_activation = attn_activation

        if lstm_kwargs is None:
            lstm_kwargs = {}

        assert isinstance(lstm_kwargs, dict)
        self.lstm_kwargs = lstm_kwargs

        self.lstms = []
        self.sas = []
        for i in range(self.num_inputs):
            self.lstms.append(tf.keras.layers.LSTM(self.lstm_units, return_sequences=True, **self.lstm_kwargs))
            self.sas.append(SelfAttention(self.attn_units, self.attn_activation))

    def __call__(self, inputs, *args, **kwargs):

        assert self.num_inputs == inputs.shape[-1], f"""
        num_inputs {self.num_inputs} does not match with input features.
        Inputs are of shape {inputs.shape}"""

        outs = []
        for i in range(inputs.shape[-1]):
            lstm = self.lstms[i](tf.expand_dims(inputs[..., i], axis=-1))
            out, _ = self.sas[i](lstm)
            outs.append(out)

        return tf.concat(outs, axis=-1)
