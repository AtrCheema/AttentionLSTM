comibining SelfAttention mechnism with LSTM

```python
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from atten_lstm import AttentionLSTM
seq_len = 20
num_inputs = 2
inp = Input(shape=(seq_len, num_inputs))
outs = AttentionLSTM(num_inputs, 16)(inp)
outs = Dense(1)(outs)

model = Model(inputs=inp, outputs=outs)
model.compile(loss="mse")

print(model.summary())
# define input
x = np.random.random((100, seq_len, num_inputs))
y = np.random.random((100, 1))
h = model.fit(x=x, y=y)
```

For more comprehensive illustration see examples