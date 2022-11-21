"""
==============
ARG prediction
==============
"""

import matplotlib.pyplot as plt
from easy_mpl import imshow

from sklearn.preprocessing import MinMaxScaler

from ai4water import Model
from ai4water.datasets import busan_beach

from SeqMetrics import RegressionMetrics

data = busan_beach(inputs=[
    'tide_cm', 'wat_temp_c', 'air_temp_c', 'sal_psu',
    'pcp_mm', 'wind_dir_deg', 'wind_speed_mps'
])

print(data.shape)

#%%
# input features
input_features = data.columns.tolist()[0:-1]
print(input_features)

#%%

output_features = data.columns.tolist()[-1:]
print(output_features)

#%%

seq_len = 14
num_inputs = len(input_features)

#%%
# build the model

model = Model(
    model = {"layers": {
        "Input_1": {"shape": (seq_len, num_inputs)},
        "AttentionLSTM": {"num_inputs": num_inputs, "lstm_units": 10},
        "Dense": 1
    }},
    x_transformation='minmax',
    y_transformation="log",
    input_features=input_features,
    output_features = output_features,
    train_fraction=1.0,
    split_random=True,
    ts_args={"lookback": seq_len},
    lr=0.005,
    batch_size=24,
    epochs=50000,
    patience=1000,
    monitor=["nse"],
)

#%%
# train the model

h = model.fit(data=data, verbose=1)

#%%

x_val, y_val = model.validation_data(data=data)

#%%
# check performance

pred_val = model.predict_on_validation_data(data=data, process_results=False)
metrics = RegressionMetrics(y_val, pred_val)
metrics.nse(), metrics.r2()

#%%

attention_weights = model.get_attention_lstm_weights(x_val)

#%%
# plot attention maps

num_examples = 40  # number of examples to show


for idx, key in enumerate(attention_weights.keys()):

    fig, axis = plt.subplots(2, sharex="all", figsize=(6, 8))

    val = attention_weights[key][0:num_examples].T
    val = MinMaxScaler().fit_transform(val)
    imshow(val, colorbar=True, ax=axis[0], show=False, cmap="hot",
           title=f"Attention map for {input_features[idx]}")
    imshow(x_val[:, :, idx][0:num_examples].T, colorbar=True,
        cmap="hot", ax=axis[1], show=False, title=input_features[idx])
    plt.tight_layout()
    plt.show()

#%%
# Training data
#--------------

#%%
x_train, y_train = model.training_data(data=data)

#%%
# check performance

pred_train = model.predict_on_training_data(data=data, process_results=False)
metrics = RegressionMetrics(y_train, pred_train)
metrics.nse(), metrics.r2()

#%%
model.data_config['allow_nan_labels'] = 2
model.data_config['split_random'] = False
x, y = model.all_data(data=data)

attention_weights_tr = model.get_attention_lstm_weights(x)

#%%
# plot attention maps

num_examples = 1400  # number of examples to show

for idx, key in enumerate(attention_weights_tr.keys()):

    fig, axis = plt.subplots(2, sharex="all", figsize=(8, 6),
                             gridspec_kw={"hspace": 0.1})

    val = attention_weights_tr[key][0:num_examples].T
    val = MinMaxScaler().fit_transform(val)
    imshow(val, colorbar=True, ax=axis[0], show=False, cmap="hot",
           title=f"Attention map for {input_features[idx]}",
           aspect="auto")
    imshow(x[:, :, idx][0:num_examples].T, colorbar=True,
        cmap="hot", ax=axis[1], show=False, title=input_features[idx],
           aspect="auto")
    plt.show()


# %%
# plot attention weights without normalization

for idx, key in enumerate(attention_weights_tr.keys()):

    fig, axis = plt.subplots(2, sharex="all", figsize=(8, 6),
                             gridspec_kw={"hspace": 0.1})

    val = attention_weights_tr[key][0:num_examples].T
    imshow(val, colorbar=True, ax=axis[0], show=False, cmap="hot",
           title=f"Attention map for {input_features[idx]}",
           aspect="auto")
    imshow(x[:, :, idx][0:num_examples].T, colorbar=True,
        cmap="hot", ax=axis[1], show=False, title=input_features[idx],
           aspect="auto")
    plt.show()
