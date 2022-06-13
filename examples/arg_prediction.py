"""
==============
ARG prediction
==============
"""

import pandas as pd
import matplotlib.pyplot as plt
from easy_mpl import imshow, plot

from ai4water import Model
from ai4water.datasets import busan_beach

data = busan_beach(inputs=['tide_cm', 'wat_temp_c', 'air_temp_c', 'sal_psu', 'pcp_mm', 'wind_dir_deg'])

data.shape
#%%

input_features = data.columns.tolist()[0:-1]
input_features
#%%

output_features = data.columns.tolist()[-1:]
output_features
#%%

seq_len = 14
num_inputs = len(input_features)

#%%
# build the model

model = Model(
    model = {"layers": {
        "Input_1": {"shape": (seq_len, num_inputs)},
        "AttentionLSTM": {"num_inputs": num_inputs, "lstm_units": 16},
        "Dense": 1
    }},
    x_transformation='minmax',
    input_features=input_features,
    output_features = output_features,
    train_fraction=1.0,
    ts_args={"lookback": seq_len},
    epochs=40000,
    patience=5000
)

#%%
# train the model

h = model.fit(data=data, verbose=0)

#%%

x_test, y_test = model.validation_data(data=data)

#%%

attention_weights = model.get_attention_lstm_weights(x_test)

#%%
# plot attention maps

num_examples = 40  # number of examples to show

fig, axis = plt.subplots(len(attention_weights), sharex="all", figsize=(6, 8))

idx = 0
for key, ax in zip(attention_weights.keys(), axis):

    imshow(attention_weights[key][0:num_examples].T,
           ylabel="Sequence len",
           title=input_features[idx],
           cmap="hot",
           ax=ax,
           show=False,
           vmin=0,
           vmax=1.0,
            aspect="auto",
           )
    idx += 1

plt.xlabel("Examples")
plt.tight_layout()
plt.savefig("attention_maps")
plt.show()

#%%
# plot the input data

x_test_df = pd.DataFrame(x_test[:, -1, :][0:num_examples], columns=input_features)
x_test_df.plot(subplots=True, use_index=False)
plt.xlabel("Examples")
plt.savefig("input data")
plt.show()

#%%
# plot the actual target data

plot(y_test[0:num_examples], xlabel="Examples",
     ylabel="ARG coppml",
     title=output_features[0], show=False)
plt.savefig("target_data")
plt.show()

