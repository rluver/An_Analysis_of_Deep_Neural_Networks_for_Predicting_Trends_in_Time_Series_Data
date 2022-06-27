from ...config.config import model_config

import tensorflow as tf
from tensorflow.keras.layers import Input, Layer, Dropout, Dense
from tensorflow.keras.models import Model


class LSTM(Layer):
    def __init__(self, config=model_config, **kwargs):
        from tensorflow.keras.layers import LSTM
        super().__init__(self, **kwargs)
        self.config = config

        self.lstm = LSTM(
            units=self.config.lstm_units, 
            return_sequences=True, 
            activation='relu'
            )
        self.dropout = Dropout(rate=self.config.dropout_rate)
    
    def call(self, inputs, training=None):
        x = self.lstm(inputs)
        output = self.dropout(x)

        return output


def build_lstm(input_shape, config=model_config):
    input = Input(
        shape=input_shape[1:], 
        name='historical_trend_input'
        )

    x = input
    for i in range(config.n_lstm):
        x = LSTM(name=f'lstm_layer_{i+1}')(x)
    output = Dense(units=config.units, name='historical_trend_fusion_input')(x[:, -1, :])
    
    return Model(input, output)
