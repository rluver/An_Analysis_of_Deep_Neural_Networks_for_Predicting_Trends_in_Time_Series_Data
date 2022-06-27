from ...config.config import model_config

import tensorflow as tf
from tensorflow.keras.layers import Input, Layer, Dropout, Dense
from tensorflow.keras.models import Model


class MLP(Layer):
    def __init__(self, config=model_config, **kwargs):
        super(MLP, self).__init__(**kwargs)
        self.config = config

        self.dense = Dense(units=self.config.units)
    
    def call(self, inputs, training=None):
        return self.dense(inputs)


def build_mlp(input_shape, config=model_config):
    input = Input(
        shape=input_shape,
        name='feature_fusion_input'
        )

    x = input
    for i in range(config.n_mlp):
        x = MLP(name=f'mlp_layer_{i+1}')(x)

        if i%2 == 0 and i != config.n_mlp-1:
            x = Dropout(config.dropout_rate)(x)

    output = Dense(units=2, name='output')(x)
    
    return Model(input, output)
