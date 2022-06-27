from ...config.config import model_config

import tensorflow as tf
from tensorflow.keras.layers import Input, Layer, Conv1D, MaxPool1D, Dropout, Dense, Flatten
from tensorflow.keras.models import Model

class CNN(Layer):
    def __init__(self, config=model_config, **kwargs):
        super(CNN, self).__init__(**kwargs)
        self.config = config

        self.conv = Conv1D(
            filters=self.config.filters, 
            kernel_size=self.config.kernel_size, 
            activation='relu',
            kernel_initializer='he_normal'
            )
        self.pooling = MaxPool1D()
        self.dropout = Dropout(rate=self.config.dropout_rate)

    def call(self, inputs, training=None):
        x = self.conv(inputs)
        x = self.pooling(x)
        output = self.dropout(x)
        
        return output


def build_cnn(input_shape, config=model_config):
    input = Input(
        shape=input_shape[1:],
        name='local_input'
        )

    x = input
    for i in range(config.n_cnn):
        x = CNN(name=f'cnn_layer_{i+1}')(x)
    x = Flatten()(x)
    output = Dense(units=config.units, name='local_feature_fusion_input')(x)
    
    return Model(input, output)
