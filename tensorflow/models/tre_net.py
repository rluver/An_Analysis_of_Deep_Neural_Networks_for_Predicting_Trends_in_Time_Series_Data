from ...config.config import model_config
from ..layers.cnn import build_cnn
from ..layers.lstm import build_lstm
from ..layers.mlp import build_mlp

import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model


class TreNet(Model):
    def __init__(self, config=model_config, **kwargs):
        super(TreNet, self).__init__()
        self.config = config

        self.mlp = build_mlp((config.units,), config)

    def build(self, input_shape):
        self.cnn = build_cnn(input_shape, self.config)
        self.lstm = build_lstm(input_shape, self.config)
        
    def call(self, inputs):
        cnn_output = self.cnn(inputs)
        lstm_output = self.lstm(inputs)

        feature_fusion_input = cnn_output + lstm_output
        output = self.mlp(feature_fusion_input)

        return output
