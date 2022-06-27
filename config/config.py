class model_config:
    batch_size = 2000
    epochs = 100

    dropout_rate = 0.0
    l2 = 5e-4
    lr = 1e-3

    n_mlp = 5   # fully connected neural network layers, 1~5
    units=1200   # 300, 600, 900, 1200

    n_lstm = 3  # lstm layers, 1~3
    lstm_units = 1500

    n_cnn = 3   # cnn layers, 1~3
    filters =  4
    kernel_size = 4