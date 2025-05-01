class config:
    INPUT_DIM = 784
    hidden_dim = 128 # we use same size for both hidden layers
    output_dim = 10
    LEARNING_RATE = 0.001
    decay_rate = 0.95
    beta = 0.001
    batch_size = 256
    EPOCHS = 120
    activation = 'sigmoid'