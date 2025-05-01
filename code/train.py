from MLP_model import *
from config import config

def optimize_model_parameters(
    network, 
    training_data, 
    training_labels, 
    validation_data, 
    validation_labels, 
    lr_initial, 
    lr_decay, 
    regularisation_strength, 
    dropout_rate,   # 新增 dropout_rate 参数
    total_epochs, 
    samples_per_batch, 
    activation_fn=config.activation
):
    train_size = training_data.shape[0]
    number_of_batches = train_size // samples_per_batch

    history_loss_train = []
    history_accuracy_train = []
    history_loss_val = []
    history_accuracy_val = []
    highest_val_accuracy = 0.0
    optimal_model = network

    for epoch in range(total_epochs):
        # Shuffle the training data at the start of each epoch
        shuffled_indices = np.random.permutation(train_size)
        training_data_shuffled = training_data[shuffled_indices]
        training_labels_shuffled = training_labels[shuffled_indices]

        for batch_index in range(number_of_batches):
            start_index = batch_index * samples_per_batch
            end_index = start_index + samples_per_batch
            batch_data = training_data_shuffled[start_index:end_index]
            batch_labels = training_labels_shuffled[start_index:end_index]

            # Forward propagation with Batch Normalization and Dropout (training mode)
            predictions, fwd_cache, bn_cache = model_forward_with_bn(
                network, 
                batch_data, 
                activation=activation_fn, 
                dropout_rate=dropout_rate, 
                training=True
            )
            
            # Backward propagation with Batch Normalization
            gradients = model_backward_with_bn(
                network, 
                fwd_cache, 
                bn_cache, 
                batch_data, 
                batch_labels, 
                predictions, 
                regularisation_strength, 
                activation=activation_fn
            )
            
            # Update parameters
            network = update_parameters(network, gradients, lr_initial)

        # Evaluate metrics on the training data
        predictions_train, _, _ = model_forward_with_bn(
            network, 
            training_data, 
            activation=activation_fn, 
            dropout_rate=0.0,  # Disable Dropout during evaluation
            training=False
        )
        loss_train = calculate_loss(training_labels, predictions_train, network, regularisation_strength)
        accuracy_train = calculate_accuracy(predictions_train, training_labels)

        # Evaluate metrics on the validation data
        predictions_val, _, _ = model_forward_with_bn(
            network, 
            validation_data, 
            activation=activation_fn, 
            dropout_rate=0.0,  # Disable Dropout during evaluation
            training=False
        )
        loss_val = calculate_loss(validation_labels, predictions_val, network, regularisation_strength)
        accuracy_val = calculate_accuracy(predictions_val, validation_labels)

        # Update the best model if validation accuracy improves
        if highest_val_accuracy < accuracy_val:
            highest_val_accuracy = accuracy_val
            optimal_model = network

        # Append history for metrics
        history_loss_train.append(loss_train)
        history_accuracy_train.append(accuracy_train)
        history_loss_val.append(loss_val)
        history_accuracy_val.append(accuracy_val)

        # Decay learning rate
        lr_initial *= lr_decay

        # Print epoch metrics
        print(f'Epoch {epoch+1} of {total_epochs}: '
              f'Train Loss = {loss_train:.4f}, Train Accuracy = {accuracy_train:.4%}; '
              f'Val Loss = {loss_val:.4f}, Val Accuracy = {accuracy_val:.4%}')

    return network, optimal_model, history_loss_train, history_accuracy_train, history_loss_val, history_accuracy_val