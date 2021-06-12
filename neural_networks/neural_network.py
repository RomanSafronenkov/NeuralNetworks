import numpy as np


class NeuralNetwork:
    """
    NeuralNetwork class is used to build the Neural Network and train it
    Methods:
    use(loss, loss_derivative) - set the loss function and it's derivative
    add_layer(layer) - constructor of the NN, add one of the layers described above
    predict(x) - forward pass through the network
    fit(x, y, learning_rate, n_epochs, x_val, y_val, custom_metric, batch_size) - fit the network
    """
    def __init__(self) -> None:
        self.layers = []

    def use(self, loss: callable, loss_derivative: callable) -> None:
        self.loss = loss
        self.loss_derivative = loss_derivative

    def add_layer(self, layer) -> None:
        self.layers.append(layer)

    def predict(self, x: np.array) -> np.array:
        prediction = x
        for layer in self.layers:
            prediction = layer.forward(prediction)
        return prediction

    def fit(self,
            x: np.array,
            y: np.array,
            learning_rate: float,
            n_epochs: int,
            x_val: np.array = None,
            y_val: np.array = None,
            custom_metric: callable = None,
            batch_size: int = None):
        batch_size = batch_size or len(x)
        loss_print_epoch = n_epochs / 100
        idxs = np.random.permutation(len(x))
        amount_of_batches = np.ceil(len(x) / batch_size).astype(int)
        metric_name = 'val_loss' if custom_metric is None else 'custom_metric'

        for _ in range(n_epochs):
            train_error = 0
            for batch_idx in range(amount_of_batches):
                batch_slice = idxs[batch_idx * batch_size:batch_idx * batch_size + batch_size]
                x_batch = x[batch_slice]
                y_batch = y[batch_slice]

                preds = self.predict(x_batch)
                train_error += self.loss(y_batch, preds)
                output_error = self.loss_derivative(y_batch, preds)
                for layer in reversed(self.layers):
                    output_error = layer.backward(output_error, learning_rate)

            if x_val is not None and y_val is not None:
                if custom_metric is not None:
                    err_val = custom_metric(y_val.reshape(-1), np.argmax(self.predict(x_val), axis=1))
                else:
                    err_val = self.loss(y_val, self.predict(x_val))
                if _ % loss_print_epoch == 0:
                    print('*' * 30)
                    print(f'Epoch {_}  train_loss:{train_error / amount_of_batches}, {metric_name}:{err_val}')
            else:
                if _ % loss_print_epoch == 0:
                    print('*' * 30)
                    print(f'Epoch {_}  train_loss:{train_error / amount_of_batches}')

        return self
