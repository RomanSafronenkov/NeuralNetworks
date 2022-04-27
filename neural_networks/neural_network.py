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

    def __init__(self, optimizer, random_state=None) -> None:
        self.layers = []
        self.loss = None
        self.loss_derivative = None
        if random_state:
            np.random.seed(random_state)
        self.optimizer = optimizer

    def use(self, loss: callable, loss_derivative: callable) -> None:
        self.loss = loss
        self.loss_derivative = loss_derivative

    def add_layer(self, layer) -> None:
        if 'set_optimizer' in layer.__dir__():
            layer.set_optimizer(self.optimizer)

        self.layers.append(layer)

    def predict(self, x: np.array, grad: bool = False) -> np.array:
        prediction = x
        for layer in self.layers:
            prediction = layer.forward(prediction, grad=grad)
        return prediction

    def fit(self,
            x: np.array,
            y: np.array,
            n_epochs: int,
            x_val: np.array = None,
            y_val: np.array = None,
            batch_size: int = None,
            echo: bool = True
            ):

        batch_size = batch_size or len(x)
        loss_print_epoch = n_epochs / 100
        idxs = np.random.permutation(len(x))
        amount_of_batches = np.ceil(len(x) / batch_size).astype(int)
        metric_name = 'val_loss'

        for _ in range(n_epochs):
            train_error = 0
            for batch_idx in range(amount_of_batches):
                batch_slice = idxs[batch_idx * batch_size:batch_idx * batch_size + batch_size]
                x_batch = x[batch_slice]
                y_batch = y[batch_slice]

                preds = self.predict(x_batch, grad=True)
                train_error += self.loss(y_batch, preds)
                output_error = self.loss_derivative(y_batch, preds)
                for layer in reversed(self.layers):
                    output_error = layer.backward(output_error)
            if echo:
                if x_val is not None and y_val is not None:
                    err_val = self.loss(y_val, self.predict(x_val))
                    if _ % loss_print_epoch == 0:
                        print('*' * 30)
                        print(f'Epoch {_}  train_loss:{train_error / amount_of_batches}, {metric_name}:{err_val}')
                else:
                    if _ % loss_print_epoch == 0:
                        print('*' * 30)
                        print(f'Epoch {_}  train_loss:{train_error / amount_of_batches}')

        return self
