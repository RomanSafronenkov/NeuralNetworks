import logging

import numpy as np
from sklearn.datasets import load_digits

from neural_networks.neural_network import NeuralNetwork
from neural_networks.nn import relu, relu_derivative, Conv2D, MaxPool2D, Activation, Ravel, Linear, cross_entropy_loss, \
    cross_entropy_loss_derivative, ADAM
from neural_networks.utils import accuracy_score, train_test_split

logging.basicConfig(format='[%(asctime)s] - [%(levelname)s] - %(message)s',
                    level=logging.INFO)
_logger = logging.getLogger(__name__)

x, y = load_digits(return_X_y=True)
x = x.reshape(-1, 1, 8, 8)
y = y.reshape(-1, 1)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
x_train /= 255
x_test /= 255

optimizer = ADAM(learning_rate=1e-4)

classification_nn = NeuralNetwork(optimizer, 42)
classification_nn.use(cross_entropy_loss, cross_entropy_loss_derivative)
classification_nn.add_layer(Conv2D(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2))
classification_nn.add_layer(MaxPool2D(2, 2))
classification_nn.add_layer(Activation(relu, relu_derivative))
classification_nn.add_layer(Conv2D(in_channels=6, out_channels=16, kernel_size=2, stride=1, padding=0))
classification_nn.add_layer(MaxPool2D(2, 1))
classification_nn.add_layer(Activation(relu, relu_derivative))
classification_nn.add_layer(Conv2D(in_channels=16, out_channels=120, kernel_size=2, stride=1, padding=0))
classification_nn.add_layer(Activation(relu, relu_derivative))
classification_nn.add_layer(Ravel())
classification_nn.add_layer(Linear(120, 10))

classification_nn.fit(
    x=x_train,
    y=y_train,
    x_val=x_test,
    y_val=y_test,
    n_epochs=1000,
    batch_size=32,
    echo=True
)

preds = np.argmax(classification_nn.predict(x_test), axis=1).reshape(-1, 1)

_logger.info(f'CNN MNIST 8x8 test accuracy:{accuracy_score(y_test, preds)}')
