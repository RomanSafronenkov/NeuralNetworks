import logging

import numpy as np
from sklearn.datasets import load_breast_cancer

from neural_networks.neural_network import NeuralNetwork
from neural_networks.utils.activation_functions import relu, relu_derivative
from neural_networks.utils.layers import Linear, Activation, DropOut
from neural_networks.utils.loss_functions import cross_entropy_loss, cross_entropy_loss_derivative
from neural_networks.utils.metrics import accuracy_score
from neural_networks.utils.optimizers import SGD
from neural_networks.utils.preprocessing import StandardScaler
from neural_networks.utils.preprocessing_utils import train_test_split

logging.basicConfig(format='[%(asctime)s] - [%(levelname)s] - %(message)s',
                    level=logging.INFO)
_logger = logging.getLogger(__name__)

# load data
data = load_breast_cancer()
x = data['data']
y = data['target'].reshape(-1, 1)

_logger.info(f'Data: {data["filename"]}\nx shape: {x.shape}, y shape: {y.shape}')

# split data
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.33, random_state=1)

# scaling features
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_val = scaler.transform(x_val)

# create NN with log_loss for binary classification
optimizer = SGD(p=0.5, learning_rate=0.01)
classification_nn = NeuralNetwork(optimizer=optimizer, random_state=42)
classification_nn.use(cross_entropy_loss, cross_entropy_loss_derivative)
classification_nn.add_layer(Linear(x_train.shape[1], 60))
classification_nn.add_layer(Activation(relu, relu_derivative))
classification_nn.add_layer(Linear(60, 30))
classification_nn.add_layer(Activation(relu, relu_derivative))
classification_nn.add_layer(Linear(30, 2))

classification_nn.fit(
    x=x_train,
    y=y_train,
    n_epochs=10000,
    x_val=x_val,
    y_val=y_val,
    echo=False
)

preds = np.argmax(classification_nn.predict(x_val), axis=1)  # just simple threshold for demonstration
_logger.info(f'CrossEntropyLoss NN\naccuracy score= {accuracy_score(y_val, preds)}')

# let's try the same NN but with dropout

classification_nn = NeuralNetwork(optimizer=optimizer, random_state=42)
classification_nn.use(cross_entropy_loss, cross_entropy_loss_derivative)
classification_nn.add_layer(Linear(x_train.shape[1], 60))
classification_nn.add_layer(Activation(relu, relu_derivative))
classification_nn.add_layer(DropOut(0.5))
classification_nn.add_layer(Linear(60, 30))
classification_nn.add_layer(Activation(relu, relu_derivative))
classification_nn.add_layer(DropOut(0.6))
classification_nn.add_layer(Linear(30, 2))

classification_nn.fit(
    x=x_train,
    y=y_train,
    n_epochs=10000,
    x_val=x_val,
    y_val=y_val,
    echo=False
)

preds = np.argmax(classification_nn.predict(x_val), axis=1)  # just simple threshold for demonstration
_logger.info(f'CrossEntropyLoss NN with dropout\naccuracy score= {accuracy_score(y_val, preds)}')
