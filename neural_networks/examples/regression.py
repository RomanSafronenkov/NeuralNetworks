import logging

import numpy as np
from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestRegressor

from neural_networks.neural_network import NeuralNetwork
from neural_networks.utils.activation_functions import relu, relu_derivative
from neural_networks.utils.layers import Linear, Activation, DropOut
from neural_networks.utils.loss_functions import mae, mae_derivative, mse_derivative, mse
from neural_networks.utils.metrics import r2_score
from neural_networks.utils.optimizers import SGD, ADAM
from neural_networks.utils.preprocessing import StandardScaler
from neural_networks.utils.preprocessing_utils import train_test_split

logging.basicConfig(format='[%(asctime)s] - [%(levelname)s] - %(message)s',
                    level=logging.INFO)
_logger = logging.getLogger(__name__)

# load data
data = load_boston()
x = data['data']
y = data['target'].reshape(-1, 1)

_logger.info(f'Data: {data["filename"]}\nx shape: {x.shape}, y shape: {y.shape}')
# split data
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.33, random_state=1)

# scaling features
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_val = scaler.transform(x_val)

# initialize NN with mae and some dropout layers
optimizer = SGD(p=0.3, learning_rate=0.1)
regression_nn = NeuralNetwork(optimizer=optimizer, random_state=42)
regression_nn.use(mae, mae_derivative)  # loss function
regression_nn.add_layer(Linear(13, 64))
regression_nn.add_layer(Activation(relu, relu_derivative))
regression_nn.add_layer(DropOut(0.4))
regression_nn.add_layer(Linear(64, 32))
regression_nn.add_layer(Activation(relu, relu_derivative))
regression_nn.add_layer(DropOut(0.3))
regression_nn.add_layer(Linear(32, 1))

regression_nn.fit(
    x_train,
    y_train,
    n_epochs=10000,
    x_val=x_val,
    y_val=y_val,
    echo=False
)

# write the predictions
preds = regression_nn.predict(x_val)
_logger.info(f'NeuralNetwork using MAE:\nmse={mse(y_val, preds)}, mae={mae(y_val, preds)}, r2={r2_score(y_val, preds)}')

# compare it with sklearn's forest
sklearn_forest = RandomForestRegressor(criterion='mae')
sklearn_forest.fit(x_train, y_train.ravel())
skl_preds = sklearn_forest.predict(x_val)

_logger.info(f'RandomForestRegressor using MAE:\nmse={mse(y_val, skl_preds)}, mae={mae(y_val, skl_preds)}, '
             f'r2={r2_score(y_val, skl_preds)}')

#  let's use MSE as a loss function
optimizer = SGD(p=0.4, learning_rate=0.001)
regression_nn = NeuralNetwork(optimizer=optimizer, random_state=42)
regression_nn.use(mse, mse_derivative)  # loss function
regression_nn.add_layer(Linear(13, 64))
regression_nn.add_layer(Activation(relu, relu_derivative))
regression_nn.add_layer(Linear(64, 32))
regression_nn.add_layer(Activation(relu, relu_derivative))
regression_nn.add_layer(Linear(32, 1))

regression_nn.fit(
    x_train,
    y_train,
    n_epochs=10000,
    x_val=x_val,
    y_val=y_val,
    echo=False
)

preds = regression_nn.predict(x_val)
_logger.info(f'NeuralNetwork using MSE:\nmse={mse(y_val, preds)}, mae={mae(y_val, preds)}, r2={r2_score(y_val, preds)}')

# let's try the same NN but with dropout to see benefits of its usage
optimizer = ADAM(learning_rate=1e-3)
regression_nn = NeuralNetwork(optimizer=optimizer, random_state=42)
regression_nn.use(mse, mse_derivative)  # loss function
regression_nn.add_layer(Linear(13, 64))
regression_nn.add_layer(Activation(relu, relu_derivative))
regression_nn.add_layer(DropOut(0.4))
regression_nn.add_layer(Linear(64, 32))
regression_nn.add_layer(Activation(relu, relu_derivative))
regression_nn.add_layer(DropOut(0.2))
regression_nn.add_layer(Linear(32, 1))

regression_nn.fit(
    x_train,
    y_train,
    n_epochs=10000,
    x_val=x_val,
    y_val=y_val,
    echo=False
)

preds = regression_nn.predict(x_val)
_logger.info(f'Same NeuralNetwork using MSE and DropOut:\nmse={mse(y_val, preds)}, mae={mae(y_val, preds)},'
             f' r2={r2_score(y_val, preds)}')

# compare with sklearn's forest
sklearn_forest = RandomForestRegressor(criterion='mse')
sklearn_forest.fit(x_train, y_train.ravel())
skl_preds = sklearn_forest.predict(x_val)
_logger.info(f'RandomForestRegressor using MSE:\nmse={mse(y_val, skl_preds)}, mae={mae(y_val, skl_preds)}, '
             f'r2={r2_score(y_val, skl_preds)}')

# ols with gradient descent
optimizer = SGD(p=0, learning_rate=0.1)
regression_nn = NeuralNetwork(optimizer=optimizer, random_state=42)
regression_nn.use(mse, mse_derivative)  # loss function
regression_nn.add_layer(Linear(13, 1))

regression_nn.fit(
    x_train,
    y_train,
    n_epochs=10000,
    x_val=x_val,
    y_val=y_val,
    echo=False
)

preds = regression_nn.predict(x_val)
_logger.info(f'NeuralNetwork OLS:\nmse={mse(y_val, preds)}, mae={mae(y_val, preds)}, r2={r2_score(y_val, preds)}')

# normal equation for OLS
x_ols = np.hstack((np.ones((len(x_train), 1)), x_train))
w_ols = np.linalg.inv(x_ols.T.dot(x_ols)).dot(x_ols.T).dot(y_train)

nn_ols_w = np.vstack((regression_nn.layers[0].b, regression_nn.layers[0].w))

_logger.info(f'NN OLS weights:\n{nn_ols_w.ravel()}')
_logger.info(f'OLS normal equation:\n{w_ols.ravel()}')
_logger.info(f'Weights NN and OLS are close np.allclose(): {np.allclose(w_ols.ravel(), nn_ols_w.ravel())}')
