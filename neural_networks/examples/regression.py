import numpy as np

from utils.layers import Linear, Activation, DropOut
from utils.loss_functions import mse, mse_derivative, mae, mae_derivative
from utils.activation_functions import relu, relu_derivative
from utils.metrics import r2_score
from utils.preprocessing import StandardScaler
from utils.preprocessing_utils import train_test_split
from neural_network import NeuralNetwork

from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestRegressor

# load data
data = load_boston()
x = data['data']
y = data['target'].reshape(-1, 1)

# split data
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.33, random_state=1)

# scaling features
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_val = scaler.transform(x_val)

# initialize NN
regression_nn = NeuralNetwork(42)
regression_nn.use(mae, mae_derivative)  # loss function
regression_nn.add_layer(Linear(13, 128))
regression_nn.add_layer(Activation(relu, relu_derivative))
regression_nn.add_layer(DropOut(0.4))
regression_nn.add_layer(Linear(128, 64))
regression_nn.add_layer(Activation(relu, relu_derivative))
regression_nn.add_layer(DropOut(0.3))
regression_nn.add_layer(Linear(64, 1))


regression_nn.fit(
    x_train,
    y_train,
    learning_rate=0.001,
    n_epochs=10000,
    x_val=x_val,
    y_val=y_val,
    echo=False
)

preds = regression_nn.predict(x_val)
print(f'NeuralNetwork using MAE: mse={mse(y_val, preds)}, mae={mae(y_val, preds)}, r2={r2_score(y_val, preds)}')

sklearn_forest = RandomForestRegressor(criterion='mae')
sklearn_forest.fit(x_train, y_train.ravel())
skl_preds = sklearn_forest.predict(x_val)

print(f'RandomForestRegressor using MAE: mse={mse(y_val, skl_preds)}, mae={mae(y_val, skl_preds)}, '
      f'r2={r2_score(y_val, skl_preds)}')

#  let's use MSE as a loss function
regression_nn = NeuralNetwork(42)
regression_nn.use(mse, mse_derivative)  # loss function
regression_nn.add_layer(Linear(13, 16))
regression_nn.add_layer(Activation(relu, relu_derivative))
regression_nn.add_layer(Linear(16, 4))
regression_nn.add_layer(Activation(relu, relu_derivative))
regression_nn.add_layer(Linear(4, 1))

regression_nn.fit(
    x_train,
    y_train,
    learning_rate=0.001,
    n_epochs=10000,
    x_val=x_val,
    y_val=y_val,
    echo=False
)

preds = regression_nn.predict(x_val)
print(f'NeuralNetwork using MSE: mse={mse(y_val, preds)}, mae={mae(y_val, preds)}, r2={r2_score(y_val, preds)}')

sklearn_forest = RandomForestRegressor(criterion='mse')
sklearn_forest.fit(x_train, y_train.ravel())
skl_preds = sklearn_forest.predict(x_val)
print(f'RandomForestRegressor using MSE: mse={mse(y_val, skl_preds)}, mae={mae(y_val, skl_preds)}, '
      f'r2={r2_score(y_val, skl_preds)}')

# ols with gradient descent
regression_nn = NeuralNetwork(42)
regression_nn.use(mse, mse_derivative)  # loss function
regression_nn.add_layer(Linear(13, 1))

regression_nn.fit(
    x_train,
    y_train,
    learning_rate=0.1,
    n_epochs=10000,
    x_val=x_val,
    y_val=y_val,
    echo=False
)

preds = regression_nn.predict(x_val)
print(f'NeuralNetwork OLS: mse={mse(y_val, preds)}, mae={mae(y_val, preds)}, r2={r2_score(y_val, preds)}')

x_ols = np.hstack((np.ones((len(x_train), 1)), x_train))
w_ols = np.linalg.inv(x_ols.T.dot(x_ols)).dot(x_ols.T).dot(y_train)

nn_ols_w = np.vstack((regression_nn.layers[0].b, regression_nn.layers[0].w))

print('NN OLS weights:', nn_ols_w.ravel())
print('OLS normal equation', w_ols.ravel())
print('Weights NN and OLS are close np.allclose():', np.allclose(w_ols.ravel(), nn_ols_w.ravel()))
