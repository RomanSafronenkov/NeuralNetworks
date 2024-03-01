import copy

import numpy as np

from neural_networks.nn import Activation, sigmoid, sigmoid_derivative, tanh, tanh_derivative, Linear
from neural_networks.nn.layers import BaseLayer


class Linear3d(BaseLayer):
    """
    Linear class permorms ordinary FC layer in neural networks, but for 3D input
    Parameters:
    n_input - size of input neurons
    n_output - size of output neurons
    Methods:
    set_optimizer(optimizer) - is used for setting an optimizer for gradient descent
    forward(x) - performs forward pass of the layer
    backward(output_error, learning_rate) - performs backward pass of the layer
    """

    def __init__(self, n_input: int, n_output: int) -> None:
        super().__init__()
        self.input = None
        self.n_input = n_input
        self.n_output = n_output
        self.w = np.random.normal(scale=np.sqrt(2 / (n_input + n_output)), size=(n_input, n_output))
        self.b = np.random.normal(scale=np.sqrt(2 / (n_input + n_output)), size=(1, n_output))

        self.w_optimizer = None
        self.b_optimizer = None

    def set_optimizer(self, optimizer) -> None:
        self.w_optimizer = copy.copy(optimizer)
        self.b_optimizer = copy.copy(optimizer)

        self.w_optimizer.set_weight(self.w)
        self.b_optimizer.set_weight(self.b)

    def forward(self, x: np.array, grad: bool = True) -> np.array:
        self.input = x
        return np.matmul(x, self.w) + self.b  # the same as @

    def backward(self, output_error: np.array) -> np.array:
        assert self.w_optimizer is not None and self.b_optimizer is not None, 'You should set an optimizer'
        # перемножаем последние 2 измерения друг с другом с помощью matmul и суммируем
        w_grad = np.sum(np.transpose(self.input, (0, 2, 1)) @ output_error, axis=0)
        b_grad = np.sum(output_error, axis=(0, 1))
        input_error = output_error @ self.w.T

        self.w = self.w_optimizer.step(w_grad)
        self.b = self.b_optimizer.step(b_grad)
        return input_error


class Conv1d(BaseLayer):
    """
    Сверточный слой, со страйдом 1 и без паддингов, для батча
    """

    def __init__(self, in_channels, out_channels, kernel_size):
        self.bias_optimizer = None
        self.kernel_optimizer = None
        self.output_len = None
        self.input_len = None
        self.batch_size = None
        self.input = None
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        scale = np.sqrt(1 / (in_channels * kernel_size))
        self.kernel = np.random.uniform(-scale, scale, size=(out_channels, in_channels, kernel_size))
        self.bias = np.random.uniform(-scale, scale, size=out_channels)

    def set_optimizer(self, optimizer):
        self.kernel_optimizer = copy.copy(optimizer)
        self.bias_optimizer = copy.copy(optimizer)

        self.kernel_optimizer.set_weight(self.kernel)
        self.bias_optimizer.set_weight(self.bias)

    def forward(self, x, grad=True):
        """
        Работает с битчами вида [BATCH_SIZE, SENTENCE_LEN, EMB_DIM]
        """
        self.input = x
        self.batch_size = x.shape[0]
        self.input_len = x.shape[1]
        self.output_len = self.input_len - self.kernel_size + 1

        result = []

        for sentence in x:
            result.append(self._forward_for_one(sentence))

        return np.array(result)

    def _forward_for_one(self, x):
        """
        Просто свертка для 1 предложения
        """
        output = np.zeros(shape=(self.output_len, self.out_channels))

        # для каждого выходного канала и ядра, отвечающего за этот канал
        for kernel_i, ker in enumerate(self.kernel):
            # по выходной длине
            for i in range(self.output_len):
                # умножаем срез по размеру ядра на ядро и суммируем
                output[i:self.kernel_size + i, kernel_i] = self.bias[kernel_i] + np.sum(
                    x[i:self.kernel_size + i, :] * ker.T)

        return output

    def backward(self, output_error):
        """
        Градиенты по всемy батчу
        """
        dy_dkernels = []
        dy_dbiass = []
        dy_dxs = []

        for i in range(self.batch_size):
            dy_dkernel, dy_dbias, dy_dx = self._calc_grad_for_one(output_error[i], self.input[i])
            dy_dkernels.append(dy_dkernel)
            dy_dbiass.append(dy_dbias)
            dy_dxs.append(dy_dx)

        dy_dkernels = np.sum(np.array(dy_dkernels), axis=0)  # суммируем градиенты по батчу
        dy_dbiass = np.sum(np.array(dy_dbiass), axis=0)
        dy_dxs = np.array(dy_dxs)

        self.kernel = self.kernel_optimizer.step(dy_dkernels)  # делаем шаг спуска по сумме градиентов
        self.bias = self.bias_optimizer.step(dy_dbiass)

        return dy_dxs

    def _calc_grad_for_one(self, output_error, x):
        dy_dkernel = np.zeros(shape=self.kernel.shape)
        dy_dbias = np.zeros(shape=self.bias.shape)
        dy_dx = np.zeros(shape=x.shape)

        for kernel_i, ker in enumerate(self.kernel):
            helper_k = np.zeros(shape=ker.T.shape)

            for i in range(self.output_len):
                helper_k += x[i:self.kernel_size + i, :] * output_error[i, kernel_i]
                dy_dx[i:self.kernel_size + i, :] += ker.T * output_error[i, kernel_i]

            dy_dkernel[kernel_i] = helper_k.T
            dy_dbias[kernel_i] = np.sum(output_error[:, kernel_i])

        return dy_dkernel, dy_dbias, dy_dx


class LinearLSTM(BaseLayer):
    """
    Linear class permorms ordinary FC layer in neural networks, but it returns grads in backward
    Parameters:
    n_input - size of input neurons
    n_output - size of output neurons
    Methods:
    set_optimizer(optimizer) - is used for setting an optimizer for gradient descent
    forward(x) - performs forward pass of the layer
    backward(output_error, learning_rate) - performs backward pass of the layer
    """

    def __init__(self, n_input: int, n_output: int) -> None:
        super().__init__()
        self.input = None
        self.n_input = n_input
        self.n_output = n_output
        self.w = np.random.normal(scale=np.sqrt(2 / (n_input + n_output)), size=(n_input, n_output))
        self.b = np.random.normal(scale=np.sqrt(2 / (n_input + n_output)), size=(1, n_output))

        self.w_optimizer = None
        self.b_optimizer = None

    def set_optimizer(self, optimizer) -> None:
        self.w_optimizer = copy.copy(optimizer)
        self.b_optimizer = copy.copy(optimizer)

        self.w_optimizer.set_weight(self.w)
        self.b_optimizer.set_weight(self.b)

    def forward(self, x: np.array, grad: bool = True) -> np.array:
        self.input = x
        return x.dot(self.w) + self.b

    def backward(self, output_error: np.array) -> np.array:
        assert self.w_optimizer is not None and self.b_optimizer is not None, 'You should set an optimizer'
        w_grad = self.input.T.dot(output_error)
        b_grad = np.ones((1, len(output_error))).dot(output_error)
        input_error = output_error.dot(self.w.T)

        return w_grad, b_grad, input_error

    def step(self, w_grad, b_grad):
        self.w = self.w_optimizer.step(w_grad)
        self.b = self.b_optimizer.step(b_grad)


class LSTM(BaseLayer):
    def __init__(
            self,
            n_input,
            n_hidden,
            n_output,
            bptt_trunc=4
    ):
        self.input = None
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.bptt_trunc = bptt_trunc

        self.forget_gate = LinearLSTM(n_input + n_hidden, n_hidden)
        self.forget_gate_activation = Activation(sigmoid, sigmoid_derivative)

        self.input_layer_gate = LinearLSTM(n_input + n_hidden, n_hidden)
        self.input_layer_gate_activation = Activation(sigmoid, sigmoid_derivative)

        self.candidate_gate = LinearLSTM(n_input + n_hidden, n_hidden)
        self.candidate_gate_activation = Activation(tanh, tanh_derivative)

        self.output_gate = LinearLSTM(n_input + n_hidden, n_hidden)
        self.output_gate_activation = Activation(sigmoid, sigmoid_derivative)

        self.output_to_logits = LinearLSTM(n_hidden, n_output)

        self.x_and_h = None
        self.hidden_states = None
        self.cell_states = None

        self.forget_inputs = None
        self.forget_states = None

        self.input_inputs = None
        self.input_states = None

        self.candidate_inputs = None
        self.candidate_states = None

        self.output_input = None
        self.output_states = None

        self.outputs = None

    def set_optimizer(self, optimizer) -> None:
        self.forget_gate.set_optimizer(optimizer)
        self.input_layer_gate.set_optimizer(optimizer)
        self.candidate_gate.set_optimizer(optimizer)
        self.output_gate.set_optimizer(optimizer)
        self.output_to_logits.set_optimizer(optimizer)

    def forward(self, x, grad=True):
        self.input = x
        batch_size, timesteps, input_dim = x.shape

        self.x_and_h = np.zeros((batch_size, timesteps, self.n_input + self.n_hidden))
        self.hidden_states = np.zeros((batch_size, timesteps + 1, self.n_hidden))
        self.cell_states = np.zeros((batch_size, timesteps + 1, self.n_hidden))

        self.forget_inputs = np.zeros((batch_size, timesteps, self.n_hidden))
        self.forget_states = np.zeros((batch_size, timesteps, self.n_hidden))

        self.input_inputs = np.zeros((batch_size, timesteps, self.n_hidden))
        self.input_states = np.zeros((batch_size, timesteps, self.n_hidden))

        self.candidate_inputs = np.zeros((batch_size, timesteps, self.n_hidden))
        self.candidate_states = np.zeros((batch_size, timesteps, self.n_hidden))

        self.output_input = np.zeros((batch_size, timesteps, self.n_hidden))
        self.output_states = np.zeros((batch_size, timesteps, self.n_hidden))

        self.outputs = np.zeros((batch_size, timesteps, self.n_output))

        self.hidden_states[:, -1] = np.zeros((batch_size, self.n_hidden))
        self.cell_states[:, -1] = np.zeros((batch_size, self.n_hidden))
        for t in range(timesteps):
            # соединяем вход и прошлый h
            self.x_and_h[:, t] = np.concatenate((x[:, t], self.hidden_states[:, t - 1]), axis=1)

            # forget gate проход
            self.forget_inputs[:, t] = self.forget_gate(self.x_and_h[:, t])
            self.forget_states[:, t] = self.forget_gate_activation(self.forget_inputs[:, t])

            # выбор кандидатов для C
            self.input_inputs[:, t] = self.input_layer_gate(self.x_and_h[:, t])
            self.input_states[:, t] = self.input_layer_gate_activation(self.input_inputs[:, t])

            self.candidate_inputs[:, t] = self.candidate_gate(self.x_and_h[:, t])
            self.candidate_states[:, t] = self.candidate_gate_activation(self.candidate_inputs[:, t])

            self.cell_states[:, t] = self.forget_states[:, t] * self.cell_states[:, t - 1] \
                                     + self.input_states[:, t] * self.candidate_states[:, t]

            self.output_input[:, t] = self.output_gate(self.x_and_h[:, t])
            self.output_states[:, t] = self.output_gate_activation(self.output_input[:, t])

            self.hidden_states[:, t] = self.output_states[:, t] * tanh(self.cell_states[:, t])

            # дополнительный слой, не указан на рисунке, для перевода состояния в логиты выхода
            self.outputs[:, t] = self.output_to_logits(self.hidden_states[:, t])

        return self.outputs

    def backward(self, output_error):
        _, timesteps, _ = output_error.shape

        forgate_gate_w_grad = np.zeros_like(self.forget_gate.w)
        forgate_gate_b_grad = np.zeros_like(self.forget_gate.b)

        input_layer_gate_w_grad = np.zeros_like(self.input_layer_gate.w)
        input_layer_gate_b_grad = np.zeros_like(self.input_layer_gate.b)

        candidate_gate_w_grad = np.zeros_like(self.candidate_gate.w)
        candidate_gate_b_grad = np.zeros_like(self.candidate_gate.b)

        output_gate_w_grad = np.zeros_like(self.output_gate.w)
        output_gate_b_grad = np.zeros_like(self.output_gate.b)

        output_to_logits_w_grad = np.zeros_like(self.output_to_logits.w)
        output_to_logits_b_grad = np.zeros_like(self.output_to_logits.b)

        input_error = np.zeros_like(self.input)

        for t in np.arange(timesteps)[::-1]:
            # в разные моменты времени у слоев был разный вход, необходимо искусственно его поменять
            self.forget_gate.input = self.x_and_h[:, t]
            self.forget_gate_activation.input = self.forget_inputs[:, t]
            self.input_layer_gate.input = self.x_and_h[:, t]
            self.input_layer_gate_activation.input = self.input_inputs[:, t]
            self.candidate_gate.input = self.x_and_h[:, t]
            self.candidate_gate_activation.input = self.candidate_inputs[:, t]
            self.output_gate.input = self.x_and_h[:, t]
            self.output_gate_activation.input = self.output_input[:, t]
            self.output_to_logits.input = self.hidden_states[:, t]

            # проход по нижнему уровню
            w_grad, b_grad, hidden_error = self.output_to_logits.backward(output_error[:, t])
            output_to_logits_w_grad += w_grad
            output_to_logits_b_grad += b_grad

            # та, что идет вверх
            cell_error = tanh_derivative(self.cell_states[:, t]) * self.output_states[:, t] * hidden_error

            # ошибка идет и вниз и вверх
            hidden_error = self.output_gate_activation.backward(hidden_error) * tanh(self.cell_states[:, t])

            # та, что идет вниз
            w_grad, b_grad, hidden_error = self.output_gate.backward(hidden_error)
            output_gate_w_grad += w_grad
            output_gate_b_grad += b_grad

            # идем по верху
            hidden_candidate_error = self.candidate_gate_activation.backward(cell_error) * self.input_states[:, t]
            w_grad, b_grad, hidden_candidate_error = self.candidate_gate.backward(hidden_candidate_error)
            candidate_gate_w_grad += w_grad
            candidate_gate_b_grad += b_grad

            hidden_inputs_error = self.input_layer_gate_activation.backward(cell_error) * self.candidate_states[:, t]
            w_grad, b_grad, hidden_inputs_error = self.input_layer_gate.backward(hidden_inputs_error)
            input_layer_gate_w_grad += w_grad
            input_layer_gate_b_grad += b_grad

            hidden_forget_error = self.forget_gate_activation.backward(cell_error) * self.cell_states[:, t - 1]
            w_grad, b_grad, hidden_forget_error = self.forget_gate.backward(hidden_forget_error)
            forgate_gate_w_grad += w_grad
            forgate_gate_b_grad += b_grad

            # добавляются ошибки с мест копии
            hidden_error += hidden_candidate_error
            hidden_error += hidden_inputs_error
            hidden_error += hidden_forget_error

            # ошибка входа
            input_error[:, t] = hidden_error[:, :self.n_input]
            # ошибка, которая по времени уходит по низу назад
            hidden_error = hidden_error[:, self.n_input:]
            # ошибка, которая по времени уходит по верху назад
            cell_error = cell_error * self.forget_states[:, t]

            for t_ in np.arange(max(0, t - self.bptt_trunc), t)[::-1]:
                # проход по времени
                self.forget_gate.input = self.x_and_h[:, t_]
                self.forget_gate_activation.input = self.forget_inputs[:, t_]
                self.input_layer_gate.input = self.x_and_h[:, t_]
                self.input_layer_gate_activation.input = self.input_inputs[:, t_]
                self.candidate_gate.input = self.x_and_h[:, t_]
                self.candidate_gate_activation.input = self.candidate_inputs[:, t_]
                self.output_gate.input = self.x_and_h[:, t_]
                self.output_gate_activation.input = self.output_input[:, t_]

                # та, что идет по верху
                cell_error += tanh_derivative(self.cell_states[:, t_]) * self.output_states[:, t_] \
                              * hidden_error

                hidden_error = self.output_gate_activation.backward(hidden_error) * tanh(self.cell_states[:, t_])
                # та, что идет вниз
                w_grad, b_grad, hidden_error = self.output_gate.backward(hidden_error)
                output_gate_w_grad += w_grad
                output_gate_b_grad += b_grad

                hidden_candidate_error = self.candidate_gate_activation.backward(cell_error) * self.input_states[:, t_]
                w_grad, b_grad, hidden_candidate_error = self.candidate_gate.backward(hidden_candidate_error)
                candidate_gate_w_grad += w_grad
                candidate_gate_b_grad += b_grad

                hidden_inputs_error = self.input_layer_gate_activation.backward(cell_error) * self.candidate_states[:,
                                                                                              t_]
                w_grad, b_grad, hidden_inputs_error = self.input_layer_gate.backward(hidden_inputs_error)
                input_layer_gate_w_grad += w_grad
                input_layer_gate_b_grad += b_grad

                hidden_forget_error = self.forget_gate_activation.backward(cell_error) * self.cell_states[:, t_ - 1]
                w_grad, b_grad, hidden_forget_error = self.forget_gate.backward(hidden_forget_error)
                forgate_gate_w_grad += w_grad
                forgate_gate_b_grad += b_grad

                # добавляются ошибки с мест копии
                hidden_error += hidden_candidate_error
                hidden_error += hidden_inputs_error
                hidden_error += hidden_forget_error

                # ошибка входа
                input_error[:, t_] += hidden_error[:, :self.n_input]
                # ошибка которая по времени уходит по низу назад
                hidden_error = hidden_error[:, self.n_input:]
                cell_error = cell_error * self.forget_states[:, t_]

        # накопили градиенты и только тогда делаем шаг
        self.forget_gate.step(forgate_gate_w_grad, forgate_gate_b_grad)
        self.input_layer_gate.step(input_layer_gate_w_grad, input_layer_gate_b_grad)
        self.candidate_gate.step(candidate_gate_w_grad, candidate_gate_b_grad)
        self.output_gate.step(output_gate_w_grad, output_gate_b_grad)
        self.output_to_logits.step(output_to_logits_w_grad, output_to_logits_b_grad)

        return input_error


class Embedding(BaseLayer):
    def __init__(self, n_input, emb_dim, pad_idx=None):
        self.input = None
        self.weights_optimizer = None
        self.n_input = n_input
        self.emb_dim = emb_dim
        self.pad_idx = pad_idx

        self.weights = np.random.normal(scale=np.sqrt(2 / (n_input + emb_dim)), size=(n_input, emb_dim))

    def set_optimizer(self, optimizer):
        self.weights_optimizer = copy.copy(optimizer)

        self.weights_optimizer.set_weight(self.weights)

    def forward(self, x, grad=True):
        self.input = x
        return self.weights[x]

    def backward(self, output_error):
        weights_grad = np.zeros_like(self.weights)
        input_shape_len = len(self.input.shape)

        if input_shape_len == 2:
            for batch_n, s in enumerate(self.input):
                for i, emb_i in enumerate(s):
                    weights_grad[emb_i] += output_error[batch_n][i]

        elif input_shape_len == 1:
            for i, emb_i in enumerate(self.input):
                weights_grad[emb_i] += output_error[i]

        if self.pad_idx is not None:
            weights_grad[self.pad_idx] = 0

        self.weights = self.weights_optimizer.step(weights_grad)


class RNNCell(BaseLayer):
    def __init__(
            self,
            n_input,
            n_hidden,
            hidden_activation,
            hidden_activation_derivative,
            bptt_trunc=4
    ):
        self.input = None
        self.b_y_optimizer = None
        self.b_h_optimizer = None
        self.w_ho_optimizer = None
        self.w_hh_optimizer = None
        self.w_in_optimizer = None
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.bptt_trunc = bptt_trunc

        self.w_in = np.random.normal(scale=np.sqrt(2 / (n_input + n_hidden)), size=(n_input, n_hidden))
        self.w_hh = np.random.normal(scale=np.sqrt(1 / n_hidden), size=(n_hidden, n_hidden))
        self.w_ho = np.random.normal(scale=np.sqrt(2 / (n_input + n_hidden)), size=(n_hidden, n_input))

        self.b_h = np.random.normal(scale=np.sqrt(1 / n_hidden), size=(1, n_hidden))
        self.b_y = np.random.normal(scale=np.sqrt(1 / n_hidden), size=(1, n_input))

        self.hidden_activation = hidden_activation
        self.hidden_activation_derivative = hidden_activation_derivative

        self.state_inputs = None
        self.hidden_states = None
        self.outputs = None

    def set_optimizer(self, optimizer) -> None:
        self.w_in_optimizer = copy.copy(optimizer)
        self.w_hh_optimizer = copy.copy(optimizer)
        self.w_ho_optimizer = copy.copy(optimizer)

        self.b_h_optimizer = copy.copy(optimizer)
        self.b_y_optimizer = copy.copy(optimizer)

        self.w_in_optimizer.set_weight(self.w_in)
        self.w_hh_optimizer.set_weight(self.w_hh)
        self.w_ho_optimizer.set_weight(self.w_ho)

        self.b_h_optimizer.set_weight(self.b_h)
        self.b_y_optimizer.set_weight(self.b_y)

    def forward(self, x: np.array, grad: bool = True) -> np.array:
        self.input = x
        batch_size, timesteps, input_dim = x.shape

        self.state_inputs = np.zeros((batch_size, timesteps, self.n_hidden))
        self.hidden_states = np.zeros((batch_size, timesteps + 1, self.n_hidden))
        self.outputs = np.zeros((batch_size, timesteps, self.n_input))

        self.hidden_states[:, -1] = np.zeros((batch_size, self.n_hidden))
        for t in range(timesteps):
            self.state_inputs[:, t] = x[:, t].dot(self.w_in) + self.hidden_states[:, t - 1].dot(self.w_hh) + self.b_h
            self.hidden_states[:, t] = self.hidden_activation(self.state_inputs[:, t])
            self.outputs[:, t] = self.hidden_states[:, t].dot(self.w_ho) + self.b_y

        return self.outputs

    def backward(self, output_error: np.array) -> np.array:
        _, timesteps, _ = output_error.shape

        w_in_grad = np.zeros_like(self.w_in)
        w_hh_grad = np.zeros_like(self.w_hh)
        w_ho_grad = np.zeros_like(self.w_ho)
        b_h_grad = np.zeros_like(self.b_h)
        b_y_grad = np.zeros_like(self.b_y)
        input_error = np.zeros_like(output_error)

        for t in np.arange(timesteps)[::-1]:
            w_ho_grad += self.hidden_states[:, t].T.dot(output_error[:, t])
            b_y_grad += np.ones((1, len(output_error[:, t]))).dot(output_error[:, t])
            hidden_error = output_error[:, t].dot(self.w_ho.T) * self.hidden_activation_derivative(
                self.state_inputs[:, t])
            input_error[:, t] = hidden_error.dot(self.w_in.T)
            for t_ in np.arange(max(0, t - self.bptt_trunc), t + 1)[::-1]:
                w_in_grad += self.input[:, t_].T.dot(hidden_error)
                w_hh_grad += self.hidden_states[:, t_ - 1].T.dot(hidden_error)
                b_h_grad += np.ones((1, len(hidden_error))).dot(hidden_error)
                hidden_error = hidden_error.dot(self.w_hh.T) * self.hidden_activation_derivative(
                    self.state_inputs[:, t_ - 1])

        self.w_in = self.w_in_optimizer.step(w_in_grad)
        self.w_hh = self.w_hh_optimizer.step(w_hh_grad)
        self.w_ho = self.w_ho_optimizer.step(w_ho_grad)
        self.b_h = self.b_h_optimizer.step(b_h_grad)
        self.b_y = self.b_y_optimizer.step(b_y_grad)

        return input_error


class RNNUnit(BaseLayer):
    def __init__(
            self,
            n_input,
            n_hidden,
            bptt_trunc=4
    ):
        self.input = None
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.bptt_trunc = bptt_trunc

        self.rnn_update = Linear(n_input + n_hidden, n_hidden)
        self.hidden_activation = Activation(tanh, tanh_derivative)
        self.rnn_to_logits = Linear(n_hidden, n_input)

        self.x_and_h = None
        self.state_inputs = None
        self.hidden_states = None
        self.outputs = None

    def set_optimizer(self, optimizer) -> None:
        self.rnn_update.set_optimizer(optimizer)
        self.rnn_to_logits.set_optimizer(optimizer)

    def forward(self, x: np.array, grad: bool = True) -> np.array:
        self.input = x
        batch_size, timesteps, input_dim = x.shape

        self.x_and_h = np.zeros((batch_size, timesteps, self.n_input + self.n_hidden))
        self.state_inputs = np.zeros((batch_size, timesteps, self.n_hidden))
        self.hidden_states = np.zeros((batch_size, timesteps + 1, self.n_hidden))
        self.outputs = np.zeros((batch_size, timesteps, self.n_input))

        self.hidden_states[:, -1] = np.zeros((batch_size, self.n_hidden))
        for t in range(timesteps):
            self.x_and_h[:, t] = np.concatenate((x[:, t], self.hidden_states[:, t - 1]), axis=1)
            self.state_inputs[:, t] = self.rnn_update(self.x_and_h[:, t])
            self.hidden_states[:, t] = self.hidden_activation(self.state_inputs[:, t])
            self.outputs[:, t] = self.rnn_to_logits(self.hidden_states[:, t])

        return self.outputs

    def backward(self, output_error: np.array) -> np.array:
        _, timesteps, _ = output_error.shape

        input_error = np.zeros_like(output_error)
        for t in np.arange(timesteps)[::-1]:
            self.rnn_to_logits.input = self.hidden_states[:, t]
            self.rnn_update.input = self.x_and_h[:, t]
            self.hidden_activation.input = self.state_inputs[:, t]

            hidden_error = self.rnn_to_logits.backward(output_error[:, t])
            hidden_error = self.hidden_activation.backward(hidden_error)
            hidden_and_input_error = self.rnn_update.backward(hidden_error)

            input_error[:, t] = hidden_and_input_error[:, :self.n_input]
            hidden_error = hidden_and_input_error[:, -self.n_hidden:]

            for t_ in np.arange(max(0, t - self.bptt_trunc), t)[::-1]:
                self.rnn_update.input = self.x_and_h[:, t_]
                self.hidden_activation.input = self.state_inputs[:, t_]

                hidden_error = self.hidden_activation.backward(hidden_error)
                hidden_error = self.rnn_update.backward(hidden_error)[:, -self.n_hidden:]

        return input_error
