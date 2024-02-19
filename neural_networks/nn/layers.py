import copy
from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np

from neural_networks.nn.activation_functions import tanh, tanh_derivative, sigmoid, sigmoid_derivative


class BaseLayer(ABC):
    @abstractmethod
    def __init__(self) -> None:
        pass

    def __call__(self, x: np.array, grad: bool = True) -> np.array:
        return self.forward(x, grad)

    @abstractmethod
    def forward(self, x: np.array, grad: bool = True) -> np.array:
        pass

    @abstractmethod
    def backward(self, output_error: np.array) -> np.array:
        pass


class Linear(BaseLayer):
    """
    Linear class permorms ordinary FC layer in neural networks
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

        self.w = self.w_optimizer.step(w_grad)
        self.b = self.b_optimizer.step(b_grad)
        return input_error


class Activation(BaseLayer):
    """
    Activation class is used for activation function of the FC layer
    Params:
    activation_function - activation function (e.g. sigmoid, RElU, tanh)
    activation_derivative - derivative of the activation function
    Methods:
    forward(x) - performs forward pass of the layer
    backward(output_error, learning_rate) - performs backward pass of the layer
    """

    def __init__(self, activation_function: callable, activation_derivative: callable) -> None:
        super().__init__()
        self.input = None
        self.activation = activation_function
        self.derivative = activation_derivative

    def forward(self, x: np.array, grad: bool = True) -> np.array:
        self.input = x
        return self.activation(x)

    def backward(self, output_error: np.array) -> np.array:
        return output_error * self.derivative(self.input)


class DropOut(BaseLayer):
    """
    DropOut class is used for DropOut layer
    p – probability of an element to be zeroed. Default: 0.5

    https://jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf
    """

    def __init__(self, p):
        super().__init__()
        self.input = None
        self.p = p
        self.q = 1 / (1 - p)
        self.mask = None

    def forward(self, x: np.array, grad: bool = True) -> np.array:
        self.input = x
        if grad:
            self.mask = np.random.uniform(0, 1, size=x.shape) > self.p
            return self.input * self.q * self.mask

        return self.input

    def backward(self, output_error: np.array) -> np.array:
        return output_error * self.q * self.mask


class BatchNorm(BaseLayer):
    """
    num_features – number of features or channels CC of the input
    num_dims – number of input features
    eps – a value added to the denominator for numerical stability. Default: 1e-5
    momentum – the value used for the running_mean and running_var computation.
    Can be set to None for cumulative moving average (i.e. simple average). Default: 0.1
    affine – a boolean value that when set to True, this module has learnable affine parameters. Default: True

    https://arxiv.org/pdf/1502.03167.pdf'
    """

    def __init__(self, num_features, num_dims, eps=1e-05, momentum=0.1, affine=True):
        super().__init__()
        self.input = None
        self.x_centered = None
        self.x_std = None
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine

        if num_dims == 2:
            shape = (1, num_features)
            self.axis = 0
        elif num_dims == 4:
            shape = (1, num_features, 1, 1)
            self.axis = (0, 2, 3)
        else:
            raise ValueError("num_dims must be in (2, 4)")

        self.gamma = np.ones(shape=shape)
        self.beta = np.zeros(shape=shape)
        self.gamma_optimizer = None
        self.beta_optimizer = None

        self.moving_mean = np.zeros(shape=shape)
        self.moving_var = np.zeros(shape=shape)

    def set_optimizer(self, optimizer) -> None:
        self.gamma_optimizer = copy.copy(optimizer)
        self.gamma_optimizer.set_weight(self.gamma)
        if self.affine:
            self.beta_optimizer = copy.copy(optimizer)
            self.beta_optimizer.set_weight(self.beta)

    def forward(self, x: np.array, grad: bool = True) -> np.array:
        self.input = x
        if not grad:
            self.x_centered = (x - self.moving_mean)
            self.x_std = np.sqrt(self.moving_var + self.eps)
            x_hat = self.x_centered / self.x_std
        else:
            assert len(x.shape) in (2, 4)

            mean = x.mean(axis=self.axis, keepdims=True)
            var = ((x - mean) ** 2).mean(axis=self.axis, keepdims=True)

            self.x_centered = (x - mean)
            self.x_std = np.sqrt(var + self.eps)
            x_hat = (x - mean) / np.sqrt(var + self.eps)
            self.moving_mean = self.momentum * self.moving_mean + (1 - self.momentum) * mean
            self.moving_var = self.momentum * self.moving_var + (1 - self.momentum) * var

        y = self.gamma * x_hat + self.beta
        return y

    def backward(self, output_error: np.array) -> np.array:
        assert self.gamma_optimizer is not None, 'You should set an optimizer'

        gamma_grad = np.sum(output_error * self.x_centered / self.x_std, axis=self.axis, keepdims=True)
        batch_size = output_error.shape[0]

        # following lines are got from the original paper, they are the same as final version for input error
        #
        # dldx = output_error * self.gamma
        # dldsigma2 = np.sum(dldx * self.x_centered * (-1/2) * (self.x_std ** -3), axis=0)
        # dldmu = np.sum((dldx * (-1/self.x_std)), axis=0) + dldsigma2 * np.sum(-2*self.x_centered, axis=0) / batch_size
        # input_error = dldx / self.x_std + dldsigma2 * 2 * self.x_centered / batch_size + dldmu / batch_size

        input_error = (1 / batch_size) * self.gamma / self.x_std * \
                      (batch_size * output_error - np.sum(output_error, axis=self.axis, keepdims=True) -
                       self.x_centered / self.x_std ** 2 * np.sum(output_error * self.x_centered, axis=self.axis,
                                                                  keepdims=True))

        self.gamma = self.gamma_optimizer.step(gamma_grad)
        if self.affine:
            beta_grad = np.sum(output_error, axis=self.axis, keepdims=True)
            self.beta = self.beta_optimizer.step(beta_grad)
        return input_error


class Conv2D(BaseLayer):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int = 0) -> None:
        super().__init__()
        self.in_channels = in_channels  # number of input channels
        self.out_channels = out_channels  # number of output channels
        self.kernel_size = kernel_size  # kernel size, int
        self.stride = stride  # stride, int
        self.padding = padding  # padding, int, only zeros padding

        limit = 1 / np.sqrt(self.kernel_size ** 2)
        self.kernel = np.random.uniform(
            -limit,
            limit,
            size=(self.out_channels, self.in_channels, self.kernel_size, self.kernel_size)
        )

        self.bias = np.zeros((self.out_channels, 1))
        self.kernel_optimizer = None
        self.bias_optimizer = None

        self.input_tensor = None
        self.batch_size = None
        self.reformatted_input = None
        self.reformatted_kernel = None

        self.channel_steps = []

    def set_kernel(self, kernel: np.array) -> None:
        """
        For setting the kernel manually
        """
        assert len(kernel.shape) == 4
        self.kernel = kernel

    def set_optimizer(self, optimizer) -> None:
        self.kernel_optimizer = copy.copy(optimizer)
        self.bias_optimizer = copy.copy(optimizer)

        self.kernel_optimizer.set_weight(self.kernel)
        self.bias_optimizer.set_weight(self.bias)

    def _reformat_kernel(self) -> np.array:
        """
        Reformat the kernel to perform convolution as matrix multiplication
        """
        assert len(self.kernel.shape) == 4
        return self.kernel.reshape((self.out_channels, -1))

    def _reformat_input(self, input_tensor: np.array) -> np.array:
        """
        Reformat the batch of input images to perform convolution as matrix multiplication
        """
        result = []
        for image in input_tensor:
            result.append(self._reformat_image(image))

        return np.hstack(result)

    def _reformat_image(self, image: np.array) -> np.array:
        """
        Reformatting each image in the batch
        """
        result = []
        for channel in image:
            result.append(self._reformat_channel(channel))

        return np.vstack(result)

    def _reformat_channel(self, channel: np.array) -> np.array:
        """
        Reformat each channel in the image with addinction of zeros as padding
        """
        input_map = self._add_padding(channel)

        self.padded_width_in, self.padded_height_in = input_map.shape

        width_in, height_in = channel.shape

        self.width_out = (width_in - self.kernel_size + 2 * self.padding) // self.stride + 1
        self.height_out = (height_in - self.kernel_size + 2 * self.padding) // self.stride + 1

        output_map = []

        write_steps = True
        if self.channel_steps:
            write_steps = False

        for i in range(self.width_out):
            for j in range(self.height_out):
                horizontal_slice_from = i * self.stride
                horizontal_slice_to = self.kernel_size + i * self.stride

                vertical_slice_from = j * self.stride
                vertical_slice_to = self.kernel_size + j * self.stride

                if write_steps:
                    self.channel_steps.append(
                        (horizontal_slice_from, horizontal_slice_to, vertical_slice_from, vertical_slice_to)
                    )

                subsample = input_map[horizontal_slice_from:horizontal_slice_to, vertical_slice_from:vertical_slice_to]
                output_map.append(subsample.reshape(-1, 1))

        return np.hstack(output_map)

    def _add_padding(self, input_tensor: np.array) -> np.array:
        new_width = input_tensor.shape[0] + self.padding * 2
        new_height = input_tensor.shape[1] + self.padding * 2
        padded_map = np.zeros((new_width, new_height))

        padded_map[self.padding:new_width - self.padding, self.padding:new_height - self.padding] = input_tensor
        return padded_map

    def _reformat_forward(self, forward_output: np.array) -> np.array:
        result = []
        for image in forward_output:
            result.append(image.reshape(self.batch_size, self.height_out, self.width_out))
        return np.hstack(result).reshape((self.batch_size, self.out_channels, self.height_out, self.width_out))

    @staticmethod
    def _reformat_output(output_error: np.array) -> np.array:
        images_result = []
        for image in output_error:
            channel_result = []
            for channel in image:
                channel_result.append(channel.ravel())
            images_result.append(np.vstack(channel_result))
        return np.hstack(images_result)

    def _format_image_back(self, image: np.array) -> np.array:
        zeros = np.zeros((self.in_channels, self.padded_width_in, self.padded_height_in))
        elements_in_kernel = self.kernel_size ** 2
        for i in range(self.in_channels):
            for j, line in enumerate(image.T):
                reshaped_conv = line[i * elements_in_kernel:(i + 1) * elements_in_kernel].reshape(self.kernel_size,
                                                                                                  self.kernel_size)

                horizontal_slice_from, horizontal_slice_to, vertical_slice_from, vertical_slice_to = self.channel_steps[
                    j]
                zeros[i][horizontal_slice_from:horizontal_slice_to,
                vertical_slice_from:vertical_slice_to] = reshaped_conv
        return zeros

    def _cut_padding(self, image: np.array) -> np.array:
        result = []
        for channel in image:
            channel_no_pad = channel[self.padding:self.padded_width_in - self.padding,
                             self.padding:self.padded_height_in - self.padding]
            result.append(channel_no_pad[np.newaxis, :])
        return np.vstack(result)

    def _reformat_input_error(self, input_error: np.array) -> np.array:
        size_for_one_image = int(len(input_error.T) / self.batch_size)
        result = []
        for i in range(self.batch_size):
            image = input_error[:, i * size_for_one_image:(i + 1) * size_for_one_image]
            image = self._format_image_back(image)

            if self.padding != 0:
                image = self._cut_padding(image)

            result.append(image[np.newaxis, :])
        return np.vstack(result)

    def forward(self, input_tensor: np.array, grad: bool = False) -> np.array:
        assert len(input_tensor.shape) == 4
        self.input_tensor = input_tensor
        self.batch_size = self.input_tensor.shape[0]

        self.reformatted_input = self._reformat_input(input_tensor)
        self.reformatted_kernel = self._reformat_kernel()

        result = self.reformatted_kernel.dot(self.reformatted_input) + self.bias
        return self._reformat_forward(result)

    def backward(self, output_error: np.array) -> np.array:
        assert self.kernel_optimizer is not None and self.bias_optimizer is not None, 'You should set an optimizer'
        output_error = self._reformat_output(output_error)
        w_grad = output_error.dot(self.reformatted_input.T).reshape(self.kernel.shape)
        b_grad = output_error.dot(np.ones((output_error.shape[1], 1)))

        self.kernel = self.kernel_optimizer.step(w_grad)
        self.bias = self.bias_optimizer.step(b_grad)

        input_error = self.reformatted_kernel.T.dot(output_error)
        input_error = self._reformat_input_error(input_error)
        return input_error


class MaxPool2D(BaseLayer):
    def __init__(self, kernel_size: int, stride: int, padding: int = 0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.input_tensor = None
        self.channel_steps = []

    def _reformat_input(self, input_tensor: np.array) -> np.array:
        result = []
        grad_result = []
        for image in input_tensor:
            forward_result, grad = self._reformat_image(image)
            result.append(forward_result)
            grad_result.append(grad)

        self.grad = grad_result
        return np.vstack(result)

    def _reformat_image(self, image: np.array) -> np.array:
        result = []
        grad_result = []
        for channel in image:
            forward_result, grad = self._reformat_channel(channel)
            result.append(forward_result)
            grad_result.append(grad)

        return np.vstack(result), grad_result

    def _reformat_channel(self, channel: np.array) -> Tuple[np.array, np.array]:
        input_map = self._add_padding(channel)

        width_in, height_in = channel.shape

        self.width_out = (width_in - self.kernel_size + 2 * self.padding) // self.stride + 1
        self.height_out = (height_in - self.kernel_size + 2 * self.padding) // self.stride + 1

        output_map = np.zeros((self.height_out, self.width_out))

        grad = []

        write_steps = True
        if self.channel_steps:
            write_steps = False

        for i in range(self.width_out):
            for j in range(self.height_out):
                horizontal_slice_from = i * self.stride
                horizontal_slice_to = self.kernel_size + i * self.stride

                vertical_slice_from = j * self.stride
                vertical_slice_to = self.kernel_size + j * self.stride

                if write_steps:
                    self.channel_steps.append(
                        (horizontal_slice_from, horizontal_slice_to, vertical_slice_from, vertical_slice_to)
                    )

                subsample = input_map[horizontal_slice_from:horizontal_slice_to, vertical_slice_from:vertical_slice_to]
                output_map[i][j] = np.max(subsample)

                grad.append(np.argmax(subsample))

        return output_map, grad

    def _add_padding(self, input_tensor: np.array) -> np.array:
        new_width = input_tensor.shape[0] + self.padding * 2
        new_height = input_tensor.shape[1] + self.padding * 2
        padded_map = np.zeros((new_width, new_height))

        padded_map[self.padding:new_width - self.padding, self.padding:new_height - self.padding] = input_tensor
        return padded_map

    def _rebuild_output_error(self, output_error: np.array) -> np.array:
        zeros = np.zeros(self.input_tensor.shape)
        for image_num, image in enumerate(output_error):
            for channel_num, channel in enumerate(image):
                for i in range(self.width_out * self.height_out):
                    horizontal_slice_from, horizontal_slice_to, vertical_slice_from, vertical_slice_to \
                        = self.channel_steps[i]

                    slice_ = zeros[image_num][channel_num][horizontal_slice_from:horizontal_slice_to,
                             vertical_slice_from:vertical_slice_to]
                    cur_max = self.grad[image_num][channel_num][i]

                    flatten_channel = channel.flatten()
                    flatten_zeros = slice_.flatten()

                    flatten_zeros[cur_max] = flatten_channel[i]

                    slice_ = flatten_zeros.reshape(self.kernel_size, self.kernel_size)
                    zeros[image_num][channel_num][horizontal_slice_from:horizontal_slice_to,
                    vertical_slice_from:vertical_slice_to] = slice_
        return zeros

    def forward(self, input_tensor: np.array, grad: bool = False) -> np.array:
        assert len(input_tensor.shape) == 4
        self.input_tensor = input_tensor

        result = self._reformat_input(input_tensor)
        return result.reshape((input_tensor.shape[0], input_tensor.shape[1], self.height_out, self.width_out))

    def backward(self, output_error: np.array) -> np.array:
        return self._rebuild_output_error(output_error)


class Ravel(BaseLayer):
    def __init__(self) -> None:
        super().__init__()
        self.input = None

    def forward(self, input_tensor: np.array, grad: bool = False) -> np.array:
        length = len(input_tensor)
        self.input = input_tensor
        return input_tensor.reshape(length, -1)

    def backward(self, output_error: np.array) -> np.array:
        return output_error.reshape(self.input.shape)


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


class Embedding(BaseLayer):
    def __init__(self, n_input, emb_dim, pad_idx=None):
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
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        scale = np.sqrt(1 / (in_channels * kernel_size))
        self.kernel = np.random.uniform(-scale, scale, size=(out_channels, in_channels, kernel_size))
        self.bias = np.random.uniform(-scale, scale, size=(out_channels))

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
