import copy

import numpy as np

from neural_networks.nn.activation_functions import tanh, tanh_derivative
from neural_networks.nn.layers import RNNCell, RNNUnit, LSTM
from neural_networks.nn.loss_functions import cross_entropy_loss, cross_entropy_loss_derivative
from neural_networks.nn.optimizers import ADAM
from neural_networks.utils.text import to_matrix, generate_sample

np.random.seed(42)


def train_rnn_for_names(rnn_model):
    rnn_model = copy.deepcopy(rnn_model)
    batch_size = 32

    for i in range(1000):
        batch_ix = to_matrix(np.random.choice(names, size=batch_size), token_to_id, max_len=MAX_LENGTH)

        encoded_data = np.zeros(shape=(batch_size, MAX_LENGTH, len(token_to_id)))

        for text_i, text in enumerate(encoded_data):
            for letter_i, letter in enumerate(text):
                encoded_data[text_i, letter_i, batch_ix[text_i, letter_i]] = 1

        pred = rnn_model(encoded_data)

        loss = 0
        for t in range(batch_ix.shape[1] - 1):
            loss += cross_entropy_loss(batch_ix[:, t + 1].reshape(-1, 1), pred[:, t, :])

        errors = np.zeros(shape=(batch_size, MAX_LENGTH - 1, len(token_to_id)))
        for t in range(errors.shape[1] - 1):
            errors[:, t, :] = cross_entropy_loss_derivative(batch_ix[:, t + 1].reshape(-1, 1), pred[:, t, :])

        rnn_model.backward(errors)
        if (i + 1) % 100 == 0:
            print(f'epoch: {i + 1}. Current loss: {loss / batch_size}')
    return rnn_model


with open('data/russian_names.txt') as input_file:
    names = input_file.read()[:-1].split('\n')
    names = [' ' + line for line in names]

tokens = list(set(''.join(names)))

num_tokens = len(tokens)
print('num_tokens = ', num_tokens)

token_to_id = {token: idx for idx, token in enumerate(tokens)}

print('\n'.join(names[::2000]))
print(to_matrix(names[::2000], token_to_id))

MAX_LENGTH = max(map(len, names))
print(f'MAX_LENGTH={MAX_LENGTH}')

rnncell = RNNCell(
    n_input=len(token_to_id),
    n_hidden=64,
    hidden_activation=tanh,
    hidden_activation_derivative=tanh_derivative,
    bptt_trunc=15
)
rnncell.set_optimizer(ADAM(learning_rate=0.001))
print('FITTING RNNCELL...')
rnncell = train_rnn_for_names(rnncell)

rnnunit = RNNUnit(
    n_input=len(token_to_id),
    n_hidden=64,
    bptt_trunc=20
)
rnnunit.set_optimizer(ADAM(learning_rate=0.001))
print('FITTING RNNUNIT...')
rnnunit = train_rnn_for_names(rnnunit)

lstm_unit = LSTM(
    n_input=len(token_to_id),
    n_hidden=64,
    n_output=len(token_to_id),
    bptt_trunc=20
)
lstm_unit.set_optimizer(ADAM(learning_rate=0.001))
print('FITTING LSTM...')
lstm_unit = train_rnn_for_names(lstm_unit)

print('\nRNNCELL generating...')
for _ in range(10):
    print(generate_sample(rnncell, token_to_id=token_to_id, tokens=tokens, max_length=MAX_LENGTH, seed_phrase=' ',
                          temperature=0.4))

print('\nRNNUNIT generating...')
for _ in range(10):
    print(generate_sample(rnnunit, token_to_id=token_to_id, tokens=tokens, max_length=MAX_LENGTH, seed_phrase=' ',
                          temperature=0.4))

print('\nLSTM generating...')
for _ in range(10):
    print(generate_sample(lstm_unit, token_to_id=token_to_id, tokens=tokens, max_length=MAX_LENGTH, seed_phrase=' ',
                          temperature=0.4))
