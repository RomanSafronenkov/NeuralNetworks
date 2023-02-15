import copy

import numpy as np

from neural_networks.nn.activation_functions import softmax


def to_matrix(data, token_to_id, max_len=None, dtype='int32', batch_first=True):
    """Casts a list of names into rnn-digestable matrix"""

    max_len = max_len or max(map(len, data))
    data_ix = np.zeros([len(data), max_len], dtype) + token_to_id[' ']

    for i in range(len(data)):
        line_ix = [token_to_id[c] for c in data[i]]
        data_ix[i, :len(line_ix)] = line_ix

    if not batch_first:  # convert [batch, time] into [time, batch]
        data_ix = np.transpose(data_ix)

    return data_ix


def generate_sample(char_rnn, token_to_id, tokens, max_length, seed_phrase=' ', temperature=1.0):
    phrase = copy.copy(seed_phrase)

    for t in range(len(seed_phrase) - 1, max_length - len(seed_phrase)):
        x_sequence = to_matrix([phrase], token_to_id, max_len=max_length)
        encoded_data = np.zeros(shape=(1, max_length, len(token_to_id)))

        for text_i, text in enumerate(encoded_data):
            for letter_i, letter in enumerate(text):
                encoded_data[text_i, letter_i, x_sequence[text_i, letter_i]] = 1

        pred = char_rnn(encoded_data)
        probs = softmax(pred[:, t] / temperature).ravel()
        next_ix = np.random.choice(len(tokens), p=probs)
        phrase += tokens[next_ix]
    return phrase
