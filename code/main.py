#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import utils
import model

# hyperparams:
nb_filters = 100
filter_length = 3
nb_dense_dims = 300
nb_epochs = 200
batch_size = 50
nb_encoding_layers = 3

def main():
    # load and preprocess data
    tokens, lemmas = utils.load_data('IT.txt')
    train_tokens, train_lemmas,\
    dev_tokens, dev_lemmas,\
    test_tokens, test_lemmas = utils.train_dev_test_split(tokens, lemmas)

    # fit vectorizers etc.
    in_char_vector_dict, in_char_idx = \
        utils.get_char_vector_dict(train_tokens)
    out_char_vector_dict, out_char_idx = \
        utils.get_char_vector_dict(train_lemmas)

    max_in_len = utils.max_len(train_tokens)
    max_out_len = utils.max_len(train_lemmas)

    # vectorize train, then dev and test:
    train_in_X = utils.vectorize_in_sequences(\
                    train_tokens,
                    in_char_vector_dict,
                    max_in_len)
    train_out_X = utils.vectorize_out_sequences(\
                    train_lemmas,
                    out_char_vector_dict,
                    max_out_len)
    dev_in_X = utils.vectorize_in_sequences(\
                    dev_tokens,
                    in_char_vector_dict,
                    max_in_len)
    dev_out_X = utils.vectorize_out_sequences(\
                    dev_lemmas,
                    out_char_vector_dict,
                    max_out_len)
    test_in_X = utils.vectorize_in_sequences(\
                    test_tokens,
                    in_char_vector_dict,
                    max_in_len)
    test_out_X = utils.vectorize_out_sequences(\
                    test_lemmas,
                    out_char_vector_dict,
                    max_out_len)

    m = model.build_model(max_in_len=max_in_len,
                          in_char_vector_dict=in_char_vector_dict,
                          nb_filters=nb_filters,
                          filter_length=filter_length,
                          nb_encoding_layers=nb_encoding_layers,
                          nb_dense_dims=nb_dense_dims,
                          max_out_len=max_out_len,
                          out_char_vector_dict=out_char_vector_dict)

    # fit:
    for e in range(nb_epochs):
        print("-> epoch ", e+1, "...")

        # fit on train:
        d = {'in': train_in_X, 'out': train_out_X}
        m.fit(data=d,
              nb_epoch = 1,
              batch_size = batch_size)

        # get loss on train:
        train_loss = m.evaluate(data=d, batch_size=batch_size)
        print("\t - loss:\t{:.3}".format(train_loss))

        # get dev predictions:
        d = {'in': dev_in_X}
        predictions = m.predict(data=d, batch_size=batch_size)
        
        for token, prediction in zip(dev_tokens[300:400], predictions['out'][300:400]):
            print(token)
            for positions in prediction:
                top_idx = np.argmax(positions)
                print(out_char_idx[top_idx], end='')
            print('\n')


if __name__ == '__main__':
    main()