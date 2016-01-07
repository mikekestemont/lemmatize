#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import numpy as np

import utils
import model
import evaluation

# hyperparams:
model_name = 'xxx'
nb_filters = 100
filter_length = 3
nb_dense_dims = 300
include_conv = False
nb_epochs = 100
batch_size = 100
nb_encoding_layers = 3

def main():
    # load and preprocess data
    #tokens, lemmas = utils.load_data('../data/IT.txt', nb_instances=1000)
    tokens, lemmas = utils.load_data('../data/IT.txt')
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

    train_token_set = set(train_tokens)

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

    print('::: Dev baseline scores :::')
    dev_baseline_lems = evaluation.apply_baseline(train_toks=train_tokens,
                            train_lems=train_lemmas,
                            test_toks=dev_tokens,
                            test_lems=dev_lemmas)
    all_acc, kno_acc, unk_acc = evaluation.accuracies(gold=dev_lemmas,
                                             silver=dev_baseline_lems,
                                             test_tokens=dev_tokens,
                                             known_tokens=train_token_set)
    print('+\tall acc:', all_acc)
    print('+\tkno acc:', kno_acc)
    print('+\tunk acc:', unk_acc)

    print('::: Test baseline scores :::')
    test_baseline_lems = evaluation.apply_baseline(train_toks=train_tokens,
                            train_lems=train_lemmas,
                            test_toks=test_tokens,
                            test_lems=test_lemmas)
    all_acc, kno_acc, unk_acc = evaluation.accuracies(gold=test_lemmas,
                                             silver=test_baseline_lems,
                                             test_tokens=test_tokens,
                                             known_tokens=train_token_set)
    print('+\tall acc:', all_acc)
    print('+\tkno acc:', kno_acc)
    print('+\tunk acc:', unk_acc)

    m = model.build_model(max_in_len=max_in_len,
                          in_char_vector_dict=in_char_vector_dict,
                          nb_filters=nb_filters,
                          filter_length=filter_length,
                          nb_encoding_layers=nb_encoding_layers,
                          nb_dense_dims=nb_dense_dims,
                          max_out_len=max_out_len,
                          out_char_vector_dict=out_char_vector_dict,
                          include_conv=include_conv)

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
        
        # convert predictions to actual strings:
        pred_lemmas = evaluation.convert_to_lemmas(predictions=predictions['out'],
                        out_char_idx=out_char_idx)

        """
        # check a random selection
        for token, pred_lem in zip(dev_tokens[3000:4000], pred_lemmas[3000:4000]):
            if token not in train_token_set:
                print(token, '>', pred_lem, '(UNK)')
            else:
                print(token, '>', pred_lem)
        """


        print('::: Dev scores :::')
        all_acc, kno_acc, unk_acc = evaluation.accuracies(gold=dev_lemmas,
                                             silver=pred_lemmas,
                                             test_tokens=dev_tokens,
                                             known_tokens=train_token_set)
        print('+\tall acc:', all_acc)
        print('+\tkno acc:', kno_acc)
        print('+\tunk acc:', unk_acc)

        print('::: Test predictions :::')
        d = {'in': test_in_X}
        predictions = m.predict(data=d, batch_size=batch_size)

        # convert predictions to actual strings:
        pred_lemmas = evaluation.convert_to_lemmas(predictions=predictions['out'],
                                out_char_idx=out_char_idx)

        """
        # check a random selection
        for token, pred_lem in zip(test_tokens[3000:4000], pred_lemmas[3000:4000]):
            if token not in train_token_set:
                try:
                    print(token, '>', pred_lem[:pred_lem.index('$')], '(UNK)')
                except ValueError:
                    print(token, '>', pred_lem, '(UNK)')
            else:
                try:
                    print(token, '>', pred_lem[:pred_lem.index('$')])
                except ValueError:
                    print(token, '>', pred_lem)
        """

        print('::: Test scores :::')
        all_acc, kno_acc, unk_acc = evaluation.accuracies(gold=test_lemmas,
                                                 silver=pred_lemmas,
                                                 test_tokens=test_tokens,
                                                 known_tokens=train_token_set)
        print('+\tall acc:', all_acc)
        print('+\tkno acc:', kno_acc)
        print('+\tunk acc:', unk_acc)


if __name__ == '__main__':
    main()