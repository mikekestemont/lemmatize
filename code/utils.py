#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import logging
import sys

logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)

import numpy as np
from sklearn.cross_validation import train_test_split

def load_data(filepath, nb_instances=sys.maxint):
    """ Returns a tuple of tokens and lemmas
        from the entire data file.
    """

    tokens, lemmas = [], []

    for line in open(filepath):

        line = line.strip()
        if line:
            try:
                comps = line.split()
                token, lemma = comps[1:3]
                tokens.append(token.strip().lower())
                lemmas.append(lemma.strip().lower())
                #print(token, lemma)

                # cutoff for development:
                nb_instances -= 1
                if nb_instances <= 0:
                    break
            except ValueError:
                pass

    # sanity check
    assert len(tokens) == len(lemmas)

    logging.info('Returning %d token-lemma pairs' % (len(tokens)))

    return tokens, lemmas

def max_len(items):
    """ Return length of longest item in items"""
    return len(max(items, key=len))

def train_dev_test_split(tokens, lemmas, test_size=.3, dev_size=.15):
    """
    Return random train-dev-test split.
    No sequential information included.
    """

    random_state = len(tokens)

    # split off test from rest:
    train_tokens, test_tokens, train_lemmas, test_lemmas = \
        train_test_split(tokens, lemmas,
                         test_size=test_size,
                         random_state=random_state)

    # sample dev from all train:
    train_tokens, dev_tokens, train_lemmas, dev_lemmas = \
        train_test_split(train_tokens, train_lemmas,
                         test_size=dev_size,
                         random_state=random_state)
    logging.info('=== Splitting stats ===')
    logging.info('\t- Train: %d token-lemma pairs' % (len(train_tokens)))
    logging.info('\t- Dev: %d token-lemma pairs' % (len(dev_tokens)))
    logging.info('\t- Test: %d token-lemma pairs' % (len(test_tokens)))
    
    return train_tokens, train_lemmas,\
           dev_tokens, dev_lemmas,\
           test_tokens, test_lemmas

def get_char_vector_dict(tokens):
    char_vocab = tuple({ch for tok in tokens+['$'] for ch in tok+" "})
    char_vector_dict, char_idx = {}, {}
    filler = np.zeros(len(char_vocab), dtype='float32')

    for idx, char in enumerate(char_vocab):
        ph = filler.copy()
        ph[idx] = 1
        char_vector_dict[char] = ph
        char_idx[idx] = char

    return char_vector_dict, char_idx

def vectorize_in_sequences(items, in_char_vector_dict,
                           max_in_len=15):
    X_in = []
    for item in items:
        x = vectorize_in_charseq(seq=item,
                                 char_vector_dict=in_char_vector_dict,
                                 std_seq_len=max_in_len)
        X_in.append(x)

    X_in = np.asarray(X_in, dtype='float32')
    return X_in

def vectorize_out_sequences(items, out_char_vector_dict,
                           max_out_len=15):
    X_out = []
    for item in items:
        x = vectorize_out_charseq(seq=item,
                                 char_vector_dict=out_char_vector_dict,
                                 std_seq_len=max_out_len)
        X_out.append(x)

    X_out = np.asarray(X_out, dtype='float32')
    return X_out

def vectorize_in_charseq(seq, char_vector_dict, std_seq_len):
    # cut, if needed:
    seq = seq[:std_seq_len]
    seq = seq[::-1] # reverse order!

    # pad, if needed:
    while len(seq) < std_seq_len:
        seq = '$'+seq

    seq_X = []
    filler = np.zeros(len(char_vector_dict), dtype='float32')
    for char in seq:
        try:
            seq_X.append(char_vector_dict[char])
        except KeyError:
            seq_X.append(filler)
    return np.vstack(seq_X)

def vectorize_out_charseq(seq, char_vector_dict, std_seq_len):
    # cut, if needed:
    seq = seq[:std_seq_len]
    # pad, if needed:
    while len(seq) < std_seq_len:
        seq += '$'
    seq_X = []
    filler = np.zeros(len(char_vector_dict), dtype='float32')
    for char in seq:
        try:
            seq_X.append(char_vector_dict[char])
        except KeyError:
            seq_X.append(filler)
    return np.vstack(seq_X)
