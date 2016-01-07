#!/usr/bin/env python
# -*- coding: utf-8 -*-

from collections import Counter
from operator import itemgetter
import numpy as np
import editdist

def convert_to_lemmas(predictions, out_char_idx):
    """
    For each prediction, convert 2D char matrix
    with probabilities to actual lemmas, using
    character index for the output strings.
    """
    pred_lemmas = []
    for pred in predictions:
        pred_lem = ''
        for positions in pred:
            top_idx = np.argmax(positions) # winning position
            c = out_char_idx[top_idx] # look up corresponding char
            pred_lem += c # add character
        pred_lemmas.append(pred_lem)
    return pred_lemmas

def accuracies(gold, silver, test_tokens, known_tokens):
    """
    Calculate accuracies for all, known and unknown tokens.
    Uses index of items seen during training.
    """
    kno_corr, unk_corr = 0.0, 0.0
    nb_kno, nb_unk = 0.0, 0.0

    for gold_lem, silver_lem, tok in zip(gold, silver, test_tokens):
        # rm trailing $:
        try:
            silver_lem = silver_lem[:silver_lem.index('$')]
        except ValueError:
            pass

        if tok in known_tokens:
            nb_kno += 1
            if gold_lem == silver_lem:
                kno_corr += 1
        else:
            nb_unk += 1
            if gold_lem == silver_lem:
                unk_corr += 1

    all_acc = (kno_corr + unk_corr) / (nb_kno + nb_unk)
    kno_acc = kno_corr / nb_kno

    # account for situation with no unknowns:
    unk_acc = 1.0
    if nb_unk > 0:
        unk_acc = unk_corr / nb_unk

    return all_acc, kno_acc, unk_acc

def apply_baseline(train_toks, train_lems,
                   test_toks, test_lems):
    print("Calculating baseline")
    train_dict = {}

    for tok, lem in zip(train_toks, train_lems):
        if tok not in train_dict:
            train_dict[tok] = {}
        if lem not in train_dict[tok]:
            train_dict[tok][lem] = 0
        train_dict[tok][lem] += 1

    silver_lemmas = []
    for test_tok, test_lem in zip(test_toks, test_lems):
        # shortcut:
        if test_tok in train_dict:
            k = test_tok
        else:
            candidates = train_dict.keys()
            distances = [(editdist.distance(test_tok, c), c) for c in candidates]
            k = min(distances, key=itemgetter(0))[1]
        silver_lem = max(train_dict[k].iteritems(), key=itemgetter(1))[0]
        silver_lemmas.append(silver_lem)

    return silver_lemmas

    