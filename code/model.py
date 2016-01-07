#!/usr/bin/env python
# -*- coding: utf-8 -*-

from keras.models import Graph
from keras.layers.recurrent import LSTM, GRU
from keras.layers.core import Dense, TimeDistributedDense, Dropout, Activation, RepeatVector
from keras.layers.convolutional import Convolution1D
from keras.optimizers import Adam

def build_model(max_in_len, in_char_vector_dict,
                nb_filters, filter_length,
                nb_encoding_layers, nb_dense_dims,
                max_out_len, out_char_vector_dict,
                include_conv):
    m = Graph()

    # add input layer:
    m.add_input(name='in', input_shape=(max_in_len, len(in_char_vector_dict)))

    if include_conv:
        # add convolutional layer:
        m.add_node(Convolution1D(input_dim=len(in_char_vector_dict),
                                        nb_filter=nb_filters,
                                        filter_length=filter_length,
                                        activation='tanh',
                                        border_mode="valid",
                                        subsample_length=1), 
                               name='conv', input='in')

    # add recurrent layers:
    return_seqs = True
    for i in range(nb_encoding_layers):
        if i == 0:
            if include_conv:
                input_name = 'conv'
            else:
                input_name = 'in'
        else:
            input_name = 'encoder_dropout_'+str(i)

        if i == nb_encoding_layers-1:
            output_name = 'final_encoder'
            return_seqs = False
        else:
            output_name = 'encoder_dropout_'+str(i+1)

        m.add_node(LSTM(output_dim=nb_dense_dims,
                        return_sequences=return_seqs,
                        activation='tanh'),
                        name='encoder_'+str(i+1),
                        input=input_name)
        m.add_node(Dropout(0.05),
                    name=output_name,
                    input='encoder_'+str(i+1))

    # repeat final input
    m.add_node(RepeatVector(max_out_len),
          name='encoder_repeat',
          input='final_encoder')

    # 2nd, single recurrent layer to generate output sequence:
    m.add_node(LSTM(input_dim=nb_dense_dims,
                    output_dim=nb_dense_dims,
                    activation='tanh',
                    return_sequences=True),
           input='encoder_repeat',
           name='decoder')
    m.add_node(Dropout(0.05),
                name='decoder_dropout',
                input='decoder')

    m.add_node(TimeDistributedDense(output_dim=len(out_char_vector_dict)),
                name='label_dense',
                input='decoder_dropout')
    m.add_node(Dropout(0.05),
                name='label_dropout',
                input='label_dense')
    m.add_node(Activation('softmax'),
                name='label_softmax',
                input='label_dropout')

    m.add_output(name='out', input='label_softmax')

    adam = Adam(epsilon=1e-8, clipnorm=5)

    m.compile(optimizer=adam,
              loss={'out':'categorical_crossentropy'})

    return m