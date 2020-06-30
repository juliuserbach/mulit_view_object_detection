"""
Mask R-CNN
The main Mask R-CNN model implementation.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
"""

import os
import random
import datetime
import re
import math
import logging
from collections import OrderedDict
import multiprocessing
import numpy as np
import tensorflow as tf
import keras
import keras.backend as K
import keras.layers as KL
import keras.engine as KE
import keras.models as KM
import tensorflow.contrib.slim as slim
from mrcnn.recurrent import ConvRNN3D

from keras.backend.tensorflow_backend import set_session
config_tf = tf.ConfigProto()
config_tf.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config_tf.log_device_placement = True  # to log device placement (on which device the operation ran)
                                    # (nothing gets printed in Jupyter, only if you run it standalone)
sess = tf.Session(config=config_tf)
set_session(sess) 

from mrcnn import utils

# Requires TensorFlow 1.3+ and Keras 2.0.8+.
from distutils.version import LooseVersion
assert LooseVersion(tf.__version__) >= LooseVersion("1.3")
assert LooseVersion(keras.__version__) >= LooseVersion('2.0.8')

reused_lay = {}

############################################################
#  Utility Functions
############################################################

def log(text, array=None):
    """Prints a text message. And, optionally, if a Numpy array is provided it
    prints it's shape, min, and max values.
    """
    if array is not None:
        text = text.ljust(25)
        text += ("shape: {:20}  ".format(str(array.shape)))
        if array.size:
            text += ("min: {:10.5f}  max: {:10.5f}".format(array.min(),array.max()))
        else:
            text += ("min: {:10}  max: {:10}".format("",""))
        text += "  {}".format(array.dtype)
    print(text)


class BatchNorm(KL.BatchNormalization):
    """Extends the Keras BatchNormalization class to allow a central place
    to make changes if needed.

    Batch normalization has a negative effect on training if batches are small
    so this layer is often frozen (via setting in Config class) and functions
    as linear layer.
    """
    def call(self, inputs, training=None):
        """
        Note about training values:
            None: Train BN layers. This is the normal mode
            False: Freeze BN layers. Good when batch size is small
            True: (don't use). Set layer in training mode even when making inferences
        """
        return super(self.__class__, self).call(inputs, training=training)


def compute_backbone_shapes(config, image_shape):
    """Computes the width and height of each stage of the backbone network.

    Returns:
        [N, (height, width)]. Where N is the number of stages
    """
    if callable(config.BACKBONE):
        return config.COMPUTE_BACKBONE_SHAPE(image_shape)

    # Currently supports ResNet only
    assert config.BACKBONE in ["resnet50", "resnet101"]
    return np.array(
        [[int(math.ceil(image_shape[0] / stride)),
            int(math.ceil(image_shape[1] / stride))]
            for stride in config.BACKBONE_STRIDES])

############################################################
#  ConvLSTM
############################################################




# class ConvLSTMCell(tf.contrib.rnn.RNNCell):
#     """A LSTM cell with convolutions instead of multiplications.
#     Reference:
#       Xingjian, S. H. I., et al. "Convolutional LSTM network: 
#       A machine learning approach for precipitation nowcasting.
#       " Advances in Neural Information Processing Systems. 2015.
#     """

#     def __init__(self,
#                  shape,
#                  input_dim,
#                  filters,
#                  kernel,
#                  initializer=None,
#                  forget_bias=1.0,
#                  activation=tf.tanh,
#                  normalize=True, data_format='channels_last'):
#         self.data_format = data_format
#         self._kernel = kernel
#         self._filters = filters
#         self.filters = filters
#         self._initializer = initializer
#         self._forget_bias = forget_bias
#         self._activation = activation
#         self._size = tf.TensorShape(shape + [self._filters])
#         self._normalize = normalize
#         self._feature_axis = self._size.ndims
#         if data_format=='channels_first':
#             channel_axis = 1
#         else:
#             channel_axis = -1
#         input_dim = shape[channel_axis]
#         kernel_shape = self._kernel + [input_dim, self.filters * 4]
#         self.kernel_shape = kernel_shape

#     @property
#     def state_size(self):
#         return tf.contrib.rnn.LSTMStateTuple(self._size, self._size)

#     @property
#     def output_size(self):
#         return self._size

#     def __call__(self, x, h, **kwargs):
#         scope = None
#         with tf.variable_scope(scope or self.__class__.__name__):
#             previous_memory, previous_output = h

#             channels = x.shape[-1].value
#             filters = self._filters
#             gates = 4 * filters if filters > 1 else 4
#             x = tf.concat([x, previous_output], axis=self._feature_axis)
#             n = channels + filters
#             m = gates
#             W = tf.get_variable(
#                 'kernel', self._kernel + [n, m], initializer=self._initializer)
#             y = tf.nn.convolution(x, W, 'SAME')
#             if not self._normalize:
#                 y += tf.get_variable(
#                     'bias', [m], initializer=tf.constant_initializer(0.0))
#             input_contribution, input_gate, forget_gate, output_gate = tf.split(
#                 y, 4, axis=self._feature_axis)

#             if self._normalize:
#                 input_contribution = tf.contrib.layers.layer_norm(
#                     input_contribution)
#                 input_gate = tf.contrib.layers.layer_norm(input_gate)
#                 forget_gate = tf.contrib.layers.layer_norm(forget_gate)
#                 output_gate = tf.contrib.layers.layer_norm(output_gate)

#             memory = (
#                 previous_memory * tf.sigmoid(forget_gate + self._forget_bias) +
#                 tf.sigmoid(input_gate) * self._activation(input_contribution))

#             if self._normalize:
#                 memory = tf.contrib.layers.layer_norm(memory)

#             output = self._activation(memory) * tf.sigmoid(output_gate)

#             return output, tf.contrib.rnn.LSTMStateTuple(memory, output)

class ConvLSTMCell(KL.Layer):
    
    
    def __init__(self, shape, kernel, filters, 
                 initializer=slim.initializers.xavier_initializer(),
                 data_format='channels_last', activation=tf.tanh,
                 normalize=False, forget_bias = 1., **kwargs):
        self._normalize = normalize
        self.kernel = kernel
        self.kernel_size = kernel
        self.filters = filters 
        self._initializer = initializer
        self._activation = activation
        self._forget_bias = forget_bias
        self._size = tf.TensorShape(shape + [self.filters])
        self._feature_axis = self._size.ndims
        self.data_format = data_format
        if self._normalize:
            self.layer_norm_input_contribution = KL.LayerNormalization()
            self.layer_norm_input_gate = KL.LayerNormalization()
            self.layer_norm_output_gate = KL.LayerNormalization()
            self.layer_norm_forget_gate = KL.LayerNormalization()
            self.layer_norm_memory = KL.LayerNormalization()
        super(ConvLSTMCell, self).__init__(**kwargs)
    
    
    @property
    def state_size(self):
        return tf.contrib.rnn.LSTMStateTuple(self._size, self._size)

    
    @property
    def output_size(self):
        return self._size
        
        
    def build(self, input_shape):
        print("input shape in build:  {}".format(input_shape))
        bs, h, w, d, ch = input_shape
        self.input_dim = ch
        filters = self.filters
        gates = 4 * filters if filters > 1 else 4
        n = ch + filters
        m = gates
        
        self.W = self.add_weight(name='weights_lstm3d',
                                 shape=self.kernel + [n, m], 
                                 initializer=self._initializer
                                 , trainable=True)

        self.bias = self.add_weight(name='bias_lstm3d',
                                    shape=[m], 
                                    initializer=tf.constant_initializer(0.0), 
                                    trainable=True)

        if self.data_format=='channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        input_dim = input_shape[channel_axis]
        kernel_shape = self.kernel + [input_dim, self.filters * 4]
        self.kernel_shape = kernel_shape
        
        
    def call(self, x, h, **kwargs):
        
        scope = None
        with tf.variable_scope(scope or self.__class__.__name__):
            previous_memory, previous_output = h
            
            print("shape of x: {}".format(x.get_shape().as_list()))
            
            channels = x.shape[-1].value
            filters = self.filters
            gates = 4 * filters if filters > 1 else 4
            x = tf.concat([x, previous_output], axis=self._feature_axis)
            n = channels + filters
            m = gates
            
            y = tf.nn.convolution(x, self.W, 'SAME')
            if not self._normalize:
                y += self.bias
            input_contribution, input_gate, forget_gate, output_gate = tf.split(
                y, 4, axis=self._feature_axis)

            if self._normalize:
                input_contribution = self.layer_norm_input_contribution(
                    input_contribution)
                input_gate = self.layer_norm_input_gate(input_gate)
                forget_gate = self.layer_norm_forget_gate(forget_gate)
                output_gate = self.layer_norm_output_gate(output_gate)

            memory = (
                previous_memory * tf.sigmoid(forget_gate + self._forget_bias) +
                tf.sigmoid(input_gate) * self._activation(input_contribution))

            if self._normalize:
                memory = self.layer_norm_memory(memory)

            output = self._activation(memory) * tf.sigmoid(output_gate)

            return output, tf.contrib.rnn.LSTMStateTuple(memory, output)

    
class ConvGRUCell(tf.contrib.rnn.RNNCell):
    """A GRU cell with convolutions instead of multiplications."""

    def __init__(self,
                 shape,
                 filters,
                 kernel,
                 initializer=None,
                 activation=tf.tanh,
                 normalize=True):
        self._filters = filters
        self._kernel = kernel
        self._initializer = initializer
        self._activation = activation
        self._size = tf.TensorShape(shape + [self._filters])
        self._normalize = normalize
        self._feature_axis = self._size.ndims

    @property
    def state_size(self):
        return self._size

    @property
    def output_size(self):
        return self._size

    def __call__(self, x, h, scope=None):
        with tf.variable_scope(scope or self.__class__.__name__):

            with tf.variable_scope('Gates'):
                channels = x.shape[-1].value
                inputs = tf.concat([x, h], axis=self._feature_axis)
                n = channels + self._filters
                m = 2 * self._filters if self._filters > 1 else 2
                W = tf.get_variable(
                    'kernel',
                    self._kernel + [n, m],
                    initializer=self._initializer)
                y = tf.nn.convolution(inputs, W, 'SAME')
                if self._normalize:
                    reset_gate, update_gate = tf.split(
                        y, 2, axis=self._feature_axis)
                    reset_gate = tf.contrib.layers.layer_norm(reset_gate)
                    update_gate = tf.contrib.layers.layer_norm(update_gate)
                else:
                    y += tf.get_variable(
                        'bias', [m], initializer=tf.constant_initializer(1.0))
                    reset_gate, update_gate = tf.split(
                        y, 2, axis=self._feature_axis)
                reset_gate, update_gate = tf.sigmoid(reset_gate), tf.sigmoid(
                    update_gate)

            with tf.variable_scope('Output'):
                inputs = tf.concat(
                    [x, reset_gate * h], axis=self._feature_axis)
                n = channels + self._filters
                m = self._filters
                W = tf.get_variable(
                    'kernel',
                    self._kernel + [n, m],
                    initializer=self._initializer)
                y = tf.nn.convolution(inputs, W, 'SAME')
                if self._normalize:
                    y = tf.contrib.layers.layer_norm(y)
                else:
                    y += tf.get_variable(
                        'bias', [m], initializer=tf.constant_initializer(0.0))
                y = self._activation(y)
                output = update_gate * h + (1 - update_gate) * y

            return output, output
        
def convgru(grid, kernel=[3, 3, 3], filters=32):
    bs, im_bs, h, w, d, ch = grid.get_shape().as_list()
    
    
    conv_gru = ConvGRUCell(
        shape=[h, w, d],
        initializer=slim.initializers.xavier_initializer(),
        kernel=kernel,
        filters=filters)
    seq_length = [im_bs for _ in range(bs)]
    outputs, states = tf.nn.dynamic_rnn(
        conv_gru,
        grid,
        parallel_iterations=64,
        sequence_length=seq_length,
        dtype=tf.float32,
        time_major=False)
    return outputs, states


def convlstm(grid, name, kernel=(3, 3, 3), filters=32):
    bs, im_bs, h, w, d, ch = grid.get_shape().as_list()
    
    if name not in reused_lay:
        reused_lay[name] = ConvLSTMCell(
            shape=[h, w, d], 
            initializer=slim.initializers.xavier_initializer(),
            kernel=kernel,
            filters=filters)
        
    print(grid.get_shape().as_list())
    print("h, w, d: {}, {}, {}".format(h, w, d))
    outputs = ConvRNN3D(reused_lay[name], stateful=False, return_state=False, input_shape=grid.get_shape().as_list(), name=name)(grid)
    print(outputs.shape)
#     outputs, states = tf.nn.dynamic_rnn(
#         conv_lstm,
#         grid,
#         parallel_iterations=64,
#         sequence_length=seq_length,
#         dtype=tf.float32,
#         time_major=False)
    return outputs

############################################################
#  Attention Layers
############################################################
def get_angles(pos, i, num_pos_feats):
    angle_rates = 1. / tf.math.pow(10000., (2. * (tf.cast(i, dtype=tf.float32) // 2.)) / tf.cast(num_pos_feats, dtype=tf.float32))
    return pos * angle_rates


def positional_encoding(positions, d_model):
    """computes the positional encoding for all axes
    positions: batch_size, im_bs, depth_samples, 3(x,y,z), npix
    pos_encoding: batch_size, npix*depth_samples flattened in order x, y, z, d_model
    """

    assert d_model % 3 == 0, "The depth of the model must be divisible by 3."
    num_pos_feats = d_model // 3

    # reshape positions to batch_size, 3(x,y,z), N*im_bs
    positions_shape = tf.shape(positions)
    positions = tf.transpose(positions, [0, 1, 2, 4, 3])
    positions = tf.reshape(positions, [-1, 3])
    x_embed, y_embed, z_embed = tf.meshgrid(positions[:, 0],
                                            positions[:, 1],
                                            positions[:, 2])
    print("shape of x_embed: {}".format(x_embed.get_shape().as_list()))
    grid_size = positions_shape[1]*positions_shape[2]*positions_shape[4]
    x_embed = tf.reshape(x_embed, [positions_shape[0], grid_size, grid_size, grid_size])
    y_embed = tf.reshape(y_embed, [positions_shape[0], grid_size, grid_size, grid_size])
    z_embed = tf.reshape(z_embed, [positions_shape[0], grid_size, grid_size, grid_size])
    print("shape of x_embed: {}".format(x_embed.get_shape().as_list()))
    angle_rads_x = get_angles(x_embed[..., tf.newaxis],
                              tf.range(num_pos_feats)[tf.newaxis, ...],
                              num_pos_feats)
    angle_rads_y = get_angles(y_embed[..., tf.newaxis],
                              tf.range(num_pos_feats)[tf.newaxis, ...],
                              num_pos_feats)
    angle_rads_z = get_angles(z_embed[..., tf.newaxis],
                              tf.range(num_pos_feats)[tf.newaxis, ...],
                              num_pos_feats)
    print("shape of angle_rads: {}".format(angle_rads_z.get_shape().as_list()))
    # apply sin to even indices in the array; 2i
    angle_rads_x = tf.stack([tf.math.sin(angle_rads_x[..., 0::2]),
                            tf.math.cos(angle_rads_x[..., 1::2])], axis=5)
    print("shape of angle_rads: {}".format(angle_rads_x.get_shape().as_list()))
    angle_rads_x = tf.reshape(angle_rads_x, (tf.shape(angle_rads_x)[0],
                                             tf.shape(angle_rads_x)[1],
                                             tf.shape(angle_rads_x)[2],
                                             tf.shape(angle_rads_x)[3],
                                             -1))
    angle_rads_y = tf.stack([tf.math.sin(angle_rads_y[..., 0::2]),
                            tf.math.cos(angle_rads_y[..., 1::2])], axis=5)
    angle_rads_y = tf.reshape(angle_rads_y, (tf.shape(angle_rads_y)[0],
                                             tf.shape(angle_rads_y)[1],
                                             tf.shape(angle_rads_y)[2],
                                             tf.shape(angle_rads_y)[3],
                                             -1))
    angle_rads_z = tf.stack([tf.math.sin(angle_rads_z[..., 0::2]),
                            tf.math.cos(angle_rads_z[..., 1::2])], axis=5)
    angle_rads_z = tf.reshape(angle_rads_z, (tf.shape(angle_rads_z)[0],
                                             tf.shape(angle_rads_z)[1],
                                             tf.shape(angle_rads_z)[2],
                                             tf.shape(angle_rads_z)[3],
                                             -1))
    print("shape of angle_rads: {}".format(angle_rads_z.get_shape().as_list()))
    # angle_rads_x[..., 0::2] = tf.math.sin(angle_rads_x[..., 0::2])
    # angle_rads_y[..., 0::2] = tf.math.sin(angle_rads_y[..., 0::2])
    # angle_rads_z[..., 0::2] = tf.math.sin(angle_rads_z[..., 0::2])
    #
    # # apply cos to odd indices in the array; 2i+1
    # angle_rads_x[..., 1::2] = tf.math.cos(angle_rads_x[..., 1::2])
    # angle_rads_y[..., 1::2] = tf.math.cos(angle_rads_y[..., 1::2])
    # angle_rads_z[..., 1::2] = tf.math.cos(angle_rads_z[..., 1::2])

    pos_encoding = tf.concat([angle_rads_x, angle_rads_y, angle_rads_z], axis=4)
    # flatten into 1D
    pos_encoding = tf.transpose(pos_encoding, [0, 4, 1, 2, 3])
    pos_encoding = tf.reshape(pos_encoding, [tf.shape(pos_encoding)[0], tf.shape(pos_encoding)[1], -1])
    pos_encoding = tf.transpose(pos_encoding, [0, 2, 1])
    print("pos_encoding: {}".format(pos_encoding.get_shape().as_list()))
    return pos_encoding


def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

    # add extra dimensions to add the padding
    # to the attention logits.
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)


def scaled_dot_product_attention(q, k, v, mask):
    """Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on its type(padding or look ahead)
    but it must be broadcastable for addition.

    Args:
    q: query shape == (..., seq_len_q, depth)
    k: key shape == (..., seq_len_k, depth)
    v: value shape == (..., seq_len_v, depth_v)
    mask: Float tensor with shape broadcastable
          to (..., seq_len_q, seq_len_k). Defaults to None.

    Returns:
    output, attention_weights
    """

    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

        # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights


class MultiHeadAttention(KL.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = KL.Dense(d_model)
        self.wk = KL.Dense(d_model)
        self.wv = KL.Dense(d_model)

        self.dense = KL.Dense(d_model)

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs, mask):
        v, k, q = inputs
        batch_size = tf.shape(q)[0]
        print("shape q: {}".format(q.get_shape().as_list()))
        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, mask)

        scaled_attention = tf.transpose(scaled_attention,
                                        perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output


def point_wise_feed_forward_network(d_model, dff):
    return keras.Sequential([
        KL.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
        KL.Dense(d_model)  # (batch_size, seq_len, d_model)
    ])


class EncoderLayer(KL.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = utils.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = utils.LayerNormalization(epsilon=1e-6)

        self.dropout1 = KL.Dropout(rate=rate)
        self.dropout2 = KL.Dropout(rate=rate)

    def call(self, x, training, mask):
        attn_output = self.mha([x, x, x], mask=mask)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

        return out2


class Encoder(KL.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff,
                 rate=0.1):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate)
                           for _ in range(num_layers)]

        self.dropout = KL.Dropout(rate=rate)

    def call(self, inputs, training, mask):
        x, positions = inputs
        pos_encoding = positional_encoding(positions, self.d_model)
        # adding embedding and position encoding.
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += pos_encoding

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training=training, mask=mask)

        return x  # (batch_size, input_seq_len, d_model)


class Transformer(KM.Model):
    def __init__(self, num_layers, d_model, num_heads, dff,
                 target_size, rate=0.1):
        super(Transformer, self).__init__()

        self.encoder = Encoder(num_layers, d_model, num_heads, dff,
                               rate)

        self.final_layer = KL.Dense(target_size)

    def call(self, inputs, training, mask):
        inp, positions = inputs
        enc_output = self.encoder([inp, positions], training=training, mask=mask)  # (batch_size, inp_seq_len, d_model)
        enc_output = tf.transpose(enc_output, [0, 2, 1])

        final_output = self.final_layer(enc_output)  # (batch_size, tar_seq_len, target_vocab_size)
        final_output = tf.tranpose(final_output, [0, 2, 1])
        return final_output


def transformer_encoder(feats, Rcam, Kmat, config, training, mask=None):
    positions = unproj_vector([feats, Rcam, Kmat], config)
    feats = tf.reshape(feats, [tf.shape(feats)[0], -1])
    print("shape pos, feats: {}, {}".format(positions.get_shape().as_list(), feats.get_shape().as_list()))
    # if not mask:
    #     mask = tf.zeros(tf.shape(feats))
    transformer = Transformer(num_layers=6, d_model=72, num_heads=8, dff=256, target_size=400, rate=0.1)
    feats = transformer([feats, positions], training=training, mask=mask)
    return feats


############################################################
#  Projective Geometry Layers
############################################################
def unproj_vector(inputs, config):
    feats, Rcam, Kmat = inputs
    bs, im_bs, fh, fw, fdim = feats.get_shape().as_list()
    print("feats shape: {}".format(feats.get_shape().as_list()))
    #im_bs = tf.shape(feats)[1]
    bs = config.BATCH_SIZE
    #Rcam = collapse_dims(Rcam)
    print("Kmat shape: {}".format(Kmat.get_shape().as_list()))
    Kmat = repeat_tensor_tf(Kmat, im_bs, rep_dim=1)
    print("Kmat shape: {}".format(Kmat.get_shape().as_list()))
    print("Rcam shape: {}".format(Rcam.get_shape().as_list()))
    # resize Kmat because original image plane had other dimension
    # than the one of the feature map
    assert fh == fw, "The height and width of the feature map are not matching {} and {}".format(fh, fw)
    rsz_factor = float(fh) / config.IMAGE_SHAPE[0]
    Kmat = Kmat * rsz_factor

    im_range = tf.range(0.5, fh, 1)
    npix = fh * fh
    image_indices = tf.stack(tf.meshgrid(im_range, im_range))
    image_indices = tf.reshape(image_indices, [2, -1])

    # Append rsz_factor to ensure that
    image_indices = tf.concat(
        [image_indices, tf.ones((1, npix))], axis=0)
    image_indices = tf.reshape(image_indices, [1, 1, 3, npix])
    image_indices = tf.tile(image_indices, [bs, im_bs, 1, 1])
    print("image_indices shape: {}".format(image_indices.get_shape().as_list()))
    print("Kmat shape: {}".format(Kmat.get_shape().as_list()))
    camera_coordinates = tf.matrix_triangular_solve(Kmat, image_indices, lower=False,)
    # sample from different depths
    camera_coordinates = repeat_tensor(camera_coordinates, config.samples, rep_dim=2)
    rho = tf.linspace(config.min_z, config.max_z, config.samples)
    camera_coordinates = camera_coordinates * rho[tf.newaxis, tf.newaxis, :, tf.newaxis,
                                              tf.newaxis]
    camera_coordinates = tf.reshape(camera_coordinates, [bs, im_bs, config.samples, 3, npix])
    print("camera_coordinates shape: {}".format(camera_coordinates.get_shape().as_list()))
    camera_coordinates = tf.concat(
        [camera_coordinates, tf.ones([bs, im_bs, config.samples, 1, npix])],
        axis=-2)
    # dimensions are now: bs, im_bs, different_depths, 4(x,y,z,1), npix
    print("camera_coordinates shape: {}".format(camera_coordinates.get_shape().as_list()))
    #camera_coordinates = collapse_dims(camera_coordinates)
    Rcam = repeat_tensor(Rcam, config.samples, rep_dim=2)
    world_coordinates = tf.matmul(Rcam, camera_coordinates)
    world_coordinates = tf.reshape(world_coordinates, [bs, im_bs, config.samples, 3, npix])
    print("world_coordinates shape: {}".format(world_coordinates.get_shape().as_list()))
    # dimensions are now: bs, im_bs, different_depths, 3(x,y,z), npix
    return world_coordinates

def unproj_feat(inputs, config):
    feats, Rcam, Kmat = inputs

    # Construct [R^{T}|-R^{T}t]
    Rcam_old = Rcam
    Rt = tf.matrix_transpose(Rcam[:, :, :, :3])
    tr = tf.expand_dims(Rcam[:, :, :, 3], axis=-1)
    # repeat Kmat 
    num_views = tf.shape(feats)[1]
    Kmat = gather_repeat(Kmat, num_views, axis=1)

    Rcam = tf.concat([Rt, -tf.matmul(Rt, tr)], axis=3)
    print("shape of Kmat: {}".format(Kmat.get_shape().as_list()))
    Rcam = collapse_dims(Rcam)
    print("shape of Rcam: {}".format(Rcam.get_shape().as_list()))
    Kmat = collapse_dims(Kmat)


    KRcam = tf.linalg.matmul(Kmat, Rcam)
    
    feats = collapse_dims(feats)
    feats_shape = tf.shape(feats)
    nR = feats_shape[0]
    _, fh, fw, fdim = tf_static_shape(feats)
    nR = num_views * config.BATCH_SIZE
#         fh = feats.shape[1].value
#         fw = feats.shape[2].value
#         fdim = feats.shape[3].value
    rsz_h = float(fh) / config.IMAGE_SHAPE[0]   # image height
    rsz_w = float(fw) / config.IMAGE_SHAPE[1]   # image width

    # Create Voxel grid 
    # !! change coordinates, grid has to be rotated, P-C?!!
    grid_range = tf.range(config.vmin + config.vsize / 2.0, config.vmax,
                                  config.vsize)
    grid_range_z = tf.range(-(config.nvox_z-1)*0.5*config.vsize, (config.nvox_z-1)*0.5*config.vsize+config.vsize/2, config.vsize)
    grid_range = tf.expand_dims(grid_range, 0)
    grid_range_z = tf.expand_dims(grid_range_z, 0)
    grid_range = tf.tile(grid_range, [config.BATCH_SIZE, 1])
    grid_range_z = tf.tile(grid_range_z, [config.BATCH_SIZE, 1])
    # calculate position of grid in world coordinate frame
    if hasattr(config, 'GRID_DIST'):
        grid_dist =config.GRID_DIST
    else: 
        grid_dist = 600/320 * config.vmax
    grid_position = K.dot(tf.reshape(Rcam_old[:,0,:,:], [-1, 3, 4]), tf.constant([0.0, 0.0, grid_dist, 1.0], shape=[4,1]) )
    grid_position = tf.reshape(grid_position, [config.BATCH_SIZE, 3])
    print("grid position shape: {}".format(grid_position.get_shape().as_list()))
    # adjust grid coordinates to world frame
    grid = tf.stack(
        tf.meshgrid(grid_range + grid_position[:,0, tf.newaxis],
                    grid_range + grid_position[:,1, tf.newaxis],
                    grid_range_z + grid_position[:,2, tf.newaxis]))# set z-offset from camera so that the grid side length equals the visible width
    rs_grid = tf.reshape(grid, [3, -1])
    nV = tf.shape(rs_grid)[1]
    rs_grid = tf.concat([rs_grid, tf.ones([1, nV])], axis=0)
    print("grid shape: {}".format(rs_grid.get_shape().as_list()))

    # Project grid/
    im_p = tf.matmul(tf.reshape(KRcam, [-1, 4]), rs_grid) 
    im_x, im_y, im_z = im_p[::3, :], im_p[1::3, :], im_p[2::3, :]
    im_x = (im_x / im_z) * rsz_w
    im_y = (im_y / im_z) * rsz_h


    # Bilinear interpolation
#     im_x = tf.clip_by_value(im_x, 0, fw - 1)
#     im_y = tf.clip_by_value(im_y, 0, fh - 1)
#     mask = tf.math.logical_and(im_x > 0,  im_x < fw - 1)
    im_x0 = tf.cast(tf.floor(im_x), 'int32')
    im_x1 = im_x0 + 1
    im_y0 = tf.cast(tf.floor(im_y), 'int32')
    im_y1 = im_y0 + 1
    im_x0_f, im_x1_f = tf.to_float(im_x0), tf.to_float(im_x1)
    im_y0_f, im_y1_f = tf.to_float(im_y0), tf.to_float(im_y1)

    ind_grid = tf.range(0, nR)
    ind_grid = tf.expand_dims(ind_grid, 1)
    im_ind = tf.tile(ind_grid, [1, nV])

    
    def _get_gather_inds(x, y):
        indices = tf.reshape(tf.stack([im_ind, y, x], axis=2), [-1, 3])
        return indices

    # Gather  values
    Ia = tf.gather_nd(feats, _get_gather_inds(im_x0, im_y0))
    Ib = tf.gather_nd(feats, _get_gather_inds(im_x0, im_y1))
    Ic = tf.gather_nd(feats, _get_gather_inds(im_x1, im_y0))
    Id = tf.gather_nd(feats, _get_gather_inds(im_x1, im_y1))
    # Calculate bilinear weights
    wa = (im_x1_f - im_x) * (im_y1_f - im_y)
    wb = (im_x1_f - im_x) * (im_y - im_y0_f)
    wc = (im_x - im_x0_f) * (im_y1_f - im_y)
    wd = (im_x - im_x0_f) * (im_y - im_y0_f)
    wa, wb = tf.reshape(wa, [-1, 1]), tf.reshape(wb, [-1, 1])
    wc, wd = tf.reshape(wc, [-1, 1]), tf.reshape(wd, [-1, 1])
    Ibilin = tf.add_n([wa * Ia, wb * Ib, wc * Ic, wd * Id])

    fdim = Ibilin.get_shape().as_list()[-1]
    Ibilin = tf.reshape(Ibilin, [
        config.BATCH_SIZE, num_views, config.nvox, config.nvox, config.nvox_z,
        fdim
    ])
    Ibilin = tf.transpose(Ibilin, [0, 1, 3, 2, 4, 5])

    return [Ibilin, grid_position]


def proj_grid(inputs, config, proj_size):
    """ projects the 3D feature grid back into a 2D image plane with equal side lengths of proj_size
    """
    grid, grid_pos, Rcam, Kmat = inputs
    rsz_factor = float(proj_size) / config.IMAGE_SHAPE[0]   # image height
    Kmat = Kmat * rsz_factor

    Kmat = tf.expand_dims(Kmat, axis=1)
   
    bs, h, w, d, ch = grid.get_shape().as_list()
#     im_bs = Rcam.get_shape().as_list()[1]

    Rcam = tf.reshape(Rcam[:,0,:,:], [bs, 1, 3, 4])
    
    npix = proj_size**2
    with tf.variable_scope('ProjSlice'):
        # Setup dimensions
        with tf.name_scope('PixelCenters'):
            # Setup image grids to unproject along rays
            im_range = tf.range(0.5, proj_size, 1)
            im_grid = tf.stack(tf.meshgrid(im_range, im_range))
            rs_grid = tf.reshape(im_grid, [2, -1])
            # Append rsz_factor to ensure that
            rs_grid = tf.concat(
                [rs_grid, tf.ones((1, npix)) * rsz_factor], axis=0)
            rs_grid = tf.reshape(rs_grid, [1, 1, 3, npix])
            rs_grid = tf.tile(rs_grid, [bs, 1, 1, 1])#[bs, im_bs, 1, 1]

        with tf.name_scope('Im2Cam'):
            # Compute Xc - points in camera frame
            Xc = tf.matrix_triangular_solve(
                Kmat, rs_grid, lower=False, name='KinvX')
            
            # Define z values of samples along ray
            if hasattr(config, 'GRID_DIST'):
                grid_dist =config.GRID_DIST
            else: 
                grid_dist = 600/320 * config.vmax
            z_samples = tf.linspace(grid_dist - config.vmax*0.5, grid_dist + config.vmax*0.5, config.samples)

            # Transform Xc to Xw using transpose of rotation matrix
            Xc = repeat_tensor(Xc, config.samples, rep_dim=2)
            
            Xc = Xc * z_samples[tf.newaxis, tf.newaxis, :, tf.newaxis,
                                tf.newaxis]
            
            Xc = tf.concat(
                [Xc, tf.ones([bs, 1, config.samples, 1, npix])],#[bs, im_bs, 1, 1]
                axis=-2)
            
        with tf.name_scope('Cam2World'):
            Rcam = repeat_tensor(Rcam, config.samples, rep_dim=2)
            Xw = tf.matmul(Rcam, Xc)
            # Transform world points to grid locations to sample from
            grid_pos = repeat_tensor(grid_pos, npix, rep_dim=-1)
            # take non symmetric grid into account
            vmin = tf.reshape(tf.constant([config.vmin, config.vmin, -config.nvox_z*0.5*config.vsize]), [1, 1, 3, 1])
            vmax = tf.reshape(tf.constant([config.vmax, config.vmax, config.nvox_z*0.5*config.vsize]), [1, 1, 3, 1])
            nvox = tf.reshape(tf.constant([config.nvox*1.0, config.nvox*1.0, config.nvox_z*1.0]), [1, 1, 3, 1]) 
            Xw = (Xw - grid_pos - vmin)
            Xw = (Xw / (vmax - vmin)) * nvox

            # bs, im_bs, samples, npix, 3
            Xw = tf.transpose(Xw, [0, 1, 2, 4, 3])

        with tf.name_scope('Interp'):
#                 sample_grid = collapse_dims(grid)
            sample_grid = grid
            sample_locs = collapse_dims(Xw)

            lshape = [bs * 1] + tf_static_shape(sample_locs)[1:] #tf_static_shape(sample_locs) [bs * 1]
            vox_idx = tf.range(lshape[0])
            vox_idx = repeat_tensor(vox_idx, lshape[1], rep_dim=1)
            vox_idx = tf.reshape(vox_idx, [-1, 1])
            vox_idx = repeat_tensor(vox_idx, 1 * npix, rep_dim=1)
            vox_idx = tf.reshape(vox_idx, [-1, 1])
            sample_idx = tf.concat(
                [tf.to_float(vox_idx),
                tf.reshape(sample_locs, [-1, 3])],
                axis=1)

            g_val = nearest3(sample_grid, sample_idx)
            g_val = tf.reshape(g_val, [
                bs, config.samples, proj_size, proj_size, -1    #bs, im_bs, proj_size, proj_size, -1
            ])
            ray_slices = tf.transpose(g_val, [0, 1, 2, 3, 4])
            return ray_slices

            
def repeat_tensor_tf(T, nrep, rep_dim=1):
        repT = tf.expand_dims(T, rep_dim)
        tile_dim = [1] * len(tf_static_shape(repT))
        tile_dim[rep_dim] = nrep
        repT = tf.tile(repT, tile_dim)
        return repT

def repeat_tensor(T, nrep, rep_dim=1):
        repT = tf.expand_dims(T, rep_dim)
        tile_dim = tf.Variable(tf.ones(shape=tf.rank(repT), dtype=tf.int32))
        tile_dim = tile_dim[rep_dim].assign(nrep)
        repT = tf.tile(repT, tile_dim)
        return repT
    
def gather_repeat(values, repeats, axis=1):
    values = tf.expand_dims(values, axis)
    indices = tf.tile(tf.zeros(shape=1, dtype=tf.int32), repeats[None])
    print(indices)
    return tf.gather(values, indices, axis=axis)
    
def collapse_dims(T):
    shape = tf_static_shape(T)
    shape = tf.concat([tf.constant(-1)[None], shape[2:]], axis=0) 
    return tf.reshape(T, shape)


def tf_static_shape(T):
    return T.get_shape().as_list()

def tf_dynamic_shape(T):
    return tf.shape(T)


def nearest3(grid, idx, clip=False):
    with tf.variable_scope('NearestInterp'):
        _, h, w, d, f = grid.get_shape().as_list()
        x, y, z = idx[:, 1], idx[:, 2], idx[:, 3]
        g_val = tf.gather_nd(grid, tf.cast(tf.round(idx), 'int32'))
        if clip:
            x_inv = tf.logical_or(x < 0, x > h - 1)
            y_inv = tf.logical_or(y < 0, y > w - 1)
            z_inv = tf.logical_or(z < 0, x > d - 1)
            valid_idx = 1 - \
                tf.to_float(tf.logical_or(tf.logical_or(x_inv, y_inv), z_inv))
            g_val = g_val * valid_idx[tf.newaxis, ...]
        return g_val
    
    
def quat2rot(q):
    '''q = [w, x, y, z]
    https://en.wikipedia.org/wiki/Rotation_matrix#Quaternion'''
    eps = 1e-5
    w, x, y, z = q
    n = np.linalg.norm(q)
    s = (0 if n < eps else 2.0 / n)
    wx = s * w * x
    wy = s * w * y
    wz = s * w * z
    xx = s * x * x
    xy = s * x * y
    xz = s * x * z
    yy = s * y * y
    yz = s * y * z
    zz = s * z * z
    R = np.array([[1 - (yy + zz), xy - wz,
                   xz + wy], [xy + wz, 1 - (xx + zz), yz - wx],
                  [xz - wy, yz + wx, 1 - (xx + yy)]])
    return R


def grid_reas(inputs, scope, config, kernel=(3, 3, 3), filters=256):

    x = inputs
#     x = KL.Lambda(lambda x: tf.reshape(x, grid_shape[0:4] + [grid_shape[4]*grid_shape[5]]))(x)
    name_conv = scope + '_3D_conv'
    name_bn = scope + '_batch_norm'
    if config.GRID_REAS == 'mean':
        grid_shape = tf_static_shape(x)
        x = KL.Lambda(lambda x: tf.transpose(x, [0, 5, 2, 3, 4, 1]))(x)
        x = KL.Lambda(lambda x: K.mean(x, axis=-1)[:, :, :, :, :, None])(x)
        x = KL.Lambda(lambda x: tf.transpose(x, [0, 5, 2, 3, 4, 1]))(x)
        x = KL.Lambda(lambda x: tf.reshape(x, [grid_shape[0]] + grid_shape[2:]))(x)
        x = add_bn_layer(name=name_bn)(x, training=config.TRAIN_BN)
        x = KL.Activation('relu')(x)
        
    elif config.GRID_REAS == 'conv3d':
        grid_shape = tf_static_shape(x)
        x_shape = x.shape.as_list()
        # reorder dimensions from batch_size, num_views, h, w, d, chn to batch_size, h, w, d chn*num_views
        x = KL.Lambda(lambda x: tf.reshape(x, [x_shape[0]]+x_shape[2:-1]+[config.NUM_VIEWS*x_shape[-1]]))(x)
        x = KL.Activation('relu')(x)
        if name_conv not in reused_lay:
            reused_lay[name_conv] = KL.Conv3D(filters=config.TOP_DOWN_PYRAMID_SIZE, kernel_size=(3,3,3), padding='same', name=name_conv)
        x = reused_lay[name_conv](x)
        x = add_bn_layer(name=name_bn)(x, training=config.TRAIN_BN)
#         x = KL.LeakyReLU(alpha=0.01)(x)
        x = KL.Activation('relu')(x)
    
    elif config.GRID_REAS == 'ident':
        x_shape = x.get_shape().as_list()
        x = KL.Lambda(lambda x: tf.reshape(x, [x_shape[0]]+x_shape[2:-1]+[config.NUM_VIEWS*x_shape[-1]]))(x)
        x = KL.Activation('relu')(x)
        if name_conv not in reused_lay:
            reused_lay[name_conv] = KL.Conv3D(
                filters=config.TOP_DOWN_PYRAMID_SIZE, kernel_size=(1,1,1), 
                padding='same', name=scope+'ident_conv')
        x = reused_lay[name_conv](x)
        x = add_bn_layer(name=name_bn)(x, training=config.TRAIN_BN)
        x = KL.Activation('relu')(x)
        
    elif config.GRID_REAS == 'lstm3d':
        name_conv = scope + '_convlstm3d'
        x = KL.Activation('relu')(x)
        x = convlstm(x, name=name_conv, kernel=[3, 3, 3], filters=config.TOP_DOWN_PYRAMID_SIZE)
        x = add_bn_layer(name=name_bn)(x, training=config.TRAIN_BN)
        x = KL.Activation('relu')(x)
        
    elif config.GRID_REAS == 'cudnnlstm':
        name_conv = scope + '_convlstm3d'
        x = KL.Activation('relu')(x)
        
        x = KL.CuDNNLSTM(units=config.TOP_DOWN_PYRAMID_SIZE)
        x = add_bn_layer(name=name_bn)(x, training=config.TRAIN_BN)
        x = KL.Activation('relu')(x)
        
    return x
    
    
def depth_sampling(x, config, name):
    x = KL.TimeDistributed(KL.Conv2D(1, (1,1), padding='same'), name=name+'2DConv')(x)
    x = add_bn_layer(name=name+'bn')(x, training=config.TRAIN_BN)
#     x = KL.LeakyReLU(alpha=0.01)(x)
    x = KL.Activation('relu')(x)
    return x
############################################################
#  Resnet Graph
############################################################

# class bb_resnet_fpn(KL.Layer):
#     def __init__(self, config, architecture, stage5=True, train_bn=True, **kwargs):
#         self.config = config
#         super(bb_resnet_fpn, self).__init__(**kwargs)
#     def call(self, inputs):
#         ''' computes the feature map for one batch'''
#         input_image = inputs
#         print("input_image_shape: {}".format(input_image.get_shape().as_list()))
#         config = self.config
#         print("fn image_shape: {}".format(input_image.get_shape().as_list()))
#         if callable(config.BACKBONE):
#             _, C2, C3, C4, C5 = config.BACKBONE(input_image, stage5=True,
#                                                 train_bn=config.TRAIN_BN)
#         else:
#             _, C2, C3, C4, C5 = resnet_graph(input_image, architecture=config.BACKBONE, stage5=True, train_bn=True)
#         # Top-down Layers
#         # TODO: add assert to varify feature map sizes match what's in config
#         P5 = tf.keras.layers.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c5p5')(C5)
#         P4 = tf.keras.layers.Add()([
#             tf.keras.layers.UpSampling2D(size=(2, 2))(P5),
#             tf.keras.layers.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c4p4')(C4)])
#         P3 = tf.keras.layers.Add()([
#             tf.keras.layers.UpSampling2D(size=(2, 2))(P4),
#             tf.keras.layers.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c3p3')(C3)])
#         P2 = tf.keras.layers.Add()([
#             tf.keras.layers.UpSampling2D(size=(2, 2), name="fpn_p3upsampled")(P3),
#             tf.keras.layers.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c2p2')(C2)])
#         # Attach 3x3 conv to all P layers to get the final feature maps.
#         P2 = tf.keras.layers.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p2")(P2)
#         P3 = tf.keras.layers.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p3")(P3)
#         P4 = tf.keras.layers.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p4")(P4)
#         P5 = tf.keras.layers.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p5")(P5)
#         # P6 is used for the 5th anchor scale in RPN. Generated by
#         # subsampling from P5 with stride of 2.
#         P6 = tf.keras.layers.MaxPool2D(pool_size=(1, 1), strides=2)(P5)
# #         results = tf.stack([P2, P3, P4, P5, P6], axis=0, name='stack')

#         print("shape_out: {}".format(P2.get_shape().as_list()))
#         print("shape_out: {}".format(P3.get_shape().as_list()))
#         print("shape_out: {}".format(P4.get_shape().as_list()))
#         print("shape_out: {}".format(P5.get_shape().as_list()))
#         print("shape_out: {}".format(P6.get_shape().as_list()))
#         return [P2, P3, P4, P5, P6]
    
#     def compute_output_shape(self,inputShape):
#         #calculate shapes from input shape   
#         print("compute_output_shape")
#         return [(None, None, None, 256)]*5
    
# Code adopted from:
# https://github.com/fchollet/deep-learning-models/blob/master/resnet50.py

def add_conv_layer(input, scope, filters, kernel_size, strides=(1, 1), name='conv1', use_bias=True, padding='valid'):
#     if name not in reused_lay:
#         reused_lay[name] = KL.Conv2D(filters, kernel_size, strides=strides, name=name, use_bias=use_bias, padding=padding)
#     x = reused_lay[name](input)
    x = KL.TimeDistributed(KL.Conv2D(filters, kernel_size, strides=strides, use_bias=use_bias, padding=padding), name=name)(input)
    return x

def add_bn_layer(name):
#     if name not in reused_lay:
#         reused_lay[name] = BatchNorm(name=name)
#     return reused_lay[name]
    return KL.TimeDistributed(BatchNorm(), name=name)

def identity_block(input_tensor, kernel_size, filters, stage, block,
                   use_bias=True, train_bn=True):
    """The identity_block is the block that has no conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        use_bias: Boolean. To use or not use a bias in conv layers.
        train_bn: Boolean. Train or freeze Batch Norm layers
    """
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = add_conv_layer(input_tensor, conv_name_base, nb_filter1, (1, 1), name=conv_name_base + '2a', use_bias=use_bias)
    x = add_bn_layer(name=bn_name_base + '2a')(x, training=train_bn)
    x = KL.Activation('relu')(x)
    
    x = add_conv_layer(x, conv_name_base, nb_filter2, (kernel_size, kernel_size), padding='same', name=conv_name_base + '2b', use_bias=use_bias)
    x = add_bn_layer(name=bn_name_base + '2b')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    x = add_conv_layer(x, conv_name_base,nb_filter3, (1, 1), name=conv_name_base + '2c', use_bias=use_bias)
    x = add_bn_layer(name=bn_name_base + '2c')(x, training=train_bn)

    x = KL.Add()([x, input_tensor])
    x = KL.Activation('relu')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block,
               strides=(2, 2), use_bias=True, train_bn=True):
    """conv_block is the block that has a conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        use_bias: Boolean. To use or not use a bias in conv layers.
        train_bn: Boolean. Train or freeze Batch Norm layers
    Note that from stage 3, the first conv layer at main path is with subsample=(2,2)
    And the shortcut should have subsample=(2,2) as well
    """
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    x = add_conv_layer(input_tensor, conv_name_base, nb_filter1, (1, 1), strides=strides, name=conv_name_base + '2a', use_bias=use_bias)
    x = add_bn_layer(name=bn_name_base + '2a')(x, training=train_bn)
    x = KL.Activation('relu')(x)
    
    x = add_conv_layer(x, conv_name_base, nb_filter2, (kernel_size, kernel_size), padding='same', name=conv_name_base + '2b', use_bias=use_bias)
    x = add_bn_layer(name=bn_name_base + '2b')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    x = add_conv_layer(x, conv_name_base, nb_filter3, (1, 1), name=conv_name_base + '2c', use_bias=use_bias)
    x = add_bn_layer(name=bn_name_base + '2c')(x, training=train_bn)
    
    shortcut = add_conv_layer(input_tensor, conv_name_base, nb_filter3, (1, 1), strides=strides, name=conv_name_base + '1', use_bias=use_bias)
    shortcut = add_bn_layer(name=bn_name_base + '1')(shortcut, training=train_bn)

    x = KL.Add()([x, shortcut])
    x = KL.Activation('relu')(x)
    return x

# def resnet_graph(input_image, architecture, stage5=False, train_bn=True):
#     """Build a ResNet graph.
#         architecture: Can be resnet50 or resnet101
#         stage5: Boolean. If False, stage5 of the network is not created
#         train_bn: Boolean. Train or freeze Batch Norm layers
#     """
#     assert architecture in ["resnet50", "resnet101"]
#     # Stage 1
#     x = KL.TimeDistributed(KL.ZeroPadding2D((3, 3)))(input_image)
#     x = KL.TimeDistributed(KL.Conv2D(64, (7, 7), strides=(2, 2), use_bias=True), name='conv1')(x)
#     x = KL.TimeDistributed(BatchNorm(), name='bn_conv1')(x, training=train_bn)
#     x = KL.Activation('relu')(x)
#     C1 = x = KL.TimeDistributed(KL.MaxPooling2D((3, 3), strides=(2, 2), padding="same"))(x)
#     # Stage 2
#     x = conv_block(x, 3, [32, 32, 128], stage=2, block='a', strides=(1, 1), train_bn=train_bn)
#     x = identity_block(x, 3, [32, 32, 128], stage=2, block='b', train_bn=train_bn)
#     C2 = x = identity_block(x, 3, [32, 32, 128], stage=2, block='c', train_bn=train_bn)
#     # Stage 3
#     x = conv_block(x, 3, [64, 64, 128], stage=3, block='a', train_bn=train_bn)
#     x = identity_block(x, 3, [64, 64, 128], stage=3, block='b', train_bn=train_bn)
#     x = identity_block(x, 3, [64, 64, 128], stage=3, block='c', train_bn=train_bn)
#     C3 = x = identity_block(x, 3, [64, 64, 128], stage=3, block='d', train_bn=train_bn)
#     # Stage 4
#     x = conv_block(x, 3, [64, 64, 256], stage=4, block='a', train_bn=train_bn)
#     block_count = {"resnet50": 3, "resnet101": 22}[architecture]
#     for i in range(block_count):
#         x = identity_block(x, 3, [128, 128, 256], stage=4, block=chr(98 + i), train_bn=train_bn)
#     C4 = x
#     # Stage 5
#     if stage5:
#         x = conv_block(x, 3, [128, 128, 256], stage=5, block='a', train_bn=train_bn)
#         x = identity_block(x, 3, [128, 128, 256], stage=5, block='b', train_bn=train_bn)
#         C5 = x = identity_block(x, 3, [128, 128, 256], stage=5, block='c', train_bn=train_bn)
#     else:
#         C5 = None
#     return [C1, C2, C3, C4, C5]


def resnet_graph(input_image, architecture, stage5=False, train_bn=True):
    """Build a ResNet graph.
        architecture: Can be resnet50 or resnet101
        stage5: Boolean. If False, stage5 of the network is not created
        train_bn: Boolean. Train or freeze Batch Norm layers
    """
    assert architecture in ["resnet50", "resnet101"]
    # Stage 1
    x = KL.TimeDistributed(KL.ZeroPadding2D((3, 3)))(input_image)
    x = add_conv_layer(x, scope='res1', filters=64, kernel_size=(7, 7), strides=(2, 2), name='conv1', use_bias=True)
    x = add_bn_layer(name='bn_conv1')(x, training=train_bn)
    x = KL.Activation('relu')(x)
    C1 = x = KL.TimeDistributed(KL.MaxPool2D((3, 3), strides=(2, 2), padding="same"))(x)
    # Stage 2
    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1), train_bn=train_bn)
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b', train_bn=train_bn)
    C2 = x = identity_block(x, 3, [64, 64, 256], stage=2, block='c', train_bn=train_bn)
    # Stage 3
    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a', train_bn=train_bn)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b', train_bn=train_bn)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c', train_bn=train_bn)
    C3 = x = identity_block(x, 3, [128, 128, 512], stage=3, block='d', train_bn=train_bn)
    # Stage 4
    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a', train_bn=train_bn)
    block_count = {"resnet50": 5, "resnet101": 22}[architecture]
    for i in range(block_count):
        x = identity_block(x, 3, [256, 256, 1024], stage=4, block=chr(98 + i), train_bn=train_bn)
    C4 = x
    # Stage 5
    if stage5:
        x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a', train_bn=train_bn)
        x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b', train_bn=train_bn)
        C5 = x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c', train_bn=train_bn)
    else:
        C5 = None
    return [C1, C2, C3, C4, C5]

def build_resnet_fpn(input_image, config):
    ''' computes the feature map for one batch'''
    print("input_image_shape: {}".format(input_image.get_shape().as_list()))
    print("fn image_shape: {}".format(input_image.get_shape().as_list()))
    if callable(config.BACKBONE):
        _, C2, C3, C4, C5 = config.BACKBONE(input_image, stage5=True,
                                            train_bn=config.TRAIN_BN)
    else:
        _, C2, C3, C4, C5 = resnet_graph(input_image, architecture=config.BACKBONE, stage5=True, train_bn=config.TRAIN_BN)
    # Top-down Layers
    # TODO: add assert to varify feature map sizes match what's in config
    P5 = add_conv_layer(C5, 'fpn', config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c5p5')
    P4 = KL.Add()([
        KL.TimeDistributed(KL.UpSampling2D(size=(2, 2)))(P5),
        add_conv_layer(C4, 'fpn', config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c4p4')])
    P3 = KL.Add()([
        KL.TimeDistributed(KL.UpSampling2D(size=(2, 2)))(P4),
        add_conv_layer(C3, 'fpn', config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c3p3')])
    P2 = KL.Add()([
        KL.TimeDistributed(KL.UpSampling2D(size=(2, 2)))(P3),
        add_conv_layer(C2, 'fpn', config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c2p2')])
    # Attach 3x3 conv to all P layers to get the final feature maps.
    P2 = add_conv_layer(P2, 'fpn', config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p2")
    P3 = add_conv_layer(P3, 'fpn', config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p3")
    P4 = add_conv_layer(P4, 'fpn', config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p4")
    P5 = add_conv_layer(P5, 'fpn', config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p5")
    # P6 is used for the 5th anchor scale in RPN. Generated by
    # subsampling from P5 with stride of 2.
    P6 = KL.TimeDistributed(KL.MaxPool2D(pool_size=(1, 1), strides=2))(P5)

    print("shape_out: {}".format(P2.get_shape().as_list()))
    print("shape_out: {}".format(P3.get_shape().as_list()))
    print("shape_out: {}".format(P4.get_shape().as_list()))
    print("shape_out: {}".format(P5.get_shape().as_list()))
    print("shape_out: {}".format(P6.get_shape().as_list()))
    return [P2, P3, P4, P5, P6]

def view_merger_model(input_shape, config):
    print("shape of input: {}".format(input_shape))
    input_images = KL.Input(shape=input_shape)
    
    P2, P3, P4, P5, P6 = build_resnet_fpn(input_images, config=config)
    
    merged_bb = KM.Model(input_images, [P2, P3, P4, P5, P6], name='backbone')
    print("finished whole bb model")
    return merged_bb

############################################################
#  Proposal Layer
############################################################

def apply_box_deltas_graph(boxes, deltas):
    """Applies the given deltas to the given boxes.
    boxes: [N, (y1, x1, y2, x2)] boxes to update
    deltas: [N, (dy, dx, log(dh), log(dw))] refinements to apply
    """
    # Convert to y, x, h, w
    height = boxes[:, 2] - boxes[:, 0]
    width = boxes[:, 3] - boxes[:, 1]
    center_y = boxes[:, 0] + 0.5 * height
    center_x = boxes[:, 1] + 0.5 * width
    # Apply deltas
    center_y += deltas[:, 0] * height
    center_x += deltas[:, 1] * width
    height *= tf.exp(deltas[:, 2])
    width *= tf.exp(deltas[:, 3])
    # Convert back to y1, x1, y2, x2
    y1 = center_y - 0.5 * height
    x1 = center_x - 0.5 * width
    y2 = y1 + height
    x2 = x1 + width
    result = tf.stack([y1, x1, y2, x2], axis=1, name="apply_box_deltas_out")
    return result


def clip_boxes_graph(boxes, window):
    """
    boxes: [N, (y1, x1, y2, x2)]
    window: [4] in the form y1, x1, y2, x2
    """
    # Split
    wy1, wx1, wy2, wx2 = tf.split(window, 4)
    y1, x1, y2, x2 = tf.split(boxes, 4, axis=1)
    # Clip
    y1 = tf.maximum(tf.minimum(y1, wy2), wy1)
    x1 = tf.maximum(tf.minimum(x1, wx2), wx1)
    y2 = tf.maximum(tf.minimum(y2, wy2), wy1)
    x2 = tf.maximum(tf.minimum(x2, wx2), wx1)
    clipped = tf.concat([y1, x1, y2, x2], axis=1, name="clipped_boxes")
    clipped.set_shape((clipped.shape[0], 4))
    return clipped


class ProposalLayer(KE.Layer):
    """Receives anchor scores and selects a subset to pass as proposals
    to the second stage. Filtering is done based on anchor scores and
    non-max suppression to remove overlaps. It also applies bounding
    box refinement deltas to anchors.

    Inputs:
        rpn_probs: [batch, num_anchors, (bg prob, fg prob)]
        rpn_bbox: [batch, num_anchors, (dy, dx, log(dh), log(dw))]
        anchors: [batch, num_anchors, (y1, x1, y2, x2)] anchors in normalized coordinates

    Returns:
        Proposals in normalized coordinates [batch, rois, (y1, x1, y2, x2)]
    """

    def __init__(self, proposal_count, nms_threshold, config=None, **kwargs):
        super(ProposalLayer, self).__init__(**kwargs)
        self.config = config
        self.proposal_count = proposal_count
        self.nms_threshold = nms_threshold

    def call(self, inputs):
        # Box Scores. Use the foreground class confidence. [Batch, num_rois, 1]
        scores = inputs[0][:, :, 1]
        # Box deltas [batch, num_rois, 4]
        deltas = inputs[1]
        deltas = deltas * np.reshape(self.config.RPN_BBOX_STD_DEV, [1, 1, 4])
        # Anchors
        anchors = inputs[2]

        # Improve performance by trimming to top anchors by score
        # and doing the rest on the smaller subset.
        pre_nms_limit = tf.minimum(self.config.PRE_NMS_LIMIT, tf.shape(anchors)[1])
        ix = tf.nn.top_k(scores, pre_nms_limit, sorted=True,
                         name="top_anchors").indices
        scores = utils.batch_slice([scores, ix], lambda x, y: tf.gather(x, y),
                                   self.config.IMAGES_PER_GPU)
        deltas = utils.batch_slice([deltas, ix], lambda x, y: tf.gather(x, y),
                                   self.config.IMAGES_PER_GPU)
        pre_nms_anchors = utils.batch_slice([anchors, ix], lambda a, x: tf.gather(a, x),
                                    self.config.IMAGES_PER_GPU,
                                    names=["pre_nms_anchors"])

        # Apply deltas to anchors to get refined anchors.
        # [batch, N, (y1, x1, y2, x2)]
        boxes = utils.batch_slice([pre_nms_anchors, deltas],
                                  lambda x, y: apply_box_deltas_graph(x, y),
                                  self.config.IMAGES_PER_GPU,
                                  names=["refined_anchors"])

        # Clip to image boundaries. Since we're in normalized coordinates,
        # clip to 0..1 range. [batch, N, (y1, x1, y2, x2)]
        window = np.array([0, 0, 1, 1], dtype=np.float32)
        boxes = utils.batch_slice(boxes,
                                  lambda x: clip_boxes_graph(x, window),
                                  self.config.IMAGES_PER_GPU,
                                  names=["refined_anchors_clipped"])

        # Filter out small boxes
        # According to Xinlei Chen's paper, this reduces detection accuracy
        # for small objects, so we're skipping it.

        # Non-max suppression
        def nms(boxes, scores):
            indices = tf.image.non_max_suppression(
                boxes, scores, self.proposal_count,
                self.nms_threshold, name="rpn_non_max_suppression")
            proposals = tf.gather(boxes, indices)
            # Pad if needed
            padding = tf.maximum(self.proposal_count - tf.shape(proposals)[0], 0)
            proposals = tf.pad(proposals, [(0, padding), (0, 0)])
            return proposals
        proposals = utils.batch_slice([boxes, scores], nms,
                                      self.config.IMAGES_PER_GPU)
        return proposals

    def compute_output_shape(self, input_shape):
        return (None, self.proposal_count, 4)


############################################################
#  ROIAlign Layer
############################################################

def log2_graph(x):
    """Implementation of Log2. TF doesn't have a native implementation."""
    return tf.log(x) / tf.log(2.0)


class PyramidROIAlign(KE.Layer):
    """Implements ROI Pooling on multiple levels of the feature pyramid.

    Params:
    - pool_shape: [pool_height, pool_width] of the output pooled regions. Usually [7, 7]

    Inputs:
    - boxes: [batch, num_boxes, (y1, x1, y2, x2)] in normalized
             coordinates. Possibly padded with zeros if not enough
             boxes to fill the array.
    - image_meta: [batch, (meta data)] Image details. See compose_image_meta()
    - feature_maps: List of feature maps from different levels of the pyramid.
                    Each is [batch, height, width, channels]

    Output:
    Pooled regions in the shape: [batch, num_boxes, pool_height, pool_width, channels].
    The width and height are those specific in the pool_shape in the layer
    constructor.
    """

    def __init__(self, pool_shape, **kwargs):
        super(PyramidROIAlign, self).__init__(**kwargs)
        self.pool_shape = tuple(pool_shape)

    def call(self, inputs):
        # Crop boxes [batch, num_boxes, (y1, x1, y2, x2)] in normalized coords
        boxes = inputs[0]

        # Image meta
        # Holds details about the image. See compose_image_meta()
        image_meta = inputs[1]

        # Feature Maps. List of feature maps from different level of the
        # feature pyramid. Each is [batch, height, width, channels]
        feature_maps = inputs[2:]

        # Assign each ROI to a level in the pyramid based on the ROI area.
        y1, x1, y2, x2 = tf.split(boxes, 4, axis=2)
        h = y2 - y1
        w = x2 - x1
        # Use shape of first image. Images in a batch must have the same size.
        image_shape = parse_image_meta_graph(image_meta)['image_shape'][0]
        # Equation 1 in the Feature Pyramid Networks paper. Account for
        # the fact that our coordinates are normalized here.
        # e.g. a 224x224 ROI (in pixels) maps to P4
        image_area = tf.cast(image_shape[0] * image_shape[1], tf.float32)
        roi_level = log2_graph(tf.sqrt(h * w) / (224.0 / tf.sqrt(image_area)))
        roi_level = tf.minimum(5, tf.maximum(
            2, 4 + tf.cast(tf.round(roi_level), tf.int32)))
        roi_level = tf.squeeze(roi_level, 2)

        # Loop through levels and apply ROI pooling to each. P2 to P5.
        pooled = []
        box_to_level = []
        for i, level in enumerate(range(2, 6)):
            ix = tf.where(tf.equal(roi_level, level))
            level_boxes = tf.gather_nd(boxes, ix)

            # Box indices for crop_and_resize.
            box_indices = tf.cast(ix[:, 0], tf.int32)

            # Keep track of which box is mapped to which level
            box_to_level.append(ix)

            # Stop gradient propogation to ROI proposals
            level_boxes = tf.stop_gradient(level_boxes)
            box_indices = tf.stop_gradient(box_indices)

            # Crop and Resize
            # From Mask R-CNN paper: "We sample four regular locations, so
            # that we can evaluate either max or average pooling. In fact,
            # interpolating only a single value at each bin center (without
            # pooling) is nearly as effective."
            #
            # Here we use the simplified approach of a single value per bin,
            # which is how it's done in tf.crop_and_resize()
            # Result: [batch * num_boxes, pool_height, pool_width, channels]
            pooled.append(tf.image.crop_and_resize(
                feature_maps[i], level_boxes, box_indices, self.pool_shape,
                method="bilinear"))

        # Pack pooled features into one tensor
        pooled = tf.concat(pooled, axis=0)

        # Pack box_to_level mapping into one array and add another
        # column representing the order of pooled boxes
        box_to_level = tf.concat(box_to_level, axis=0)
        box_range = tf.expand_dims(tf.range(tf.shape(box_to_level)[0]), 1)
        box_to_level = tf.concat([tf.cast(box_to_level, tf.int32), box_range],
                                 axis=1)

        # Rearrange pooled features to match the order of the original boxes
        # Sort box_to_level by batch then box index
        # TF doesn't have a way to sort by two columns, so merge them and sort.
        sorting_tensor = box_to_level[:, 0] * 100000 + box_to_level[:, 1]
        ix = tf.nn.top_k(sorting_tensor, k=tf.shape(
            box_to_level)[0]).indices[::-1]
        ix = tf.gather(box_to_level[:, 2], ix)
        pooled = tf.gather(pooled, ix)

        # Re-add the batch dimension
        shape = tf.concat([tf.shape(boxes)[:2], tf.shape(pooled)[1:]], axis=0)
        pooled = tf.reshape(pooled, shape)
        return pooled

    def compute_output_shape(self, input_shape):
        return input_shape[0][:2] + self.pool_shape + (input_shape[2][-1], )


############################################################
#  Detection Target Layer
############################################################

def overlaps_graph(boxes1, boxes2):
    """Computes IoU overlaps between two sets of boxes.
    boxes1, boxes2: [N, (y1, x1, y2, x2)].
    """
    # 1. Tile boxes2 and repeat boxes1. This allows us to compare
    # every boxes1 against every boxes2 without loops.
    # TF doesn't have an equivalent to np.repeat() so simulate it
    # using tf.tile() and tf.reshape.
    b1 = tf.reshape(tf.tile(tf.expand_dims(boxes1, 1),
                            [1, 1, tf.shape(boxes2)[0]]), [-1, 4])
    b2 = tf.tile(boxes2, [tf.shape(boxes1)[0], 1])
    # 2. Compute intersections
    b1_y1, b1_x1, b1_y2, b1_x2 = tf.split(b1, 4, axis=1)
    b2_y1, b2_x1, b2_y2, b2_x2 = tf.split(b2, 4, axis=1)
    y1 = tf.maximum(b1_y1, b2_y1)
    x1 = tf.maximum(b1_x1, b2_x1)
    y2 = tf.minimum(b1_y2, b2_y2)
    x2 = tf.minimum(b1_x2, b2_x2)
    intersection = tf.maximum(x2 - x1, 0) * tf.maximum(y2 - y1, 0)
    # 3. Compute unions
    b1_area = (b1_y2 - b1_y1) * (b1_x2 - b1_x1)
    b2_area = (b2_y2 - b2_y1) * (b2_x2 - b2_x1)
    union = b1_area + b2_area - intersection
    # 4. Compute IoU and reshape to [boxes1, boxes2]
    iou = intersection / union
    overlaps = tf.reshape(iou, [tf.shape(boxes1)[0], tf.shape(boxes2)[0]])
    return overlaps


def detection_targets_graph(proposals, gt_class_ids, gt_boxes, gt_masks, config):
    """Generates detection targets for one image. Subsamples proposals and
    generates target class IDs, bounding box deltas, and masks for each.

    Inputs:
    proposals: [POST_NMS_ROIS_TRAINING, (y1, x1, y2, x2)] in normalized coordinates. Might
               be zero padded if there are not enough proposals.
    gt_class_ids: [MAX_GT_INSTANCES] int class IDs
    gt_boxes: [MAX_GT_INSTANCES, (y1, x1, y2, x2)] in normalized coordinates.
    gt_masks: [height, width, MAX_GT_INSTANCES] of boolean type.

    Returns: Target ROIs and corresponding class IDs, bounding box shifts,
    and masks.
    rois: [TRAIN_ROIS_PER_IMAGE, (y1, x1, y2, x2)] in normalized coordinates
    class_ids: [TRAIN_ROIS_PER_IMAGE]. Integer class IDs. Zero padded.
    deltas: [TRAIN_ROIS_PER_IMAGE, (dy, dx, log(dh), log(dw))]
    masks: [TRAIN_ROIS_PER_IMAGE, height, width]. Masks cropped to bbox
           boundaries and resized to neural network output size.

    Note: Returned arrays might be zero padded if not enough target ROIs.
    """
    # Assertions
    asserts = [
        tf.Assert(tf.greater(tf.shape(proposals)[0], 0), [proposals],
                  name="roi_assertion"),
    ]
    with tf.control_dependencies(asserts):
        proposals = tf.identity(proposals)

    # Remove zero padding
    proposals, _ = trim_zeros_graph(proposals, name="trim_proposals")
    gt_boxes, non_zeros = trim_zeros_graph(gt_boxes, name="trim_gt_boxes")
    gt_class_ids = tf.boolean_mask(gt_class_ids, non_zeros,
                                   name="trim_gt_class_ids")
    gt_masks = tf.gather(gt_masks, tf.where(non_zeros)[:, 0], axis=2,
                         name="trim_gt_masks")

    # Handle COCO crowds
    # A crowd box in COCO is a bounding box around several instances. Exclude
    # them from training. A crowd box is given a negative class ID.
    crowd_ix = tf.where(gt_class_ids < 0)[:, 0]
    non_crowd_ix = tf.where(gt_class_ids > 0)[:, 0]
    crowd_boxes = tf.gather(gt_boxes, crowd_ix)
    gt_class_ids = tf.gather(gt_class_ids, non_crowd_ix)
    gt_boxes = tf.gather(gt_boxes, non_crowd_ix)
    gt_masks = tf.gather(gt_masks, non_crowd_ix, axis=2)

    # Compute overlaps matrix [proposals, gt_boxes]
    overlaps = overlaps_graph(proposals, gt_boxes)

    # Compute overlaps with crowd boxes [proposals, crowd_boxes]
    crowd_overlaps = overlaps_graph(proposals, crowd_boxes)
    crowd_iou_max = tf.reduce_max(crowd_overlaps, axis=1)
    no_crowd_bool = (crowd_iou_max < 0.001)

    # Determine positive and negative ROIs
    roi_iou_max = tf.reduce_max(overlaps, axis=1)
    # 1. Positive ROIs are those with >= 0.5 IoU with a GT box
    positive_roi_bool = (roi_iou_max >= 0.5)
    positive_indices = tf.where(positive_roi_bool)[:, 0]
    # 2. Negative ROIs are those with < 0.5 with every GT box. Skip crowds.
    negative_indices = tf.where(tf.logical_and(roi_iou_max < 0.5, no_crowd_bool))[:, 0]

    # Subsample ROIs. Aim for 33% positive
    # Positive ROIs
    positive_count = int(config.TRAIN_ROIS_PER_IMAGE *
                         config.ROI_POSITIVE_RATIO)
    positive_indices = tf.random_shuffle(positive_indices)[:positive_count]
    positive_count = tf.shape(positive_indices)[0]
    # Negative ROIs. Add enough to maintain positive:negative ratio.
    r = 1.0 / config.ROI_POSITIVE_RATIO
    negative_count = tf.cast(r * tf.cast(positive_count, tf.float32), tf.int32) - positive_count
    negative_indices = tf.random_shuffle(negative_indices)[:negative_count]
    # Gather selected ROIs
    positive_rois = tf.gather(proposals, positive_indices)
    negative_rois = tf.gather(proposals, negative_indices)

    # Assign positive ROIs to GT boxes.
    positive_overlaps = tf.gather(overlaps, positive_indices)
    roi_gt_box_assignment = tf.cond(
        tf.greater(tf.shape(positive_overlaps)[1], 0),
        true_fn = lambda: tf.argmax(positive_overlaps, axis=1),
        false_fn = lambda: tf.cast(tf.constant([]),tf.int64)
    )
    roi_gt_boxes = tf.gather(gt_boxes, roi_gt_box_assignment)
    roi_gt_class_ids = tf.gather(gt_class_ids, roi_gt_box_assignment)

    # Compute bbox refinement for positive ROIs
    deltas = utils.box_refinement_graph(positive_rois, roi_gt_boxes)
    deltas /= config.BBOX_STD_DEV

    # Assign positive ROIs to GT masks
    # Permute masks to [N, height, width, 1]
    transposed_masks = tf.expand_dims(tf.transpose(gt_masks, [2, 0, 1]), -1)
    # Pick the right mask for each ROI
    roi_masks = tf.gather(transposed_masks, roi_gt_box_assignment)

    # Compute mask targets
    boxes = positive_rois
    if config.USE_MINI_MASK:
        # Transform ROI coordinates from normalized image space
        # to normalized mini-mask space.
        y1, x1, y2, x2 = tf.split(positive_rois, 4, axis=1)
        gt_y1, gt_x1, gt_y2, gt_x2 = tf.split(roi_gt_boxes, 4, axis=1)
        gt_h = gt_y2 - gt_y1
        gt_w = gt_x2 - gt_x1
        y1 = (y1 - gt_y1) / gt_h
        x1 = (x1 - gt_x1) / gt_w
        y2 = (y2 - gt_y1) / gt_h
        x2 = (x2 - gt_x1) / gt_w
        boxes = tf.concat([y1, x1, y2, x2], 1)
    box_ids = tf.range(0, tf.shape(roi_masks)[0])
    masks = tf.image.crop_and_resize(tf.cast(roi_masks, tf.float32), boxes,
                                     box_ids,
                                     config.MASK_SHAPE)
    # Remove the extra dimension from masks.
    masks = tf.squeeze(masks, axis=3)

    # Threshold mask pixels at 0.5 to have GT masks be 0 or 1 to use with
    # binary cross entropy loss.
    masks = tf.round(masks)

    # Append negative ROIs and pad bbox deltas and masks that
    # are not used for negative ROIs with zeros.
    rois = tf.concat([positive_rois, negative_rois], axis=0)
    N = tf.shape(negative_rois)[0]
    P = tf.maximum(config.TRAIN_ROIS_PER_IMAGE - tf.shape(rois)[0], 0)
    rois = tf.pad(rois, [(0, P), (0, 0)])
    roi_gt_boxes = tf.pad(roi_gt_boxes, [(0, N + P), (0, 0)])
    roi_gt_class_ids = tf.pad(roi_gt_class_ids, [(0, N + P)])
    deltas = tf.pad(deltas, [(0, N + P), (0, 0)])
    masks = tf.pad(masks, [[0, N + P], (0, 0), (0, 0)])

    return rois, roi_gt_class_ids, deltas, masks


class DetectionTargetLayer(KE.Layer):
    """Subsamples proposals and generates target box refinement, class_ids,
    and masks for each.

    Inputs:
    proposals: [batch, N, (y1, x1, y2, x2)] in normalized coordinates. Might
               be zero padded if there are not enough proposals.
    gt_class_ids: [batch, MAX_GT_INSTANCES] Integer class IDs.
    gt_boxes: [batch, MAX_GT_INSTANCES, (y1, x1, y2, x2)] in normalized
              coordinates.
    gt_masks: [batch, height, width, MAX_GT_INSTANCES] of boolean type

    Returns: Target ROIs and corresponding class IDs, bounding box shifts,
    and masks.
    rois: [batch, TRAIN_ROIS_PER_IMAGE, (y1, x1, y2, x2)] in normalized
          coordinates
    target_class_ids: [batch, TRAIN_ROIS_PER_IMAGE]. Integer class IDs.
    target_deltas: [batch, TRAIN_ROIS_PER_IMAGE, (dy, dx, log(dh), log(dw)]
    target_mask: [batch, TRAIN_ROIS_PER_IMAGE, height, width]
                 Masks cropped to bbox boundaries and resized to neural
                 network output size.

    Note: Returned arrays might be zero padded if not enough target ROIs.
    """

    def __init__(self, config, **kwargs):
        super(DetectionTargetLayer, self).__init__(**kwargs)
        self.config = config

    def call(self, inputs):
        proposals = inputs[0]
        gt_class_ids = inputs[1]
        gt_boxes = inputs[2]
        gt_masks = inputs[3]

        # Slice the batch and run a graph for each slice
        # TODO: Rename target_bbox to target_deltas for clarity
        names = ["rois", "target_class_ids", "target_bbox", "target_mask"]
        outputs = utils.batch_slice(
            [proposals, gt_class_ids, gt_boxes, gt_masks],
            lambda w, x, y, z: detection_targets_graph(
                w, x, y, z, self.config),
            self.config.IMAGES_PER_GPU, names=names)
        return outputs

    def compute_output_shape(self, input_shape):
        return [
            (None, self.config.TRAIN_ROIS_PER_IMAGE, 4),  # rois
            (None, self.config.TRAIN_ROIS_PER_IMAGE),  # class_ids
            (None, self.config.TRAIN_ROIS_PER_IMAGE, 4),  # deltas
            (None, self.config.TRAIN_ROIS_PER_IMAGE, self.config.MASK_SHAPE[0],
             self.config.MASK_SHAPE[1])  # masks
        ]

    def compute_mask(self, inputs, mask=None):
        return [None, None, None, None]


############################################################
#  Detection Layer
############################################################

def refine_detections_graph(rois, probs, deltas, window, config):
    """Refine classified proposals and filter overlaps and return final
    detections.

    Inputs:
        rois: [N, (y1, x1, y2, x2)] in normalized coordinates
        probs: [N, num_classes]. Class probabilities.
        deltas: [N, num_classes, (dy, dx, log(dh), log(dw))]. Class-specific
                bounding box deltas.
        window: (y1, x1, y2, x2) in normalized coordinates. The part of the image
            that contains the image excluding the padding.

    Returns detections shaped: [num_detections, (y1, x1, y2, x2, class_id, score)] where
        coordinates are normalized.
    """
    # Class IDs per ROI
    class_ids = tf.argmax(probs, axis=1, output_type=tf.int32)
    # Class probability of the top class of each ROI
    indices = tf.stack([tf.range(probs.shape[0]), class_ids], axis=1)
    class_scores = tf.gather_nd(probs, indices)
    # Class-specific bounding box deltas
    deltas_specific = tf.gather_nd(deltas, indices)
    # Apply bounding box deltas
    # Shape: [boxes, (y1, x1, y2, x2)] in normalized coordinates
    refined_rois = apply_box_deltas_graph(
        rois, deltas_specific * config.BBOX_STD_DEV)
    # Clip boxes to image window
    refined_rois = clip_boxes_graph(refined_rois, window)

    # TODO: Filter out boxes with zero area

    # Filter out background boxes
    keep = tf.where(class_ids > 0)[:, 0]
    # Filter out low confidence boxes
    if config.DETECTION_MIN_CONFIDENCE:
        conf_keep = tf.where(class_scores >= config.DETECTION_MIN_CONFIDENCE)[:, 0]
        keep = tf.sets.set_intersection(tf.expand_dims(keep, 0),
                                        tf.expand_dims(conf_keep, 0))
        keep = tf.sparse_tensor_to_dense(keep)[0]

    # Apply per-class NMS
    # 1. Prepare variables
    pre_nms_class_ids = tf.gather(class_ids, keep)
    pre_nms_scores = tf.gather(class_scores, keep)
    pre_nms_rois = tf.gather(refined_rois,   keep)
    unique_pre_nms_class_ids = tf.unique(pre_nms_class_ids)[0]

    def nms_keep_map(class_id):
        """Apply Non-Maximum Suppression on ROIs of the given class."""
        # Indices of ROIs of the given class
        ixs = tf.where(tf.equal(pre_nms_class_ids, class_id))[:, 0]
        # Apply NMS
        class_keep = tf.image.non_max_suppression(
                tf.gather(pre_nms_rois, ixs),
                tf.gather(pre_nms_scores, ixs),
                max_output_size=config.DETECTION_MAX_INSTANCES,
                iou_threshold=config.DETECTION_NMS_THRESHOLD)
        # Map indices
        class_keep = tf.gather(keep, tf.gather(ixs, class_keep))
        # Pad with -1 so returned tensors have the same shape
        gap = config.DETECTION_MAX_INSTANCES - tf.shape(class_keep)[0]
        class_keep = tf.pad(class_keep, [(0, gap)],
                            mode='CONSTANT', constant_values=-1)
        # Set shape so map_fn() can infer result shape
        class_keep.set_shape([config.DETECTION_MAX_INSTANCES])
        return class_keep

    # 2. Map over class IDs
    nms_keep = tf.map_fn(nms_keep_map, unique_pre_nms_class_ids,
                         dtype=tf.int64)
    # 3. Merge results into one list, and remove -1 padding
    nms_keep = tf.reshape(nms_keep, [-1])
    nms_keep = tf.gather(nms_keep, tf.where(nms_keep > -1)[:, 0])
    # 4. Compute intersection between keep and nms_keep
    keep = tf.sets.set_intersection(tf.expand_dims(keep, 0),
                                    tf.expand_dims(nms_keep, 0))
    keep = tf.sparse_tensor_to_dense(keep)[0]
    # Keep top detections
    roi_count = config.DETECTION_MAX_INSTANCES
    class_scores_keep = tf.gather(class_scores, keep)
    num_keep = tf.minimum(tf.shape(class_scores_keep)[0], roi_count)
    top_ids = tf.nn.top_k(class_scores_keep, k=num_keep, sorted=True)[1]
    keep = tf.gather(keep, top_ids)

    # Arrange output as [N, (y1, x1, y2, x2, class_id, score)]
    # Coordinates are normalized.
    detections = tf.concat([
        tf.gather(refined_rois, keep),
        tf.to_float(tf.gather(class_ids, keep))[..., tf.newaxis],
        tf.gather(class_scores, keep)[..., tf.newaxis]
        ], axis=1)

    # Pad with zeros if detections < DETECTION_MAX_INSTANCES
    gap = config.DETECTION_MAX_INSTANCES - tf.shape(detections)[0]
    detections = tf.pad(detections, [(0, gap), (0, 0)], "CONSTANT")
    return detections


class DetectionLayer(KE.Layer):
    """Takes classified proposal boxes and their bounding box deltas and
    returns the final detection boxes.

    Returns:
    [batch, num_detections, (y1, x1, y2, x2, class_id, class_score)] where
    coordinates are normalized.
    """

    def __init__(self, config=None, **kwargs):
        super(DetectionLayer, self).__init__(**kwargs)
        self.config = config

    def call(self, inputs):
        rois = inputs[0]
        mrcnn_class = inputs[1]
        mrcnn_bbox = inputs[2]
        image_meta = inputs[3]

        # Get windows of images in normalized coordinates. Windows are the area
        # in the image that excludes the padding.
        # Use the shape of the first image in the batch to normalize the window
        # because we know that all images get resized to the same size.
        m = parse_image_meta_graph(image_meta)
        image_shape = m['image_shape'][0]
        window = norm_boxes_graph(m['window'], image_shape[:2])

        # Run detection refinement graph on each item in the batch
        detections_batch = utils.batch_slice(
            [rois, mrcnn_class, mrcnn_bbox, window],
            lambda x, y, w, z: refine_detections_graph(x, y, w, z, self.config),
            self.config.IMAGES_PER_GPU)

        # Reshape output
        # [batch, num_detections, (y1, x1, y2, x2, class_id, class_score)] in
        # normalized coordinates
        return tf.reshape(
            detections_batch,
            [self.config.BATCH_SIZE, self.config.DETECTION_MAX_INSTANCES, 6])

    def compute_output_shape(self, input_shape):
        return (None, self.config.DETECTION_MAX_INSTANCES, 6)


############################################################
#  Region Proposal Network (RPN)
############################################################

def rpn_graph(feature_map, anchors_per_location, anchor_stride):
    """Builds the computation graph of Region Proposal Network.

    feature_map: backbone features [batch, height, width, depth]
    anchors_per_location: number of anchors per pixel in the feature map
    anchor_stride: Controls the density of anchors. Typically 1 (anchors for
                   every pixel in the feature map), or 2 (every other pixel).

    Returns:
        
        _class_logits: [batch, H * W * anchors_per_location, 2] Anchor classifier logits (before softmax)
        rpn_probs: [batch, H * W * anchors_per_location, 2] Anchor classifier probabilities.
        rpn_bbox: [batch, H * W * anchors_per_location, (dy, dx, log(dh), log(dw))] Deltas to be
                  applied to anchors.
    """
    # TODO: check if stride of 2 causes alignment issues if the feature map
    # is not even.
    # Shared convolutional base of the RPN
    print("feature_map_rpn: {}".format(feature_map.get_shape().as_list()))
    print("feature_map_rpn2: {}".format(type(feature_map)))
    print("feature_map: {}".format(feature_map))

    shared = KL.Conv2D(512, (3, 3), padding='same', activation='relu',
                       strides=anchor_stride,
                       name='rpn_conv_shared')(feature_map)

    # Anchor Score. [batch, height, width, anchors per location * 2].
    x = KL.Conv2D(2 * anchors_per_location, (1, 1), padding='valid',
                  activation='linear', name='rpn_class_raw')(shared)

    # Reshape to [batch, anchors, 2]
    rpn_class_logits = KL.Lambda(
        lambda t: tf.reshape(t, [tf.shape(t)[0], -1, 2]))(x)

    # Softmax on last dimension of BG/FG.
    rpn_probs = KL.Activation(
        "softmax", name="rpn_class_xxx")(rpn_class_logits)

    # Bounding box refinement. [batch, H, W, anchors per location * depth]
    # where depth is [x, y, log(w), log(h)]
    x = KL.Conv2D(anchors_per_location * 4, (1, 1), padding="valid",
                  activation='linear', name='rpn_bbox_pred')(shared)

    # Reshape to [batch, anchors, 4]
    rpn_bbox = KL.Lambda(lambda t: tf.reshape(t, [tf.shape(t)[0], -1, 4]))(x)
    print("rpn_class_logits_rpn: {}".format(rpn_class_logits.get_shape().as_list()))
    print("rpn_probs: {}".format(rpn_probs.get_shape().as_list()))
    print("rpn_bbox: {}".format(rpn_bbox.get_shape().as_list()))
    return [rpn_class_logits, rpn_probs, rpn_bbox]


def build_rpn_model(anchor_stride, anchors_per_location, depth):
    """Builds a Keras model of the Region Proposal Network.
    It wraps the RPN graph so it can be used multiple times with shared
    weights.

    anchors_per_location: number of anchors per pixel in the feature map
    anchor_stride: Controls the density of anchors. Typically 1 (anchors for
                   every pixel in the feature map), or 2 (every other pixel).
    depth: Depth of the backbone feature map.

    Returns a Keras Model object. The model outputs, when called, are:
    rpn_class_logits: [batch, H * W * anchors_per_location, 2] Anchor classifier logits (before softmax)
    rpn_probs: [batch, H * W * anchors_per_location, 2] Anchor classifier probabilities.
    rpn_bbox: [batch, H * W * anchors_per_location, (dy, dx, log(dh), log(dw))] Deltas to be
                applied to anchors.
    """
    input_feature_map = KL.Input(shape=[None, None, depth],
                                 name="input_rpn_feature_map")
    outputs = rpn_graph(input_feature_map, anchors_per_location, anchor_stride)
    return KM.Model([input_feature_map], outputs, name="rpn_model")


############################################################
#  Feature Pyramid Network Heads
############################################################

def fpn_classifier_graph(rois, feature_maps, image_meta,
                         pool_size, num_classes, train_bn=True,
                         fc_layers_size=1024):
    """Builds the computation graph of the feature pyramid network classifier
    and regressor heads.

    rois: [batch, num_rois, (y1, x1, y2, x2)] Proposal boxes in normalized
          coordinates.
    feature_maps: List of feature maps from different layers of the pyramid,
                  [P2, P3, P4, P5]. Each has a different resolution.
    image_meta: [batch, (meta data)] Image details. See compose_image_meta()
    pool_size: The width of the square feature map generated from ROI Pooling.
    num_classes: number of classes, which determines the depth of the results
    train_bn: Boolean. Train or freeze Batch Norm layers
    fc_layers_size: Size of the 2 FC layers

    Returns:
        logits: [batch, num_rois, NUM_CLASSES] classifier logits (before softmax)
        probs: [batch, num_rois, NUM_CLASSES] classifier probabilities
        bbox_deltas: [batch, num_rois, NUM_CLASSES, (dy, dx, log(dh), log(dw))] Deltas to apply to
                     proposal boxes
    """
    # ROI Pooling
    # Shape: [batch, num_rois, POOL_SIZE, POOL_SIZE, channels]
    x = PyramidROIAlign([pool_size, pool_size],
                        name="roi_align_classifier")([rois, image_meta] + feature_maps)
    # Two 1024 FC layers (implemented with Conv2D for consistency)
    x = KL.TimeDistributed(KL.Conv2D(fc_layers_size, (pool_size, pool_size), padding="valid"),
                           name="mrcnn_class_conv1")(x)
    x = KL.TimeDistributed(BatchNorm(), name='mrcnn_class_bn1')(x, training=train_bn)
    x = KL.Activation('relu')(x)
    x = KL.TimeDistributed(KL.Conv2D(fc_layers_size, (1, 1)),
                           name="mrcnn_class_conv2")(x)
    x = KL.TimeDistributed(BatchNorm(), name='mrcnn_class_bn2')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    shared = KL.Lambda(lambda x: K.squeeze(K.squeeze(x, 3), 2),
                       name="pool_squeeze")(x)

    # Classifier head
    mrcnn_class_logits = KL.TimeDistributed(KL.Dense(num_classes),
                                            name='mrcnn_class_logits')(shared)
    mrcnn_probs = KL.TimeDistributed(KL.Activation("softmax"),
                                     name="mrcnn_class")(mrcnn_class_logits)

    # BBox head
    # [batch, num_rois, NUM_CLASSES * (dy, dx, log(dh), log(dw))]
    x = KL.TimeDistributed(KL.Dense(num_classes * 4, activation='linear'),
                           name='mrcnn_bbox_fc')(shared)
    # Reshape to [batch, num_rois, NUM_CLASSES, (dy, dx, log(dh), log(dw))]
    s = K.int_shape(x)
    mrcnn_bbox = KL.Reshape((s[1], num_classes, 4), name="mrcnn_bbox")(x)

    return mrcnn_class_logits, mrcnn_probs, mrcnn_bbox


def build_fpn_mask_graph(rois, feature_maps, image_meta,
                         pool_size, num_classes, train_bn=True):
    """Builds the computation graph of the mask head of Feature Pyramid Network.

    rois: [batch, num_rois, (y1, x1, y2, x2)] Proposal boxes in normalized
          coordinates.
    feature_maps: List of feature maps from different layers of the pyramid,
                  [P2, P3, P4, P5]. Each has a different resolution.
    image_meta: [batch, (meta data)] Image details. See compose_image_meta()
    pool_size: The width of the square feature map generated from ROI Pooling.
    num_classes: number of classes, which determines the depth of the results
    train_bn: Boolean. Train or freeze Batch Norm layers

    Returns: Masks [batch, num_rois, MASK_POOL_SIZE, MASK_POOL_SIZE, NUM_CLASSES]
    """
    # ROI Pooling
    # Shape: [batch, num_rois, MASK_POOL_SIZE, MASK_POOL_SIZE, channels]
    x = PyramidROIAlign([pool_size, pool_size],
                        name="roi_align_mask")([rois, image_meta] + feature_maps)

    # Conv layers
    x = KL.TimeDistributed(KL.Conv2D(128, (3, 3), padding="same"),
                           name="mrcnn_mask_conv1")(x)
    x = KL.TimeDistributed(BatchNorm(),
                           name='mrcnn_mask_bn1')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    x = KL.TimeDistributed(KL.Conv2D(128, (3, 3), padding="same"),
                           name="mrcnn_mask_conv2")(x)
    x = KL.TimeDistributed(BatchNorm(),
                           name='mrcnn_mask_bn2')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    x = KL.TimeDistributed(KL.Conv2D(128, (3, 3), padding="same"),
                           name="mrcnn_mask_conv3")(x)
    x = KL.TimeDistributed(BatchNorm(),
                           name='mrcnn_mask_bn3')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    x = KL.TimeDistributed(KL.Conv2D(128, (3, 3), padding="same"),
                           name="mrcnn_mask_conv4")(x)
    x = KL.TimeDistributed(BatchNorm(),
                           name='mrcnn_mask_bn4')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    x = KL.TimeDistributed(KL.Conv2DTranspose(128, (2, 2), strides=2, activation="relu"),
                           name="mrcnn_mask_deconv")(x)
    x = KL.TimeDistributed(KL.Conv2D(num_classes, (1, 1), strides=1, activation="sigmoid"),
                           name="mrcnn_mask")(x)
    return x


############################################################
#  Loss Functions
############################################################

def smooth_l1_loss(y_true, y_pred):
    """Implements Smooth-L1 loss.
    y_true and y_pred are typically: [N, 4], but could be any shape.
    """
    diff = K.abs(y_true - y_pred)
    less_than_one = K.cast(K.less(diff, 1.0), "float32")
    loss = (less_than_one * 0.5 * diff**2) + (1 - less_than_one) * (diff - 0.5)
    return loss


def rpn_class_loss_graph(rpn_match, rpn_class_logits):
    """RPN anchor classifier loss.

    rpn_match: [batch, anchors, 1]. Anchor match type. 1=positive,
               -1=negative, 0=neutral anchor.
    rpn_class_logits: [batch, anchors, 2]. RPN classifier logits for BG/FG.
    """
    # Squeeze last dim to simplify
    rpn_match = tf.squeeze(rpn_match, -1)
    # Get anchor classes. Convert the -1/+1 match to 0/1 values.
    anchor_class = K.cast(K.equal(rpn_match, 1), tf.int32)
    # Positive and Negative anchors contribute to the loss,
    # but neutral anchors (match value = 0) don't.
    indices = tf.where(K.not_equal(rpn_match, 0))
    # Pick rows that contribute to the loss and filter out the rest.
    rpn_class_logits = tf.gather_nd(rpn_class_logits, indices)
    anchor_class = tf.gather_nd(anchor_class, indices)
    # Cross entropy loss
    loss = K.sparse_categorical_crossentropy(target=anchor_class,
                                             output=rpn_class_logits,
                                             from_logits=True)
    loss = K.switch(tf.size(loss) > 0, K.mean(loss), tf.constant(0.0))
    return loss


def rpn_bbox_loss_graph(config, target_bbox, rpn_match, rpn_bbox):
    """Return the RPN bounding box loss graph.

    config: the model config object.
    target_bbox: [batch, max positive anchors, (dy, dx, log(dh), log(dw))].
        Uses 0 padding to fill in unsed bbox deltas.
    rpn_match: [batch, anchors, 1]. Anchor match type. 1=positive,
               -1=negative, 0=neutral anchor.
    rpn_bbox: [batch, anchors, (dy, dx, log(dh), log(dw))]
    """
    # Positive anchors contribute to the loss, but negative and
    # neutral anchors (match value of 0 or -1) don't.
    rpn_match = K.squeeze(rpn_match, -1)
    indices = tf.where(K.equal(rpn_match, 1))

    # Pick bbox deltas that contribute to the loss
    rpn_bbox = tf.gather_nd(rpn_bbox, indices)

    # Trim target bounding box deltas to the same length as rpn_bbox.
    batch_counts = K.sum(K.cast(K.equal(rpn_match, 1), tf.int32), axis=1)
    target_bbox = batch_pack_graph(target_bbox, batch_counts,
                                   config.IMAGES_PER_GPU)

    loss = smooth_l1_loss(target_bbox, rpn_bbox)
    
    loss = K.switch(tf.size(loss) > 0, K.mean(loss), tf.constant(0.0))
    return loss


def mrcnn_class_loss_graph(target_class_ids, pred_class_logits,
                           active_class_ids):
    """Loss for the classifier head of Mask RCNN.

    target_class_ids: [batch, num_rois]. Integer class IDs. Uses zero
        padding to fill in the array.
    pred_class_logits: [batch, num_rois, num_classes]
    active_class_ids: [batch, num_classes]. Has a value of 1 for
        classes that are in the dataset of the image, and 0
        for classes that are not in the dataset.
    """
    # During model building, Keras calls this function with
    # target_class_ids of type float32. Unclear why. Cast it
    # to int to get around it.
    target_class_ids = tf.cast(target_class_ids, 'int64')

    # Find predictions of classes that are not in the dataset.
    pred_class_ids = tf.argmax(pred_class_logits, axis=2)
    # TODO: Update this line to work with batch > 1. Right now it assumes all
    #       images in a batch have the same active_class_ids
    pred_active = tf.gather(active_class_ids[0], pred_class_ids)

    # Loss
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=target_class_ids, logits=pred_class_logits)

    # Erase losses of predictions of classes that are not in the active
    # classes of the image.
    loss = loss * pred_active

    # Computer loss mean. Use only predictions that contribute
    # to the loss to get a correct mean.
    loss = tf.reduce_sum(loss) / tf.reduce_sum(pred_active)
    return loss


def mrcnn_bbox_loss_graph(target_bbox, target_class_ids, pred_bbox):
    """Loss for Mask R-CNN bounding box refinement.

    target_bbox: [batch, num_rois, (dy, dx, log(dh), log(dw))]
    target_class_ids: [batch, num_rois]. Integer class IDs.
    pred_bbox: [batch, num_rois, num_classes, (dy, dx, log(dh), log(dw))]
    """
    # Reshape to merge batch and roi dimensions for simplicity.
    target_class_ids = K.reshape(target_class_ids, (-1,))
    target_bbox = K.reshape(target_bbox, (-1, 4))
    pred_bbox = K.reshape(pred_bbox, (-1, K.int_shape(pred_bbox)[2], 4))

    # Only positive ROIs contribute to the loss. And only
    # the right class_id of each ROI. Get their indices.
    positive_roi_ix = tf.where(target_class_ids > 0)[:, 0]
    positive_roi_class_ids = tf.cast(
        tf.gather(target_class_ids, positive_roi_ix), tf.int64)
    indices = tf.stack([positive_roi_ix, positive_roi_class_ids], axis=1)

    # Gather the deltas (predicted and true) that contribute to loss
    target_bbox = tf.gather(target_bbox, positive_roi_ix)
    pred_bbox = tf.gather_nd(pred_bbox, indices)

    # Smooth-L1 Loss
    loss = K.switch(tf.size(target_bbox) > 0,
                    smooth_l1_loss(y_true=target_bbox, y_pred=pred_bbox),
                    tf.constant(0.0))
    loss = K.mean(loss)
    return loss


def mrcnn_mask_loss_graph(target_masks, target_class_ids, pred_masks):
    """Mask binary cross-entropy loss for the masks head.

    target_masks: [batch, num_rois, height, width].
        A float32 tensor of values 0 or 1. Uses zero padding to fill array.
    target_class_ids: [batch, num_rois]. Integer class IDs. Zero padded.
    pred_masks: [batch, proposals, height, width, num_classes] float32 tensor
                with values from 0 to 1.
    """
    # Reshape for simplicity. Merge first two dimensions into one.
    target_class_ids = K.reshape(target_class_ids, (-1,))
    mask_shape = tf.shape(target_masks)
    target_masks = K.reshape(target_masks, (-1, mask_shape[2], mask_shape[3]))
    pred_shape = tf.shape(pred_masks)
    pred_masks = K.reshape(pred_masks,
                           (-1, pred_shape[2], pred_shape[3], pred_shape[4]))
    # Permute predicted masks to [N, num_classes, height, width]
    pred_masks = tf.transpose(pred_masks, [0, 3, 1, 2])

    # Only positive ROIs contribute to the loss. And only
    # the class specific mask of each ROI.
    positive_ix = tf.where(target_class_ids > 0)[:, 0]
    positive_class_ids = tf.cast(
        tf.gather(target_class_ids, positive_ix), tf.int64)
    indices = tf.stack([positive_ix, positive_class_ids], axis=1)

    # Gather the masks (predicted and true) that contribute to loss
    y_true = tf.gather(target_masks, positive_ix)
    y_pred = tf.gather_nd(pred_masks, indices)

    # Compute binary cross entropy. If no positive ROIs, then return 0.
    # shape: [batch, roi, num_classes]
    loss = K.switch(tf.size(y_true) > 0,
                    K.binary_crossentropy(target=y_true, output=y_pred),
                    tf.constant(0.0))
    loss = K.mean(loss)
    return loss


############################################################
#  Data Generator
############################################################

def load_image_gt(dataset, config, image_id, augment=False, augmentation=None,
                  use_mini_mask=False):
    """Load and return ground truth data for an image (image, mask, bounding boxes).

    augment: (deprecated. Use augmentation instead). If true, apply random
        image augmentation. Currently, only horizontal flipping is offered.
    augmentation: Optional. An imgaug (https://github.com/aleju/imgaug) augmentation.
        For example, passing imgaug.augmenters.Fliplr(0.5) flips images
        right/left 50% of the time.
    use_mini_mask: If False, returns full-size masks that are the same height
        and width as the original image. These can be big, for example
        1024x1024x100 (for 100 instances). Mini masks are smaller, typically,
        224x224 and are generated by extracting the bounding box of the
        object and resizing it to MINI_MASK_SHAPE.

    Returns:
    image: [height, width, 3]
    shape: the original shape of the image before resizing and cropping.
    class_ids: [instance_count] Integer class IDs
    bbox: [instance_count, (y1, x1, y2, x2)]
    mask: [height, width, instance_count]. The height and width are those
        of the image unless use_mini_mask is True, in which case they are
        defined in MINI_MASK_SHAPE.
    """
    # Load image and mask
    image = dataset.load_image(image_id)
    mask, class_ids = dataset.load_mask(image_id)
    original_shape = image.shape
    image, window, scale, padding, crop = utils.resize_image(
        image,
        min_dim=config.IMAGE_MIN_DIM,
        min_scale=config.IMAGE_MIN_SCALE,
        max_dim=config.IMAGE_MAX_DIM,
        mode=config.IMAGE_RESIZE_MODE)
    mask = utils.resize_mask(mask, scale, padding, crop)

    # Random horizontal flips.
    # TODO: will be removed in a future update in favor of augmentation
    if augment:
        logging.warning("'augment' is deprecated. Use 'augmentation' instead.")
        if random.randint(0, 1):
            image = np.fliplr(image)
            mask = np.fliplr(mask)

    # Augmentation
    # This requires the imgaug lib (https://github.com/aleju/imgaug)
    if augmentation:
        import imgaug

        # Augmenters that are safe to apply to masks
        # Some, such as Affine, have settings that make them unsafe, so always
        # test your augmentation on masks
        MASK_AUGMENTERS = ["Sequential", "SomeOf", "OneOf", "Sometimes",
                           "Fliplr", "Flipud", "CropAndPad",
                           "Affine", "PiecewiseAffine"]

        def hook(images, augmenter, parents, default):
            """Determines which augmenters to apply to masks."""
            return augmenter.__class__.__name__ in MASK_AUGMENTERS

        # Store shapes before augmentation to compare
        image_shape = image.shape
        mask_shape = mask.shape
        # Make augmenters deterministic to apply similarly to images and masks
        det = augmentation.to_deterministic()
        image = det.augment_image(image)
        # Change mask to np.uint8 because imgaug doesn't support np.bool
        mask = det.augment_image(mask.astype(np.uint8),
                                 hooks=imgaug.HooksImages(activator=hook))
        # Verify that shapes didn't change
        assert image.shape == image_shape, "Augmentation shouldn't change image size"
        assert mask.shape == mask_shape, "Augmentation shouldn't change mask size"
        # Change mask back to bool
        mask = mask.astype(np.bool)

    # Note that some boxes might be all zeros if the corresponding mask got cropped out.
    # and here is to filter them out
    _idx = np.sum(mask, axis=(0, 1)) > 0
    mask = mask[:, :, _idx]
    class_ids = class_ids[_idx]
    # Bounding boxes. Note that some boxes might be all zeros
    # if the corresponding mask got cropped out.
    # bbox: [num_instances, (y1, x1, y2, x2)]
    bbox = utils.extract_bboxes(mask)

    # Active classes
    # Different datasets have different classes, so track the
    # classes supported in the dataset of this image.
    active_class_ids = np.zeros([dataset.num_classes], dtype=np.int32)
    source_class_ids = dataset.source_class_ids[dataset.image_info[image_id]["source"]]
    active_class_ids[source_class_ids] = 1

    # Resize masks to smaller size to reduce memory usage
    if use_mini_mask:
        mask = utils.minimize_mask(bbox, mask, config.MINI_MASK_SHAPE)

    # Image meta data
    image_meta = compose_image_meta(image_id, original_shape, image.shape,
                                    window, scale, active_class_ids)

    return image, image_meta, class_ids, bbox, mask


def build_detection_targets(rpn_rois, gt_class_ids, gt_boxes, gt_masks, config):
    """Generate targets for training Stage 2 classifier and mask heads.
    This is not used in normal training. It's useful for debugging or to train
    the Mask RCNN heads without using the RPN head.

    Inputs:
    rpn_rois: [N, (y1, x1, y2, x2)] proposal boxes.
    gt_class_ids: [instance count] Integer class IDs
    gt_boxes: [instance count, (y1, x1, y2, x2)]
    gt_masks: [height, width, instance count] Ground truth masks. Can be full
              size or mini-masks.

    Returns:
    rois: [TRAIN_ROIS_PER_IMAGE, (y1, x1, y2, x2)]
    class_ids: [TRAIN_ROIS_PER_IMAGE]. Integer class IDs.
    bboxes: [TRAIN_ROIS_PER_IMAGE, NUM_CLASSES, (y, x, log(h), log(w))]. Class-specific
            bbox refinements.
    masks: [TRAIN_ROIS_PER_IMAGE, height, width, NUM_CLASSES). Class specific masks cropped
           to bbox boundaries and resized to neural network output size.
    """
    assert rpn_rois.shape[0] > 0
    assert gt_class_ids.dtype == np.int32, "Expected int but got {}".format(
        gt_class_ids.dtype)
    assert gt_boxes.dtype == np.int32, "Expected int but got {}".format(
        gt_boxes.dtype)
    assert gt_masks.dtype == np.bool_, "Expected bool but got {}".format(
        gt_masks.dtype)

    # It's common to add GT Boxes to ROIs but we don't do that here because
    # according to XinLei Chen's paper, it doesn't help.

    # Trim empty padding in gt_boxes and gt_masks parts
    instance_ids = np.where(gt_class_ids > 0)[0]
    assert instance_ids.shape[0] > 0, "Image must contain instances."
    gt_class_ids = gt_class_ids[instance_ids]
    gt_boxes = gt_boxes[instance_ids]
    gt_masks = gt_masks[:, :, instance_ids]

    # Compute areas of ROIs and ground truth boxes.
    rpn_roi_area = (rpn_rois[:, 2] - rpn_rois[:, 0]) * \
        (rpn_rois[:, 3] - rpn_rois[:, 1])
    gt_box_area = (gt_boxes[:, 2] - gt_boxes[:, 0]) * \
        (gt_boxes[:, 3] - gt_boxes[:, 1])

    # Compute overlaps [rpn_rois, gt_boxes]
    overlaps = np.zeros((rpn_rois.shape[0], gt_boxes.shape[0]))
    for i in range(overlaps.shape[1]):
        gt = gt_boxes[i]
        overlaps[:, i] = utils.compute_iou(
            gt, rpn_rois, gt_box_area[i], rpn_roi_area)

    # Assign ROIs to GT boxes
    rpn_roi_iou_argmax = np.argmax(overlaps, axis=1)
    rpn_roi_iou_max = overlaps[np.arange(
        overlaps.shape[0]), rpn_roi_iou_argmax]
    # GT box assigned to each ROI
    rpn_roi_gt_boxes = gt_boxes[rpn_roi_iou_argmax]
    rpn_roi_gt_class_ids = gt_class_ids[rpn_roi_iou_argmax]

    # Positive ROIs are those with >= 0.5 IoU with a GT box.
    fg_ids = np.where(rpn_roi_iou_max > 0.5)[0]

    # Negative ROIs are those with max IoU 0.1-0.5 (hard example mining)
    # TODO: To hard example mine or not to hard example mine, that's the question
    # bg_ids = np.where((rpn_roi_iou_max >= 0.1) & (rpn_roi_iou_max < 0.5))[0]
    bg_ids = np.where(rpn_roi_iou_max < 0.5)[0]

    # Subsample ROIs. Aim for 33% foreground.
    # FG
    fg_roi_count = int(config.TRAIN_ROIS_PER_IMAGE * config.ROI_POSITIVE_RATIO)
    if fg_ids.shape[0] > fg_roi_count:
        keep_fg_ids = np.random.choice(fg_ids, fg_roi_count, replace=False)
    else:
        keep_fg_ids = fg_ids
    # BG
    remaining = config.TRAIN_ROIS_PER_IMAGE - keep_fg_ids.shape[0]
    if bg_ids.shape[0] > remaining:
        keep_bg_ids = np.random.choice(bg_ids, remaining, replace=False)
    else:
        keep_bg_ids = bg_ids
    # Combine indices of ROIs to keep
    keep = np.concatenate([keep_fg_ids, keep_bg_ids])
    # Need more?
    remaining = config.TRAIN_ROIS_PER_IMAGE - keep.shape[0]
    if remaining > 0:
        # Looks like we don't have enough samples to maintain the desired
        # balance. Reduce requirements and fill in the rest. This is
        # likely different from the Mask RCNN paper.

        # There is a small chance we have neither fg nor bg samples.
        if keep.shape[0] == 0:
            # Pick bg regions with easier IoU threshold
            bg_ids = np.where(rpn_roi_iou_max < 0.5)[0]
            assert bg_ids.shape[0] >= remaining
            keep_bg_ids = np.random.choice(bg_ids, remaining, replace=False)
            assert keep_bg_ids.shape[0] == remaining
            keep = np.concatenate([keep, keep_bg_ids])
        else:
            # Fill the rest with repeated bg rois.
            keep_extra_ids = np.random.choice(
                keep_bg_ids, remaining, replace=True)
            keep = np.concatenate([keep, keep_extra_ids])
    assert keep.shape[0] == config.TRAIN_ROIS_PER_IMAGE, \
        "keep doesn't match ROI batch size {}, {}".format(
            keep.shape[0], config.TRAIN_ROIS_PER_IMAGE)

    # Reset the gt boxes assigned to BG ROIs.
    rpn_roi_gt_boxes[keep_bg_ids, :] = 0
    rpn_roi_gt_class_ids[keep_bg_ids] = 0

    # For each kept ROI, assign a class_id, and for FG ROIs also add bbox refinement.
    rois = rpn_rois[keep]
    roi_gt_boxes = rpn_roi_gt_boxes[keep]
    roi_gt_class_ids = rpn_roi_gt_class_ids[keep]
    roi_gt_assignment = rpn_roi_iou_argmax[keep]

    # Class-aware bbox deltas. [y, x, log(h), log(w)]
    bboxes = np.zeros((config.TRAIN_ROIS_PER_IMAGE,
                       config.NUM_CLASSES, 4), dtype=np.float32)
    pos_ids = np.where(roi_gt_class_ids > 0)[0]
    bboxes[pos_ids, roi_gt_class_ids[pos_ids]] = utils.box_refinement(
        rois[pos_ids], roi_gt_boxes[pos_ids, :4])
    # Normalize bbox refinements
    bboxes /= config.BBOX_STD_DEV

    # Generate class-specific target masks
    masks = np.zeros((config.TRAIN_ROIS_PER_IMAGE, config.MASK_SHAPE[0], config.MASK_SHAPE[1], config.NUM_CLASSES),
                     dtype=np.float32)
    for i in pos_ids:
        class_id = roi_gt_class_ids[i]
        assert class_id > 0, "class id must be greater than 0"
        gt_id = roi_gt_assignment[i]
        class_mask = gt_masks[:, :, gt_id]

        if config.USE_MINI_MASK:
            # Create a mask placeholder, the size of the image
            placeholder = np.zeros(config.IMAGE_SHAPE[:2], dtype=bool)
            # GT box
            gt_y1, gt_x1, gt_y2, gt_x2 = gt_boxes[gt_id]
            gt_w = gt_x2 - gt_x1
            gt_h = gt_y2 - gt_y1
            # Resize mini mask to size of GT box
            placeholder[gt_y1:gt_y2, gt_x1:gt_x2] = \
                np.round(utils.resize(class_mask, (gt_h, gt_w))).astype(bool)
            # Place the mini batch in the placeholder
            class_mask = placeholder

        # Pick part of the mask and resize it
        y1, x1, y2, x2 = rois[i].astype(np.int32)
        m = class_mask[y1:y2, x1:x2]
        mask = utils.resize(m, config.MASK_SHAPE)
        masks[i, :, :, class_id] = mask

    return rois, roi_gt_class_ids, bboxes, masks


def build_rpn_targets(image_shape, anchors, gt_class_ids, gt_boxes, config):
    """Given the anchors and GT boxes, compute overlaps and identify positive
    anchors and deltas to refine them to match their corresponding GT boxes.

    anchors: [num_anchors, (y1, x1, y2, x2)]
    gt_class_ids: [num_gt_boxes] Integer class IDs.
    gt_boxes: [num_gt_boxes, (y1, x1, y2, x2)]

    Returns:
    rpn_match: [N] (int32) matches between anchors and GT boxes.
               1 = positive anchor, -1 = negative anchor, 0 = neutral
    rpn_bbox: [N, (dy, dx, log(dh), log(dw))] Anchor bbox deltas.
    """
    # RPN Match: 1 = positive anchor, -1 = negative anchor, 0 = neutral
    rpn_match = np.zeros([anchors.shape[0]], dtype=np.int32)
    # RPN bounding boxes: [max anchors per image, (dy, dx, log(dh), log(dw))]
    rpn_bbox = np.zeros((config.RPN_TRAIN_ANCHORS_PER_IMAGE, 4))

    # Handle COCO crowds
    # A crowd box in COCO is a bounding box around several instances. Exclude
    # them from training. A crowd box is given a negative class ID.
    crowd_ix = np.where(gt_class_ids < 0)[0]
    if crowd_ix.shape[0] > 0:
        # Filter out crowds from ground truth class IDs and boxes
        non_crowd_ix = np.where(gt_class_ids > 0)[0]
        crowd_boxes = gt_boxes[crowd_ix]
        gt_class_ids = gt_class_ids[non_crowd_ix]
        gt_boxes = gt_boxes[non_crowd_ix]
        # Compute overlaps with crowd boxes [anchors, crowds]
        crowd_overlaps = utils.compute_overlaps(anchors, crowd_boxes)
        crowd_iou_max = np.amax(crowd_overlaps, axis=1)
        no_crowd_bool = (crowd_iou_max < 0.001)
    else:
        # All anchors don't intersect a crowd
        no_crowd_bool = np.ones([anchors.shape[0]], dtype=bool)

    # Compute overlaps [num_anchors, num_gt_boxes]
    overlaps = utils.compute_overlaps(anchors, gt_boxes)

    # Match anchors to GT Boxes
    # If an anchor overlaps a GT box with IoU >= 0.7 then it's positive.
    # If an anchor overlaps a GT box with IoU < 0.3 then it's negative.
    # Neutral anchors are those that don't match the conditions above,
    # and they don't influence the loss function.
    # However, don't keep any GT box unmatched (rare, but happens). Instead,
    # match it to the closest anchor (even if its max IoU is < 0.3).
    #
    # 1. Set negative anchors first. They get overwritten below if a GT box is
    # matched to them. Skip boxes in crowd areas.
    anchor_iou_argmax = np.argmax(overlaps, axis=1)
    anchor_iou_max = overlaps[np.arange(overlaps.shape[0]), anchor_iou_argmax]
    rpn_match[(anchor_iou_max < 0.3) & (no_crowd_bool)] = -1
    # 2. Set an anchor for each GT box (regardless of IoU value).
    # If multiple anchors have the same IoU match all of them
    gt_iou_argmax = np.argwhere(overlaps == np.max(overlaps, axis=0))[:,0]
    rpn_match[gt_iou_argmax] = 1
    # 3. Set anchors with high overlap as positive.
    rpn_match[anchor_iou_max >= 0.7] = 1

    # Subsample to balance positive and negative anchors
    # Don't let positives be more than half the anchors
    ids = np.where(rpn_match == 1)[0]
    extra = len(ids) - (config.RPN_TRAIN_ANCHORS_PER_IMAGE // 2)
    if extra > 0:
        # Reset the extra ones to neutral
        ids = np.random.choice(ids, extra, replace=False)
        rpn_match[ids] = 0
    # Same for negative proposals
    ids = np.where(rpn_match == -1)[0]
    extra = len(ids) - (config.RPN_TRAIN_ANCHORS_PER_IMAGE -
                        np.sum(rpn_match == 1))
    if extra > 0:
        # Rest the extra ones to neutral
        ids = np.random.choice(ids, extra, replace=False)
        rpn_match[ids] = 0

    # For positive anchors, compute shift and scale needed to transform them
    # to match the corresponding GT boxes.
    ids = np.where(rpn_match == 1)[0]
    ix = 0  # index into rpn_bbox
    # TODO: use box_refinement() rather than duplicating the code here
    for i, a in zip(ids, anchors[ids]):
        # Closest gt box (it might have IoU < 0.7)
        gt = gt_boxes[anchor_iou_argmax[i]]

        # Convert coordinates to center plus width/height.
        # GT Box
        gt_h = gt[2] - gt[0]
        gt_w = gt[3] - gt[1]
        gt_center_y = gt[0] + 0.5 * gt_h
        gt_center_x = gt[1] + 0.5 * gt_w
        # Anchor
        a_h = a[2] - a[0]
        a_w = a[3] - a[1]
        a_center_y = a[0] + 0.5 * a_h
        a_center_x = a[1] + 0.5 * a_w

        # Compute the bbox refinement that the RPN should predict.
        rpn_bbox[ix] = [
            (gt_center_y - a_center_y) / a_h,
            (gt_center_x - a_center_x) / a_w,
            np.log(gt_h / a_h),
            np.log(gt_w / a_w),
        ]
        # Normalize
        rpn_bbox[ix] /= config.RPN_BBOX_STD_DEV
        ix += 1

    return rpn_match, rpn_bbox


def generate_random_rois(image_shape, count, gt_class_ids, gt_boxes):
    """Generates ROI proposals similar to what a region proposal network
    would generate.

    image_shape: [Height, Width, Depth]
    count: Number of ROIs to generate
    gt_class_ids: [N] Integer ground truth class IDs
    gt_boxes: [N, (y1, x1, y2, x2)] Ground truth boxes in pixels.

    Returns: [count, (y1, x1, y2, x2)] ROI boxes in pixels.
    """
    # placeholder
    rois = np.zeros((count, 4), dtype=np.int32)

    # Generate random ROIs around GT boxes (90% of count)
    rois_per_box = int(0.9 * count / gt_boxes.shape[0])
    for i in range(gt_boxes.shape[0]):
        gt_y1, gt_x1, gt_y2, gt_x2 = gt_boxes[i]
        h = gt_y2 - gt_y1
        w = gt_x2 - gt_x1
        # random boundaries
        r_y1 = max(gt_y1 - h, 0)
        r_y2 = min(gt_y2 + h, image_shape[0])
        r_x1 = max(gt_x1 - w, 0)
        r_x2 = min(gt_x2 + w, image_shape[1])

        # To avoid generating boxes with zero area, we generate double what
        # we need and filter out the extra. If we get fewer valid boxes
        # than we need, we loop and try again.
        while True:
            y1y2 = np.random.randint(r_y1, r_y2, (rois_per_box * 2, 2))
            x1x2 = np.random.randint(r_x1, r_x2, (rois_per_box * 2, 2))
            # Filter out zero area boxes
            threshold = 1
            y1y2 = y1y2[np.abs(y1y2[:, 0] - y1y2[:, 1]) >=
                        threshold][:rois_per_box]
            x1x2 = x1x2[np.abs(x1x2[:, 0] - x1x2[:, 1]) >=
                        threshold][:rois_per_box]
            if y1y2.shape[0] == rois_per_box and x1x2.shape[0] == rois_per_box:
                break

        # Sort on axis 1 to ensure x1 <= x2 and y1 <= y2 and then reshape
        # into x1, y1, x2, y2 order
        x1, x2 = np.split(np.sort(x1x2, axis=1), 2, axis=1)
        y1, y2 = np.split(np.sort(y1y2, axis=1), 2, axis=1)
        box_rois = np.hstack([y1, x1, y2, x2])
        rois[rois_per_box * i:rois_per_box * (i + 1)] = box_rois

    # Generate random ROIs anywhere in the image (10% of count)
    remaining_count = count - (rois_per_box * gt_boxes.shape[0])
    # To avoid generating boxes with zero area, we generate double what
    # we need and filter out the extra. If we get fewer valid boxes
    # than we need, we loop and try again.
    while True:
        y1y2 = np.random.randint(0, image_shape[0], (remaining_count * 2, 2))
        x1x2 = np.random.randint(0, image_shape[1], (remaining_count * 2, 2))
        # Filter out zero area boxes
        threshold = 1
        y1y2 = y1y2[np.abs(y1y2[:, 0] - y1y2[:, 1]) >=
                    threshold][:remaining_count]
        x1x2 = x1x2[np.abs(x1x2[:, 0] - x1x2[:, 1]) >=
                    threshold][:remaining_count]
        if y1y2.shape[0] == remaining_count and x1x2.shape[0] == remaining_count:
            break

    # Sort on axis 1 to ensure x1 <= x2 and y1 <= y2 and then reshape
    # into x1, y1, x2, y2 order
    x1, x2 = np.split(np.sort(x1x2, axis=1), 2, axis=1)
    y1, y2 = np.split(np.sort(y1y2, axis=1), 2, axis=1)
    global_rois = np.hstack([y1, x1, y2, x2])
    rois[-remaining_count:] = global_rois
    return rois


def data_generator(dataset, config, shuffle=True, augment=False, augmentation=None,
                   random_rois=0, batch_size=1, detection_targets=False,
                   no_augmentation_sources=None, rnd_state=0):
    """A generator that returns images and corresponding target class ids,
    bounding box deltas, and masks.

    dataset: The Dataset object to pick data from
    config: The model config object
    shuffle: If True, shuffles the samples before every epoch
    augment: (deprecated. Use augmentation instead). If true, apply random
        image augmentation. Currently, only horizontal flipping is offered.
    augmentation: Optional. An imgaug (https://github.com/aleju/imgaug) augmentation.
        For example, passing imgaug.augmenters.Fliplr(0.5) flips images
        right/left 50% of the time.
    random_rois: If > 0 then generate proposals to be used to train the
                 network classifier and mask heads. Useful if training
                 the Mask RCNN part without the RPN.
    batch_size: How many images to return in each call
    detection_targets: If True, generate detection targets (class IDs, bbox
        deltas, and masks). Typically for debugging or visualizations because
        in trainig detection targets are generated by DetectionTargetLayer.
    no_augmentation_sources: Optional. List of sources to exclude for
        augmentation. A source is string that identifies a dataset and is
        defined in the Dataset class.

    Returns a Python generator. Upon calling next() on it, the
    generator returns two lists, inputs and outputs. The contents
    of the lists differs depending on the received arguments:
    inputs list:
    - images: [batch, H, W, C]
    - image_meta: [batch, (meta data)] Image details. See compose_image_meta()
    - rpn_match: [batch, N] Integer (1=positive anchor, -1=negative, 0=neutral)
    - rpn_bbox: [batch, N, (dy, dx, log(dh), log(dw))] Anchor bbox deltas.
    - gt_class_ids: [batch, MAX_GT_INSTANCES] Integer class IDs
    - gt_boxes: [batch, MAX_GT_INSTANCES, (y1, x1, y2, x2)]
    - gt_masks: [batch, height, width, MAX_GT_INSTANCES]. The height and width
                are those of the image unless use_mini_mask is True, in which
                case they are defined in MINI_MASK_SHAPE.

    outputs list: Usually empty in regular training. But if detection_targets
        is True then the outputs list contains target class_ids, bbox deltas,
        and masks.
    """
    b = 0  # batch item index
    
    instance_index = -1
    image_index = -1
    rnd_state_sec_views = 0
    random_shuffle = np.random.RandomState(seed=rnd_state)
#     instance_ids = np.copy(list(dataset.instance_map.keys()))
    view_ids = np.copy(list(dataset.view_map.keys()))
    instance_ids = np.copy(list(dataset.instance_map.keys()))
    error_count = 0
    if not config.USE_RPN_ROIS:
        random_rois = config.POST_NMS_ROIS_TRAINING
    no_augmentation_sources = no_augmentation_sources or []

    # Anchors
    # [anchor_count, (y1, x1, y2, x2)]
    backbone_shapes = compute_backbone_shapes(config, config.IMAGE_SHAPE)
    anchors = utils.generate_pyramid_anchors(config.RPN_ANCHOR_SCALES,
                                             config.RPN_ANCHOR_RATIOS,
                                             backbone_shapes,
                                             config.BACKBONE_STRIDES,
                                             config.RPN_ANCHOR_STRIDE)

    # Keras requires a generator to run indefinitely.
    while True:
        try:
            # Increment index to pick next image. Shuffle if at the start of an epoch.
            instance_index = (instance_index + 1) % len(instance_ids)
            image_index = (image_index + 1) % len(view_ids)
#             if shuffle and instance_index == 0:
#                 random_shuffle.shuffle(instance_ids)
            if shuffle and image_index == 0:
                rnd_state_sec_views += 1
                random_shuffle.shuffle(view_ids)

            # Get GT bounding boxes and masks for image.
            view_id = view_ids[image_index]
#             instance_id = instance_ids[instance_index]
#             image_ids = dataset.load_view(config.NUM_VIEWS, main_image=view_id)
            image_ids = dataset.load_view(config.NUM_VIEWS, main_image=view_id, rnd_state=rnd_state_sec_views)
            # skip instance if it has to few views (return of load_views=None)
            if not image_ids:
                continue
            actual_num_views = len(image_ids)
            image_id = image_ids[0]
            # If the image source is not to be augmented pass None as augmentation
            if dataset.image_info[image_id]['source'] in no_augmentation_sources:
                _, image_meta, gt_class_ids, gt_boxes, gt_masks = \
                load_image_gt(dataset, config, image_id, augment=augment,
                              augmentation=None,
                              use_mini_mask=config.USE_MINI_MASK)
                image = []
                Rcam = []
                for i in range(actual_num_views):
                    image_t, _, _, _, _ = \
                    load_image_gt(dataset, config, image_ids[i], augment=augment,
                                  augmentation=None,
                                  use_mini_mask=config.USE_MINI_MASK)
                    image.append(image_t)
                    Rcam.append(dataset.load_R(image_ids[i]))
                Kmat = dataset.K
            else:
                _, image_meta, gt_class_ids, gt_boxes, gt_masks = \
                    load_image_gt(dataset, config, image_id, augment=augment,
                                augmentation=augmentation,
                                use_mini_mask=config.USE_MINI_MASK)
                image = []
                Rcam = []
                
                for i in range(actual_num_views):
                    
                    image_t, _, _, _, _ = \
                    load_image_gt(dataset, config, image_ids[i], augment=augment,
                                augmentation=augmentation,
                                use_mini_mask=config.USE_MINI_MASK)
                    image.append(image_t)
                    Rcam.append(dataset.load_R(image_ids[i]))
                Kmat = dataset.K
                
            # Skip images that have no instances. This can happen in cases
            # where we train on a subset of classes and the image doesn't
            # have any of the classes we care about.
            if not np.any(gt_class_ids > 0):
                continue

            # RPN Targets
            rpn_match, rpn_bbox = build_rpn_targets(image[0].shape, anchors,
                                                    gt_class_ids, gt_boxes, config)
#             print("image shape in generator: {}".format(image[0].shape))
            assert np.any(rpn_match), "no rpn_match in generator"
            # Mask R-CNN Targets
            if random_rois:
                rpn_rois = generate_random_rois(
                    image[0].shape, random_rois, gt_class_ids, gt_boxes)
                if detection_targets:
                    rois, mrcnn_class_ids, mrcnn_bbox, mrcnn_mask =\
                        build_detection_targets(
                            rpn_rois, gt_class_ids, gt_boxes, gt_masks, config)
            # Init batch arrays
            if b == 0:
                batch_image_meta = np.zeros(
                    (batch_size,) + image_meta.shape, dtype=image_meta.dtype)
                batch_rpn_match = np.zeros(
                    [batch_size, anchors.shape[0], 1], dtype=rpn_match.dtype)
                batch_rpn_bbox = np.zeros(
                    [batch_size, config.RPN_TRAIN_ANCHORS_PER_IMAGE, 4], dtype=rpn_bbox.dtype)
                batch_images = np.zeros(
                    (batch_size, actual_num_views,) + image[0].shape, dtype=np.float32)
                batch_gt_class_ids = np.zeros(
                    (batch_size, config.MAX_GT_INSTANCES), dtype=np.int32)
                batch_gt_boxes = np.zeros(
                    (batch_size, config.MAX_GT_INSTANCES, 4), dtype=np.int32)
                batch_gt_masks = np.zeros(
                    (batch_size, gt_masks.shape[0], gt_masks.shape[1],
                     config.MAX_GT_INSTANCES), dtype=gt_masks.dtype)
                batch_gt_R = np.zeros(
                    (batch_size, actual_num_views, 3, 4), dtype=np.float32)
                batch_gt_Kmat = np.zeros(
                    (batch_size, 3, 3), dtype=np.float32)
                if random_rois:
                    batch_rpn_rois = np.zeros(
                        (batch_size, rpn_rois.shape[0], 4), dtype=rpn_rois.dtype)
                    if detection_targets:
                        batch_rois = np.zeros(
                            (batch_size,) + rois.shape, dtype=rois.dtype)
                        batch_mrcnn_class_ids = np.zeros(
                            (batch_size,) + mrcnn_class_ids.shape, dtype=mrcnn_class_ids.dtype)
                        batch_mrcnn_bbox = np.zeros(
                            (batch_size,) + mrcnn_bbox.shape, dtype=mrcnn_bbox.dtype)
                        batch_mrcnn_mask = np.zeros(
                            (batch_size,) + mrcnn_mask.shape, dtype=mrcnn_mask.dtype)
            # If more instances than fits in the array, sub-sample from them.
            if gt_boxes.shape[0] > config.MAX_GT_INSTANCES:
                ids = np.random.choice(
                    np.arange(gt_boxes.shape[0]), config.MAX_GT_INSTANCES, replace=False)
                gt_class_ids = gt_class_ids[ids]
                gt_boxes = gt_boxes[ids]
                gt_masks = gt_masks[:, :, ids]

            # Add to batch
            batch_image_meta[b] = image_meta
            batch_rpn_match[b] = rpn_match[:, np.newaxis]
            batch_rpn_bbox[b] = rpn_bbox
            for i in range(actual_num_views):
                batch_images[b, i] = mold_image(image[i].astype(np.float32), config)
                batch_gt_R[b,i] = Rcam[i]
            batch_gt_class_ids[b, :gt_class_ids.shape[0]] = gt_class_ids
            batch_gt_boxes[b, :gt_boxes.shape[0]] = gt_boxes
            batch_gt_masks[b, :, :, :gt_masks.shape[-1]] = gt_masks
            batch_gt_Kmat[b] = Kmat
            if random_rois:
                batch_rpn_rois[b] = rpn_rois
                if detection_targets:
                    batch_rois[b] = rois
                    batch_mrcnn_class_ids[b] = mrcnn_class_ids
                    batch_mrcnn_bbox[b] = mrcnn_bbox
                    batch_mrcnn_mask[b] = mrcnn_mask
            b += 1
             
#             ########## DEBUG
#             print("batch_gt_boxes: {}".format(batch_gt_boxes))
#             ##########

            # Batch full?
            if b >= batch_size:
                inputs = [batch_images, batch_image_meta, batch_rpn_match, batch_rpn_bbox,
                          batch_gt_class_ids, batch_gt_boxes, batch_gt_masks, batch_gt_R, batch_gt_Kmat]
                outputs = []
                
                if random_rois:
                    inputs.extend([batch_rpn_rois])
                    if detection_targets:
                        inputs.extend([batch_rois])
                        # Keras requires that output and targets have the same number of dimensions
                        batch_mrcnn_class_ids = np.expand_dims(
                            batch_mrcnn_class_ids, -1)
                        outputs.extend(
                            [batch_mrcnn_class_ids, batch_mrcnn_bbox, batch_mrcnn_mask])

                yield inputs, outputs

                # start a new batch
                b = 0
        except (GeneratorExit, KeyboardInterrupt):
            raise
        except:
            # Log it and skip the image
#             logging.exception("Error processing image {}".format(
#                 dataset.image_info[image_id]))
            print("Error occured at image_id: {}".format(image_id))
            error_count += 1
            if error_count > 5:
                raise


############################################################
#  MaskRCNN Class
############################################################

class MaskRCNN():
    """Encapsulates the Mask RCNN model functionality.

    The actual Keras model is in the keras_model property.
    """

    def __init__(self, mode, config, model_dir):
        """
        mode: Either "training" or "inference"
        config: A Sub-class of the Config class
        model_dir: Directory to save training logs and trained weights
        """
        assert mode in ['training', 'inference']
        self.mode = mode
        self.config = config
        self.model_dir = model_dir
        self.set_log_dir()
        self.keras_model = self.build(mode=mode, config=config)

    def build(self, mode, config):
        """Build Mask R-CNN architecture.
            input_shape: The shape of the input image.
            mode: Either "training" or "inference". The inputs and
                outputs of the model differ accordingly.
        """
        assert mode in ['training', 'inference']

        # Image size must be dividable by 2 multiple times
        h, w = config.IMAGE_SHAPE[:2]
        if h / 2**6 != int(h / 2**6) or w / 2**6 != int(w / 2**6):
            raise Exception("Image size must be dividable by 2 at least 6 times "
                            "to avoid fractions when downscaling and upscaling."
                            "For example, use 256, 320, 384, 448, 512, ... etc. ")

        # Inputs
        input_image = KL.Input(
            shape=[None, config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1], config.IMAGE_SHAPE[2]], name="input_image")
        input_image_meta = KL.Input(shape=[config.IMAGE_META_SIZE],
                                    name="input_image_meta")
        input_R = KL.Input(shape=[None, 3, 4],
                              name="input_R")
        input_Kmat = KL.Input(shape=[3, 3],
                              name="input_Kmat")
        if mode == "training":
            # RPN GT
            input_rpn_match = KL.Input(
                shape=[None, 1], name="input_rpn_match", dtype=tf.int32)
            input_rpn_bbox = KL.Input(
                shape=[None, 4], name="input_rpn_bbox", dtype=tf.float32)

            # Detection GT (class IDs, bounding boxes, and masks)
            # 1. GT Class IDs (zero padded)
            input_gt_class_ids = KL.Input(
                shape=[None], name="input_gt_class_ids", dtype=tf.int32)
            # 2. GT Boxes in pixels (zero padded)
            # [batch, MAX_GT_INSTANCES, (y1, x1, y2, x2)] in image coordinates
            input_gt_boxes = KL.Input(
                shape=[None, 4], name="input_gt_boxes", dtype=tf.float32)
            # Normalize coordinates
            gt_boxes = KL.Lambda(lambda x: norm_boxes_graph(
                x, K.shape(input_image)[2:4]))(input_gt_boxes)
            # 3. GT Masks (zero padded)
            # [batch, height, width, MAX_GT_INSTANCES]
            if config.USE_MINI_MASK:
                input_gt_masks = KL.Input(
                    shape=[config.MINI_MASK_SHAPE[0],
                           config.MINI_MASK_SHAPE[1], None],
                    name="input_gt_masks", dtype=bool)
            else:
                input_gt_masks = KL.Input(
                    shape=[config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1], None],
                    name="input_gt_masks", dtype=bool)
        elif mode == "inference":
            # Anchors in normalized coordinates
            input_anchors = KL.Input(shape=[None, 4], name="input_anchors")
        batch_size, num_views, h, w, chn = input_image.get_shape().as_list()
        num_views = config.NUM_VIEWS
                           
#         P2, P3, P4, P5, P6 = build_resnet_fpn(input_image, config)                   
        
        input_image_0 = KL.Lambda(lambda x: x[:,0,:,:,:], name="pred_image_selection")(input_image)

        backbone_model = view_merger_model(input_image.get_shape().as_list()[1:], config)
        
        P2, P3, P4, P5, P6 = backbone_model(input_image)

        #PG5 = transformer_encoder(P5, input_R, input_Kmat, config, training=True)
#         P2 = KL.Lambda(lambda x: tf.expand_dims(x, axis=1))(P2)
#         P3 = KL.Lambda(lambda x: tf.expand_dims(x, axis=1))(P3)
#         P4 = KL.Lambda(lambda x: tf.expand_dims(x, axis=1))(P4)
#         P5 = KL.Lambda(lambda x: tf.expand_dims(x, axis=1))(P5)
#         P6 = KL.Lambda(lambda x: tf.expand_dims(x, axis=1))(P6)
#         for i in range(1, num_views):
#             P2_t, P3_t, P4_t, P5_t, P6_t = backbone_model(KL.Lambda(lambda x: x[:,i,:,:,:])(input_image))
#             print("finished")
#             P2_t = KL.Lambda(lambda x: tf.expand_dims(x, axis=1))(P2_t)
#             P3_t = KL.Lambda(lambda x: tf.expand_dims(x, axis=1))(P3_t)
#             P4_t = KL.Lambda(lambda x: tf.expand_dims(x, axis=1))(P4_t)
#             P5_t = KL.Lambda(lambda x: tf.expand_dims(x, axis=1))(P5_t)
#             P6_t = KL.Lambda(lambda x: tf.expand_dims(x, axis=1))(P6_t)
#             print("P2_t: {}".format(P2_t.get_shape().as_list()))
#             P2 = KL.Lambda(lambda x: tf.concat([x[0], x[1]], axis=1))([P2, P2_t])
#             P3 = KL.Lambda(lambda x: tf.concat([x[0], x[1]], axis=1))([P3, P3_t])
#             P4 = KL.Lambda(lambda x: tf.concat([x[0], x[1]], axis=1))([P4, P4_t])
#             P5 = KL.Lambda(lambda x: tf.concat([x[0], x[1]], axis=1))([P5, P5_t])
#             P6 = KL.Lambda(lambda x: tf.concat([x[0], x[1]], axis=1))([P6, P6_t])
        
        
        
        
        #P2, P3, P4, P5, P6 = view_merger_model(input_image_0.get_shape().as_list()[1:], config)(input_image)
        
        print("P2_shape: {}".format(P2.get_shape().as_list()))
#         P2_t = KL.TimeDistributed(KL.Conv2D(64, (3,3), padding='same'), name='fpn_thin_2')(P2)
#         P3_t = KL.TimeDistributed(KL.Conv2D(64, (3,3), padding='same'), name='fpn_thin_3')(P3)
#         P4_t = KL.TimeDistributed(KL.Conv2D(64, (3,3), padding='same'), name='fpn_thin_4')(P4)
#         P5_t = KL.TimeDistributed(KL.Conv2D(64, (3,3), padding='same'), name='fpn_thin_5')(P5)
#         P6_t = KL.TimeDistributed(KL.Conv2D(64, (3,3), padding='same'), name='fpn_thin_6')(P6)
        P2_1 = P2
        P3_1 = P3
        P4_1 = P4
        P5_1 = P5
        P6_1 = P6

        P2 = KL.Lambda(lambda x: x[:,0,:,:,:])(P2)
        P3 = KL.Lambda(lambda x: x[:,0,:,:,:])(P3)
        P4 = KL.Lambda(lambda x: x[:,0,:,:,:])(P4)
        P5 = KL.Lambda(lambda x: x[:,0,:,:,:])(P5)
        P6 = KL.Lambda(lambda x: x[:,0,:,:,:])(P6)

#         # Project feature maps into 3D-Space
        PG2, grid_pos = KL.Lambda(lambda x: unproj_feat(x, self.config), name="unproj_P2")([P2_1, input_R, input_Kmat])
        PG3, grid_pos = KL.Lambda(lambda x: unproj_feat(x, self.config), name="unproj_P3")([P3_1, input_R, input_Kmat])
        PG4, grid_pos = KL.Lambda(lambda x: unproj_feat(x, self.config), name="unproj_P4")([P4_1, input_R, input_Kmat])
        PG5, grid_pos = KL.Lambda(lambda x: unproj_feat(x, self.config), name="unproj_P5")([P5_1, input_R, input_Kmat])
        PG6, grid_pos = KL.Lambda(lambda x: unproj_feat(x, self.config), name="unproj_P6")([P6_1, input_R, input_Kmat])
        print("PG2_shape: {}".format(PG2.get_shape().as_list()))

        PG2 = grid_reas(PG2, "grid_reas_P2", config)
        PG3 = grid_reas(PG3, "grid_reas_P3", config)
        PG4 = grid_reas(PG4, "grid_reas_P4", config)
        PG5 = grid_reas(PG5, "grid_reas_P5", config)
        PG6 = grid_reas(PG6, "grid_reas_P6", config)
        print("PG2_shape: {}".format(PG2.get_shape().as_list()))
        PG2 = KL.Lambda(lambda x: proj_grid(x, config=self.config, proj_size=160), name="projs_PG2")([PG2, grid_pos, input_R, input_Kmat])
        PG3 = KL.Lambda(lambda x: proj_grid(x, config=self.config, proj_size=80), name="projs_PG3")([PG3, grid_pos, input_R, input_Kmat])
        PG4 = KL.Lambda(lambda x: proj_grid(x, config=self.config, proj_size=40), name="projs_PG4")([PG4, grid_pos, input_R, input_Kmat])
        PG5 = KL.Lambda(lambda x: proj_grid(x, config=self.config, proj_size=20), name="projs_PG5")([PG5, grid_pos, input_R, input_Kmat])
        PG6 = KL.Lambda(lambda x: proj_grid(x, config=self.config, proj_size=10), name="projs_PG6")([PG6, grid_pos, input_R, input_Kmat])
        print("PG2_shape: {}".format(PG2.get_shape().as_list()))

        PG2 = KL.Lambda(lambda x: tf.transpose(x, [0, 4, 2, 3, 1]))(PG2)
        PG3 = KL.Lambda(lambda x: tf.transpose(x, [0, 4, 2, 3, 1]))(PG3)
        PG4 = KL.Lambda(lambda x: tf.transpose(x, [0, 4, 2, 3, 1]))(PG4)
        PG5 = KL.Lambda(lambda x: tf.transpose(x, [0, 4, 2, 3, 1]))(PG5)
        PG6 = KL.Lambda(lambda x: tf.transpose(x, [0, 4, 2, 3, 1]))(PG6)
        print("PG2_shape: {}".format(PG2.get_shape().as_list()))

        PG2_intermediate = KL.Lambda(lambda x: x[:,:,:,:,0])(PG2)
        print("PG2_shape: {}".format(PG2.get_shape().as_list()))

        PG2_intermediate = KL.Lambda(lambda x: x[:,:,:,:,0])(PG2)
        PG2 = depth_sampling(PG2, config, name='grid_reas_depth_PG2')
        PG3 = depth_sampling(PG3, config, name='grid_reas_depth_PG3')
        PG4 = depth_sampling(PG4, config, name='grid_reas_depth_PG4')
        PG5 = depth_sampling(PG5, config, name='grid_reas_depth_PG5')
        PG6 = depth_sampling(PG6, config, name='grid_reas_depth_PG6')


#         PG2 = KL.ConvLSTM2D(64, (1,1), padding='same', return_sequences=False, name='grid_reas_depth_PG2')(PG2)
#         PG3 = KL.ConvLSTM2D(64, (1,1), padding='same', return_sequences=False, name='grid_reas_depth_PG3')(PG3)
#         PG4 = KL.ConvLSTM2D(64, (1,1), padding='same', return_sequences=False, name='grid_reas_depth_PG4')(PG4)
#         PG5 = KL.ConvLSTM2D(64, (1,1), padding='same', return_sequences=False, name='grid_reas_depth_PG5')(PG5)
#         PG6 = KL.ConvLSTM2D(64, (1,1), padding='same', return_sequences=False, name='grid_reas_depth_PG6')(PG6)
#         print("PG2_shape: {}".format(PG2.get_shape().as_list()))

        PG2 = KL.Lambda(lambda x: tf.transpose(x[:,:,:,:,0], [0, 2, 3, 1]))(PG2)
        PG3 = KL.Lambda(lambda x: tf.transpose(x[:,:,:,:,0], [0, 2, 3, 1]))(PG3)
        PG4 = KL.Lambda(lambda x: tf.transpose(x[:,:,:,:,0], [0, 2, 3, 1]))(PG4)
        PG5 = KL.Lambda(lambda x: tf.transpose(x[:,:,:,:,0], [0, 2, 3, 1]))(PG5)
        PG6 = KL.Lambda(lambda x: tf.transpose(x[:,:,:,:,0], [0, 2, 3, 1]))(PG6)

        if not config.VANILLA:
            print("recurrent mrcnn")
            PG2 = KL.Add()([PG2, P2])
            PG3 = KL.Add()([PG3, P3])
            PG4 = KL.Add()([PG4, P4])
            PG5 = KL.Add()([PG5, P5])
            PG6 = KL.Add()([PG6, P6])
            #P5 = KL.Add()([PG5, P5])

            # rpn_feature_maps = [P2, P3, P4, P5, P6]
            # mrcnn_feature_maps = [P2, P3, P4, P5]
            rpn_feature_maps = [PG2, PG3, PG4, PG5, PG6]
            mrcnn_feature_maps = [PG2, PG3, PG4, PG5]
        else:
            print("vanilla mrcnn")
            rpn_feature_maps = [P2, P3, P4, P5, P6]
            mrcnn_feature_maps = [P2, P3, P4, P5]

        print("PG2_shape: {}".format(PG2.get_shape().as_list()))
        print("PG6_shape: {}".format(PG6.get_shape().as_list()))
            
                # Anchors
        if mode == "training":
            anchors = self.get_anchors(config.IMAGE_SHAPE)
            # Duplicate across the batch dimension because Keras requires it
            # TODO: can this be optimized to avoid duplicating the anchors?
            anchors = np.broadcast_to(anchors, (config.BATCH_SIZE,) + anchors.shape)
            # A hack to get around Keras's bad support for constants
            anchors = KL.Lambda(lambda x: tf.Variable(anchors), name="anchors")(input_image_0)
        else:
            anchors = input_anchors

        # RPN Model
        rpn = build_rpn_model(config.RPN_ANCHOR_STRIDE,
                              len(config.RPN_ANCHOR_RATIOS), config.TOP_DOWN_PYRAMID_SIZE)
        # Loop through pyramid layers
        layer_outputs = []  # list of lists
        for p in rpn_feature_maps:
            print(p.get_shape().as_list())
            layer_outputs.append(rpn([p]))
        # Concatenate layer outputs
        # Convert from list of lists of level outputs to list of lists
        # of outputs across levels.
        # e.g. [[a1, b1, c1], [a2, b2, c2]] => [[a1, a2], [b1, b2], [c1, c2]]
        output_names = ["rpn_class_logits", "rpn_class", "rpn_bbox"]
        outputs = list(zip(*layer_outputs))
        outputs = [KL.Concatenate(axis=1, name=n)(list(o))
                   for o, n in zip(outputs, output_names)]

        rpn_class_logits, rpn_class, rpn_bbox = outputs
        # Generate proposals
        # Proposals are [batch, N, (y1, x1, y2, x2)] in normalized coordinates
        # and zero padded.
        proposal_count = config.POST_NMS_ROIS_TRAINING if mode == "training"\
            else config.POST_NMS_ROIS_INFERENCE
        rpn_rois = ProposalLayer(
            proposal_count=proposal_count,
            nms_threshold=config.RPN_NMS_THRESHOLD,
            name="ROI",
            config=config)([rpn_class, rpn_bbox, anchors])

        if mode == "training":
            # Class ID mask to mark class IDs supported by the dataset the image
            # came from.
            active_class_ids = KL.Lambda(
                lambda x: parse_image_meta_graph(x)["active_class_ids"]
                )(input_image_meta)

            if not config.USE_RPN_ROIS:
                # Ignore predicted ROIs and use ROIs provided as an input.
                input_rois = KL.Input(shape=[config.POST_NMS_ROIS_TRAINING, 4],
                                      name="input_roi", dtype=np.float32)
                # Normalize coordinates
                target_rois = KL.Lambda(lambda x: norm_boxes_graph(
                    x, K.shape(input_image_0)[1:3]))(input_rois)
            else:
                target_rois = rpn_rois
            
            # Generate detection targets
            # Subsamples proposals and generates target outputs for training
            # Note that proposal class IDs, gt_boxes, and gt_masks are zero
            # padded. Equally, returned rois and targets are zero padded.
            rois, target_class_ids, target_bbox, target_mask =\
                DetectionTargetLayer(config, name="proposal_targets")([
                    target_rois, input_gt_class_ids, gt_boxes, input_gt_masks])

            # Network Heads
            # TODO: verify that this handles zero padded ROIs
            mrcnn_class_logits, mrcnn_class, mrcnn_bbox =\
                fpn_classifier_graph(rois, mrcnn_feature_maps, input_image_meta,
                                     config.POOL_SIZE, config.NUM_CLASSES,
                                     train_bn=config.TRAIN_BN,
                                     fc_layers_size=config.FPN_CLASSIF_FC_LAYERS_SIZE)

            mrcnn_mask = build_fpn_mask_graph(rois, mrcnn_feature_maps,
                                              input_image_meta,
                                              config.MASK_POOL_SIZE,
                                              config.NUM_CLASSES,
                                              train_bn=config.TRAIN_BN)

            # TODO: clean up (use tf.identify if necessary)
            output_rois = KL.Lambda(lambda x: x * 1, name="output_rois")(rois)
            
            ########## DEBUG
#             print_op = tf.print("rpn_class, any(target_class_ids), target_rois, rois:", {1: rpn_class, 2: tf.keras.backend.any(target_class_ids[0]), 3:target_rois, 4: rois})
#             with tf.control_dependencies([print_op]):
#                 mrcnn_bbox = KL.Lambda(lambda x: x*1.0)(mrcnn_bbox)
            ##########
            
            # Losses
            rpn_class_loss = KL.Lambda(lambda x: rpn_class_loss_graph(*x), name="rpn_class_loss")(
                [input_rpn_match, rpn_class_logits])
            rpn_bbox_loss = KL.Lambda(lambda x: rpn_bbox_loss_graph(config, *x), name="rpn_bbox_loss")(
                [input_rpn_bbox, input_rpn_match, rpn_bbox])
            class_loss = KL.Lambda(lambda x: mrcnn_class_loss_graph(*x), name="mrcnn_class_loss")(
                [target_class_ids, mrcnn_class_logits, active_class_ids])
            bbox_loss = KL.Lambda(lambda x: mrcnn_bbox_loss_graph(*x), name="mrcnn_bbox_loss")(
                [target_bbox, target_class_ids, mrcnn_bbox])
            mask_loss = KL.Lambda(lambda x: mrcnn_mask_loss_graph(*x), name="mrcnn_mask_loss")(
                [target_mask, target_class_ids, mrcnn_mask])

            # Model
            inputs = [input_image, input_image_meta,
                      input_rpn_match, input_rpn_bbox, input_gt_class_ids, input_gt_boxes, input_gt_masks, input_R, input_Kmat]
            if not config.USE_RPN_ROIS:
                inputs.append(input_rois)
            outputs = [rpn_class_logits, rpn_class, rpn_bbox,
                       mrcnn_class_logits, mrcnn_class, mrcnn_bbox, mrcnn_mask,
                       rpn_rois, output_rois,
                       rpn_class_loss, rpn_bbox_loss, class_loss, bbox_loss, mask_loss]
            model = KM.Model(inputs, outputs, name='mask_rcnn')
        else:
            # Network Heads
            # Proposal classifier and BBox regressor heads
            mrcnn_class_logits, mrcnn_class, mrcnn_bbox =\
                fpn_classifier_graph(rpn_rois, mrcnn_feature_maps, input_image_meta,
                                     config.POOL_SIZE, config.NUM_CLASSES,
                                     train_bn=config.TRAIN_BN,
                                     fc_layers_size=config.FPN_CLASSIF_FC_LAYERS_SIZE)

            # Detections
            # output is [batch, num_detections, (y1, x1, y2, x2, class_id, score)] in
            # normalized coordinates
            detections = DetectionLayer(config, name="mrcnn_detection")(
                [rpn_rois, mrcnn_class, mrcnn_bbox, input_image_meta])

            # Create masks for detections
            detection_boxes = KL.Lambda(lambda x: x[..., :4])(detections)
            mrcnn_mask = build_fpn_mask_graph(detection_boxes, mrcnn_feature_maps,
                                              input_image_meta,
                                              config.MASK_POOL_SIZE,
                                              config.NUM_CLASSES,
                                              train_bn=config.TRAIN_BN)

            model = KM.Model([input_image, input_image_meta, input_anchors, input_R, input_Kmat],
                             [detections, mrcnn_class, mrcnn_bbox,
                                 mrcnn_mask, rpn_rois, rpn_class, rpn_bbox, PG2, PG2_intermediate],
                             name='mask_rcnn')

        # Add multi-GPU support.
        if config.GPU_COUNT > 1:
            from mrcnn.parallel_model import ParallelModel
            model = ParallelModel(model, config.GPU_COUNT)

        return model

    def find_last(self):
        """Finds the last checkpoint file of the last trained model in the
        model directory.
        Returns:
            The path of the last checkpoint file
        """
        # Get directory names. Each directory corresponds to a model
        dir_names = next(os.walk(self.model_dir))[1]
        key = self.config.NAME.lower()
        dir_names = filter(lambda f: f.startswith(key), dir_names)
        dir_names = sorted(dir_names)
        if not dir_names:
            import errno
            raise FileNotFoundError(
                errno.ENOENT,
                "Could not find model directory under {}".format(self.model_dir))
        # Pick last directory
        dir_name = os.path.join(self.model_dir, dir_names[-1])
        # Find the last checkpoint
        checkpoints = next(os.walk(dir_name))[2]
        checkpoints = filter(lambda f: f.startswith("mask_rcnn"), checkpoints)
        checkpoints = sorted(checkpoints)
        if not checkpoints:
            import errno
            raise FileNotFoundError(
                errno.ENOENT, "Could not find weight files in {}".format(dir_name))
        checkpoint = os.path.join(dir_name, checkpoints[-1])
        return checkpoint

    def load_weights(self, filepath, by_name=False, exclude=None):
        """Modified version of the corresponding Keras function with
        the addition of multi-GPU support and the ability to exclude
        some layers from loading.
        exclude: list of layer names to exclude
        """
        import h5py
        # Conditional import to support versions of Keras before 2.2
        # TODO: remove in about 6 months (end of 2018)
        try:
            from keras.engine import saving
        except ImportError:
            # Keras before 2.2 used the 'topology' namespace.
            from keras.engine import topology as saving

        if exclude:
            by_name = True

        if h5py is None:
            raise ImportError('`load_weights` requires h5py.')
        f = h5py.File(filepath, mode='r')
        if 'layer_names' not in f.attrs and 'model_weights' in f:
            f = f['model_weights']

        # In multi-GPU training, we wrap the model. Get layers
        # of the inner model because they have the weights.
        keras_model = self.keras_model
        if hasattr(keras_model, "inner_model"):
            layers = keras_model.inner_model.layers
            print("inner model")
        else:
            layers = keras_model.layers
            print("normal model")
#         layers = keras_model.inner_model.layers if hasattr(keras_model, "inner_model")\
#             else keras_model.layers

        # Exclude some layers
        if exclude:
            layers = list(filter(lambda l: l.name not in exclude, layers))
        for layer in layers:
            print(layer.name)
        if by_name:
            print("load_weights_from_hdf5_group_by_name")
            saving.load_weights_from_hdf5_group_by_name(f, layers)
        else:
            saving.load_weights_from_hdf5_group(f, layers)
        if hasattr(f, 'close'):
            f.close()

        # Update the log directory
        self.set_log_dir(filepath)

    def get_imagenet_weights(self):
        """Downloads ImageNet trained weights from Keras.
        Returns path to weights file.
        """
        from keras.utils.data_utils import get_file
        TF_WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/'\
                                 'releases/download/v0.2/'\
                                 'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
        weights_path = get_file('resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
                                TF_WEIGHTS_PATH_NO_TOP,
                                cache_subdir='models',
                                md5_hash='a268eb855778b3df3c7506639542a6af')
        return weights_path

    def compile(self, learning_rate, momentum):
        """Gets the model ready for training. Adds losses, regularization, and
        metrics. Then calls the Keras compile() function.
        """
        # Optimizer object
        optimizer = keras.optimizers.SGD(
            lr=learning_rate, momentum=momentum,
            clipnorm=self.config.GRADIENT_CLIP_NORM)
#         optimizer = keras.optimizers.Adam(
#             lr=learning_rate,
#             clipnorm=self.config.GRADIENT_CLIP_NORM)
        # Add Losses
        # First, clear previously set losses to avoid duplication
        self.keras_model._losses = []
        self.keras_model._per_input_losses = {}
        loss_names = [
            "rpn_class_loss",  "rpn_bbox_loss",
            "mrcnn_class_loss", "mrcnn_bbox_loss", "mrcnn_mask_loss"]
        for name in loss_names:
            layer = self.keras_model.get_layer(name)
            if layer.output in self.keras_model.losses:
                continue
            loss = (
                tf.reduce_mean(layer.output, keepdims=True)
                * self.config.LOSS_WEIGHTS.get(name, 1.))
            self.keras_model.add_loss(loss)

        # Add L2 Regularization
        # Skip gamma and beta weights of batch normalization layers.
        reg_losses = [
            keras.regularizers.l2(self.config.WEIGHT_DECAY)(w) / tf.cast(tf.size(w), tf.float32)
            for w in self.keras_model.trainable_weights
            if 'gamma' not in w.name and 'beta' not in w.name]
        self.keras_model.add_loss(tf.add_n(reg_losses))

        # Compile
        self.keras_model.compile(
            optimizer=optimizer,
            loss=[None] * len(self.keras_model.outputs))

        # Add metrics for losses
        for name in loss_names:
            if name in self.keras_model.metrics_names:
                continue
            layer = self.keras_model.get_layer(name)
            self.keras_model.metrics_names.append(name)
            loss = (
                tf.reduce_mean(layer.output, keepdims=True)
                * self.config.LOSS_WEIGHTS.get(name, 1.))
            self.keras_model.metrics_tensors.append(loss)

    def set_trainable(self, layer_regex, keras_model=None, indent=0, verbose=1):
        """Sets model layers as trainable if their names match
        the given regular expression.
        """
        # Print message on the first call (but not on recursive calls)
        if verbose > 0 and keras_model is None:
            log("Selecting layers to train")

        keras_model = keras_model or self.keras_model

        # In multi-GPU training, we wrap the model. Get layers
        # of the inner model because they have the weights.
        layers = keras_model.inner_model.layers if hasattr(keras_model, "inner_model")\
            else keras_model.layers

        for layer in layers:
            # Is the layer a model?
            if layer.__class__.__name__ == 'Model':
                print("In model: ", layer.name)
                self.set_trainable(
                    layer_regex, keras_model=layer, indent=indent + 4)
                continue
            

            if not layer.weights:
                continue
            # Is it trainable?
            trainable = bool(re.fullmatch(layer_regex, layer.name))
            # Update layer. If layer is a container, update inner layer.
            if layer.__class__.__name__ == 'TimeDistributed':
                layer.layer.trainable = trainable
            else:
                layer.trainable = trainable
            # Print trainable layer names
            if trainable and verbose > 0:
                log("{}{:20}   ({})".format(" " * indent, layer.name,
                                            layer.__class__.__name__))

    def set_log_dir(self, model_path=None):
        """Sets the model log directory and epoch counter.

        model_path: If None, or a format different from what this code uses
            then set a new log directory and start epochs from 0. Otherwise,
            extract the log directory and the epoch counter from the file
            name.
        """
        # Set date and epoch counter as if starting a new model
        self.epoch = 0
        now = datetime.datetime.now()

        # If we have a model path with date and epochs use them
        if model_path:
            # Continue from we left of. Get epoch and date from the file name
            # A sample model path might look like:
            # \path\to\logs\coco20171029T2315\mask_rcnn_coco_0001.h5 (Windows)
            # /path/to/logs/coco20171029T2315/mask_rcnn_coco_0001.h5 (Linux)
            regex = r".*[/\\][\w-]+(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})[/\\]mask\_rcnn\_[\w-]+(\d{4})\.h5"
            m = re.match(regex, model_path)
            if m:
                now = datetime.datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)),
                                        int(m.group(4)), int(m.group(5)))
                # Epoch number in file is 1-based, and in Keras code it's 0-based.
                # So, adjust for that then increment by one to start from the next epoch
                self.epoch = int(m.group(6)) - 1 + 1
                print('Re-starting from epoch %d' % self.epoch)

        # Directory for training logs
        self.log_dir = os.path.join(self.model_dir, "{}{:%Y%m%dT%H%M}".format(
            self.config.NAME.lower(), now))

        # Path to save after each epoch. Include placeholders that get filled by Keras.
        self.checkpoint_path = os.path.join(self.log_dir, "mask_rcnn_{}_*epoch*.h5".format(
            self.config.NAME.lower()))
        self.checkpoint_path = self.checkpoint_path.replace(
            "*epoch*", "{epoch:04d}")

    def train(self, train_dataset, val_dataset, learning_rate, epochs, layers,
              augmentation=None, custom_callbacks=None, no_augmentation_sources=None):
        """Train the model.
        train_dataset, val_dataset: Training and validation Dataset objects.
        learning_rate: The learning rate to train with
        epochs: Number of training epochs. Note that previous training epochs
                are considered to be done alreay, so this actually determines
                the epochs to train in total rather than in this particaular
                call.
        layers: Allows selecting wich layers to train. It can be:
            - A regular expression to match layer names to train
            - One of these predefined values:
              heads: The RPN, classifier and mask heads of the network
              all: All the layers
              3+: Train Resnet stage 3 and up
              4+: Train Resnet stage 4 and up
              5+: Train Resnet stage 5 and up
        augmentation: Optional. An imgaug (https://github.com/aleju/imgaug)
            augmentation. For example, passing imgaug.augmenters.Fliplr(0.5)
            flips images right/left 50% of the time. You can pass complex
            augmentations as well. This augmentation applies 50% of the
            time, and when it does it flips images right/left half the time
            and adds a Gaussian blur with a random sigma in range 0 to 5.

                augmentation = imgaug.augmenters.Sometimes(0.5, [
                    imgaug.augmenters.Fliplr(0.5),
                    imgaug.augmenters.GaussianBlur(sigma=(0.0, 5.0))
                ])
	    custom_callbacks: Optional. Add custom callbacks to be called
	        with the keras fit_generator method. Must be list of type keras.callbacks.
        no_augmentation_sources: Optional. List of sources to exclude for
            augmentation. A source is string that identifies a dataset and is
            defined in the Dataset class.
        """
        assert self.mode == "training", "Create model in training mode."

        # Pre-defined layer regular expressions
        layer_regex = {
            # all layers but the backbone
            "heads": r"(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
            "grid+": r"(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)|(grid_reas\_.*)",
            "grid+-": r"(mrcnn\_.*)|(rpn\_.*)|(grid_reas\_.*)",
            "grid_only": r"(grid_reas\_.*)",
            # From a specific Resnet stage and up
            "3+": r"(res3.*)|(bn3.*)|(res4.*)|(bn4.*)|(res5.*)|(bn5.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)|(grid_reas\_.*)",
            "4+": r"(res4.*)|(bn4.*)|(res5.*)|(bn5.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)|(grid_reas\_.*)",
            "5+": r"(res5.*)|(bn5.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)|(grid_reas\_.*)",
            # All layers
            "all": ".*",
        }
        if layers in layer_regex.keys():
            layers = layer_regex[layers]

        # Data generators
        train_generator = data_generator(train_dataset, self.config, shuffle=True,
                                         augmentation=augmentation,
                                         batch_size=self.config.BATCH_SIZE,
                                         no_augmentation_sources=no_augmentation_sources, rnd_state=0)
        val_generator = data_generator(val_dataset, self.config, shuffle=True,
                                       batch_size=self.config.BATCH_SIZE, rnd_state=1)

        # Create log_dir if it does not exist
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
            
            
        class Save_BB_Callback(keras.callbacks.Callback):
            def __init__(self, model, log_dir):
                self.model = model
                self.log_dir = log_dir
            def on_epoch_end(self, epoch, logs=None):
                if epoch % 50 == 0:
                    for layer in self.model.layers:
                        if layer.name == 'backbone':
                            layer.save_weights(self.log_dir + '/backbone_callb_epoch_{:04d}.h5'.format(epoch+1))
                            break
                        
        
        class Save_all_Callback(keras.callbacks.Callback):
            def __init__(self, model, checkpoint_path):
                self.checkpoint_path = checkpoint_path
                self.model = model
            def on_epoch_end(self, epoch, logs=None):
                if epoch % 50 == 0:
                    filepath = self.checkpoint_path.format(epoch=epoch + 1, **logs)
                    self.model.save_weights(filepath, overwrite=True)
                    
                    
        save_callback = Save_all_Callback(self.keras_model, self.checkpoint_path)
        save_bb_callback = Save_BB_Callback(self.keras_model, self.log_dir)
        # Callbacks
        callbacks = [
            keras.callbacks.TensorBoard(log_dir=self.log_dir,
                                        histogram_freq=0, write_graph=True, write_images=False),
            save_callback,
            save_bb_callback
        ]

        # Add custom callbacks to the list
        if custom_callbacks:
            callbacks += custom_callbacks

        # Train
        log("\nStarting at epoch {}. LR={}\n".format(self.epoch, learning_rate))
        log("Checkpoint Path: {}".format(self.checkpoint_path))
        self.set_trainable(layers)
        self.compile(learning_rate, self.config.LEARNING_MOMENTUM)

        # Work-around for Windows: Keras fails on Windows when using
        # multiprocessing workers. See discussion here:
        # https://github.com/matterport/Mask_RCNN/issues/13#issuecomment-353124009
        if os.name is 'nt':
            workers = 0
        else:
            workers = multiprocessing.cpu_count()

        self.keras_model.fit_generator(
            train_generator,
            initial_epoch=self.epoch,
            epochs=epochs,
            steps_per_epoch=self.config.STEPS_PER_EPOCH,
            callbacks=callbacks,
            validation_data=val_generator,
            validation_steps=self.config.VALIDATION_STEPS,
            max_queue_size=10,
            workers=1,
            use_multiprocessing=False,
        )
        self.epoch = max(self.epoch, epochs)

    def mold_inputs(self, images):
        """Takes a list of images and modifies them to the format expected
        as an input to the neural network.
        images: List of image matrices [height,width,depth]. Images can have
            different sizes.

        Returns 3 Numpy matrices:
        molded_images: [N, h, w, 3]. Images resized and normalized.
        image_metas: [N, length of meta data]. Details about each image.
        windows: [N, (y1, x1, y2, x2)]. The portion of the image that has the
            original image (padding excluded).
        """
        molded_images = []
        image_metas = []
        windows = []
        for image in images:
            # Resize image
            # TODO: move resizing to mold_image()
            molded_image, window, scale, padding, crop = utils.resize_image(
                image,
                min_dim=self.config.IMAGE_MIN_DIM,
                min_scale=self.config.IMAGE_MIN_SCALE,
                max_dim=self.config.IMAGE_MAX_DIM,
                mode=self.config.IMAGE_RESIZE_MODE)
            molded_image = mold_image(molded_image, self.config)
            # Build image_meta
            image_meta = compose_image_meta(
                0, image.shape, molded_image.shape, window, scale,
                np.zeros([self.config.NUM_CLASSES], dtype=np.int32))
            # Append
            molded_images.append(molded_image)
            windows.append(window)
            image_metas.append(image_meta)
        # Pack into arrays
        molded_images = np.stack(molded_images)
        image_metas = np.stack(image_metas)
        windows = np.stack(windows)
        return molded_images, image_metas, windows

    def unmold_detections(self, detections, mrcnn_mask, original_image_shape,
                          image_shape, window):
        """Reformats the detections of one image from the format of the neural
        network output to a format suitable for use in the rest of the
        application.

        detections: [N, (y1, x1, y2, x2, class_id, score)] in normalized coordinates
        mrcnn_mask: [N, height, width, num_classes]
        original_image_shape: [H, W, C] Original image shape before resizing
        image_shape: [H, W, C] Shape of the image after resizing and padding
        window: [y1, x1, y2, x2] Pixel coordinates of box in the image where the real
                image is excluding the padding.

        Returns:
        boxes: [N, (y1, x1, y2, x2)] Bounding boxes in pixels
        class_ids: [N] Integer class IDs for each bounding box
        scores: [N] Float probability scores of the class_id
        masks: [height, width, num_instances] Instance masks
        """
        # How many detections do we have?
        # Detections array is padded with zeros. Find the first class_id == 0.
        zero_ix = np.where(detections[:, 4] == 0)[0]
        N = zero_ix[0] if zero_ix.shape[0] > 0 else detections.shape[0]

        # Extract boxes, class_ids, scores, and class-specific masks
        boxes = detections[:N, :4]
        class_ids = detections[:N, 4].astype(np.int32)
        scores = detections[:N, 5]
        masks = mrcnn_mask[np.arange(N), :, :, class_ids]

        # Translate normalized coordinates in the resized image to pixel
        # coordinates in the original image before resizing
        window = utils.norm_boxes(window, image_shape[:2])
        wy1, wx1, wy2, wx2 = window
        shift = np.array([wy1, wx1, wy1, wx1])
        wh = wy2 - wy1  # window height
        ww = wx2 - wx1  # window width
        scale = np.array([wh, ww, wh, ww])
        # Convert boxes to normalized coordinates on the window
        boxes = np.divide(boxes - shift, scale)
        # Convert boxes to pixel coordinates on the original image
        boxes = utils.denorm_boxes(boxes, original_image_shape[:2])

        # Filter out detections with zero area. Happens in early training when
        # network weights are still random
        exclude_ix = np.where(
            (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) <= 0)[0]
        if exclude_ix.shape[0] > 0:
            boxes = np.delete(boxes, exclude_ix, axis=0)
            class_ids = np.delete(class_ids, exclude_ix, axis=0)
            scores = np.delete(scores, exclude_ix, axis=0)
            masks = np.delete(masks, exclude_ix, axis=0)
            N = class_ids.shape[0]

        # Resize masks to original image size and set boundary threshold.
        full_masks = []
        for i in range(N):
            # Convert neural network mask to full size mask
            full_mask = utils.unmold_mask(masks[i], boxes[i], original_image_shape)
            full_masks.append(full_mask)
        full_masks = np.stack(full_masks, axis=-1)\
            if full_masks else np.empty(original_image_shape[:2] + (0,))

        return boxes, class_ids, scores, full_masks

    def detect(self, images, Rcam, Kmat, verbose=0):
        """Runs the detection pipeline.

        images: List of images, potentially of different sizes.

        Returns a list of dicts, one dict per image. The dict contains:
        rois: [N, (y1, x1, y2, x2)] detection bounding boxes
        class_ids: [N] int class IDs
        scores: [N] float probability scores for the class IDs
        masks: [H, W, N] instance binary masks
        """
        assert self.mode == "inference", "Create model in inference mode."
        assert len(
            images) == self.config.BATCH_SIZE, "len(images) must be equal to BATCH_SIZE"

        if verbose:
            log("Processing {} images".format(len(images)))
            for image in images:
                log("image", image)

        # Mold inputs to format expected by the neural network
        molded_images = []
        
        for i in range(self.config.BATCH_SIZE):
            molded_images_t, image_metas, windows = self.mold_inputs(images[i])
            molded_images.append(molded_images_t)
        molded_images = np.stack(molded_images)
        # Validate image sizes
        # All images in a batch MUST be of the same size
        image_shape = molded_images_t[0].shape
        for g in molded_images:
            for g_view in g:
                assert g_view.shape == image_shape,\
                    "After resizing, all images must have the same size. Check IMAGE_RESIZE_MODE and image sizes."

        # Anchors
        anchors = self.get_anchors(image_shape)
        # Duplicate across the batch dimension because Keras requires it
        # TODO: can this be optimized to avoid duplicating the anchors?
        anchors = np.broadcast_to(anchors, (self.config.BATCH_SIZE,) + anchors.shape)

        if verbose:
            log("molded_images", molded_images)
            log("image_metas", image_metas)
            log("anchors", anchors)
            log("Rcam", Rcam)
            log("Kmat", Kmat)
        # Run object detection
        detections, _, _, mrcnn_mask, _, _, _, mrcnn_features, mrcnn_intermediate_features =\
            self.keras_model.predict([molded_images, image_metas, anchors, Rcam, Kmat], verbose=0)
        # Process detections
        results = []
        for i, image in enumerate(images):
            final_rois, final_class_ids, final_scores, final_masks =\
                self.unmold_detections(detections[i], mrcnn_mask[i],
                                       image[0].shape, molded_images[0][i].shape,
                                       windows[i])
            results.append({
                "rois": final_rois,
                "class_ids": final_class_ids,
                "scores": final_scores,
                "masks": final_masks,
                "mrcnn_features": mrcnn_features[i],
                "mrcnn_intermediate_features": mrcnn_intermediate_features[i]
            })
        return results

    def detect_molded(self, molded_images, image_metas, verbose=0):
        """Runs the detection pipeline, but expect inputs that are
        molded already. Used mostly for debugging and inspecting
        the model.

        molded_images: List of images loaded using load_image_gt()
        image_metas: image meta data, also returned by load_image_gt()

        Returns a list of dicts, one dict per image. The dict contains:
        rois: [N, (y1, x1, y2, x2)] detection bounding boxes
        class_ids: [N] int class IDs
        scores: [N] float probability scores for the class IDs
        masks: [H, W, N] instance binary masks
        """
        assert self.mode == "inference", "Create model in inference mode."
        assert len(molded_images) == self.config.BATCH_SIZE,\
            "Number of images must be equal to BATCH_SIZE"

        if verbose:
            log("Processing {} images".format(len(molded_images)))
            for image in molded_images:
                log("image", image)

        # Validate image sizes
        # All images in a batch MUST be of the same size
        image_shape = molded_images[0].shape
        for g in molded_images[1:]:
            assert g.shape == image_shape, "Images must have the same size"

        # Anchors
        anchors = self.get_anchors(image_shape)
        # Duplicate across the batch dimension because Keras requires it
        # TODO: can this be optimized to avoid duplicating the anchors?
        anchors = np.broadcast_to(anchors, (self.config.BATCH_SIZE,) + anchors.shape)

        if verbose:
            log("molded_images", molded_images)
            log("image_metas", image_metas)
            log("anchors", anchors)
        # Run object detection
        detections, _, _, mrcnn_mask, _, _, _ =\
            self.keras_model.predict([molded_images, image_metas, anchors], verbose=0)
        # Process detections
        results = []
        for i, image in enumerate(molded_images):
            window = [0, 0, image.shape[0], image.shape[1]]
            final_rois, final_class_ids, final_scores, final_masks =\
                self.unmold_detections(detections[i], mrcnn_mask[i],
                                       image.shape, molded_images[i].shape,
                                       window)
            results.append({
                "rois": final_rois,
                "class_ids": final_class_ids,
                "scores": final_scores,
                "masks": final_masks,
            })
        return results

    def get_anchors(self, image_shape):
        """Returns anchor pyramid for the given image size."""
        backbone_shapes = compute_backbone_shapes(self.config, image_shape)
        # Cache anchors and reuse if image shape is the same
        if not hasattr(self, "_anchor_cache"):
            self._anchor_cache = {}
        if not tuple(image_shape) in self._anchor_cache:
            # Generate Anchors
            a = utils.generate_pyramid_anchors(
                self.config.RPN_ANCHOR_SCALES,
                self.config.RPN_ANCHOR_RATIOS,
                backbone_shapes,
                self.config.BACKBONE_STRIDES,
                self.config.RPN_ANCHOR_STRIDE)
            # Keep a copy of the latest anchors in pixel coordinates because
            # it's used in inspect_model notebooks.
            # TODO: Remove this after the notebook are refactored to not use it
            self.anchors = a
            # Normalize coordinates
            self._anchor_cache[tuple(image_shape)] = utils.norm_boxes(a, image_shape[:2])
        return self._anchor_cache[tuple(image_shape)]

    def ancestor(self, tensor, name, checked=None):
        """Finds the ancestor of a TF tensor in the computation graph.
        tensor: TensorFlow symbolic tensor.
        name: Name of ancestor tensor to find
        checked: For internal use. A list of tensors that were already
                 searched to avoid loops in traversing the graph.
        """
        checked = checked if checked is not None else []
        # Put a limit on how deep we go to avoid very long loops
        if len(checked) > 500:
            return None
        # Convert name to a regex and allow matching a number prefix
        # because Keras adds them automatically
        if isinstance(name, str):
            name = re.compile(name.replace("/", r"(\_\d+)*/"))

        parents = tensor.op.inputs
        for p in parents:
            if p in checked:
                continue
            if bool(re.fullmatch(name, p.name)):
                return p
            checked.append(p)
            a = self.ancestor(p, name, checked)
            if a is not None:
                return a
        return None

    def find_trainable_layer(self, layer):
        """If a layer is encapsulated by another layer, this function
        digs through the encapsulation and returns the layer that holds
        the weights.
        """
        if layer.__class__.__name__ == 'TimeDistributed':
            return self.find_trainable_layer(layer.layer)
        return layer

    def get_trainable_layers(self):
        """Returns a list of layers that have weights."""
        layers = []
        # Loop through all layers
        for l in self.keras_model.layers:
            # If layer is a wrapper, find inner trainable layer
            l = self.find_trainable_layer(l)
            # Include layer if it has weights
            if l.get_weights():
                layers.append(l)
        return layers

    def run_graph(self, images, Rcam, Kmat, outputs, image_metas=None):
        """Runs a sub-set of the computation graph that computes the given
        outputs.

        image_metas: If provided, the images are assumed to be already
            molded (i.e. resized, padded, and normalized)

        outputs: List of tuples (name, tensor) to compute. The tensors are
            symbolic TensorFlow tensors and the names are for easy tracking.

        Returns an ordered dict of results. Keys are the names received in the
        input and values are Numpy arrays.
        """
        model = self.keras_model

        # Organize desired outputs into an ordered dict
        outputs = OrderedDict(outputs)
        for o in outputs.values():
            print(o)
            assert o is not None

        # Build a Keras function to run parts of the computation graph
        inputs = model.inputs
        if model.uses_learning_phase and not isinstance(K.learning_phase(), int):
            inputs += [K.learning_phase()]
        kf = K.function(model.inputs, list(outputs.values()))

        # Prepare inputs
        if image_metas is None:
            molded_images = []
            for i in range(self.config.BATCH_SIZE):
                molded_images_t, image_metas, windows = self.mold_inputs(images[i])
                molded_images.append(molded_images_t)
            molded_images = np.stack(molded_images)
        else:
            molded_images = images
            
        
        
        image_shape = molded_images[0].shape
        # Anchors
        anchors = self.get_anchors(image_shape)
        # Duplicate across the batch dimension because Keras requires it
        # TODO: can this be optimized to avoid duplicating the anchors?
        anchors = np.broadcast_to(anchors, (self.config.BATCH_SIZE,) + anchors.shape)
        model_in = [molded_images, Rcam, Kmat, image_metas, anchors]
        for model_input in model_in:
            print(model_input.shape)
        # Run inference
        if model.uses_learning_phase and not isinstance(K.learning_phase(), int):
            model_in.append(0.)
        outputs_np = kf(model_in)

        # Pack the generated Numpy arrays into a a dict and log the results.
        outputs_np = OrderedDict([(k, v)
                                  for k, v in zip(outputs.keys(), outputs_np)])
        for k, v in outputs_np.items():
            log(k, v)
        return outputs_np


############################################################
#  Data Formatting
############################################################

def compose_image_meta(image_id, original_image_shape, image_shape,
                       window, scale, active_class_ids):
    """Takes attributes of an image and puts them in one 1D array.

    image_id: An int ID of the image. Useful for debugging.
    original_image_shape: [H, W, C] before resizing or padding.
    image_shape: [H, W, C] after resizing and padding
    window: (y1, x1, y2, x2) in pixels. The area of the image where the real
            image is (excluding the padding)
    scale: The scaling factor applied to the original image (float32)
    active_class_ids: List of class_ids available in the dataset from which
        the image came. Useful if training on images from multiple datasets
        where not all classes are present in all datasets.
    """
    meta = np.array(
        [image_id] +                  # size=1
        list(original_image_shape) +  # size=3
        list(image_shape) +           # size=3
        list(window) +                # size=4 (y1, x1, y2, x2) in image cooredinates
        [scale] +                     # size=1
        list(active_class_ids)        # size=num_classes
    )
    return meta


def parse_image_meta(meta):
    """Parses an array that contains image attributes to its components.
    See compose_image_meta() for more details.

    meta: [batch, meta length] where meta length depends on NUM_CLASSES

    Returns a dict of the parsed values.
    """
    image_id = meta[:, 0]
    original_image_shape = meta[:, 1:4]
    image_shape = meta[:, 4:7]
    window = meta[:, 7:11]  # (y1, x1, y2, x2) window of image in in pixels
    scale = meta[:, 11]
    active_class_ids = meta[:, 12:]
    return {
        "image_id": image_id.astype(np.int32),
        "original_image_shape": original_image_shape.astype(np.int32),
        "image_shape": image_shape.astype(np.int32),
        "window": window.astype(np.int32),
        "scale": scale.astype(np.float32),
        "active_class_ids": active_class_ids.astype(np.int32),
    }


def parse_image_meta_graph(meta):
    """Parses a tensor that contains image attributes to its components.
    See compose_image_meta() for more details.

    meta: [batch, meta length] where meta length depends on NUM_CLASSES

    Returns a dict of the parsed tensors.
    """
    image_id = meta[:, 0]
    original_image_shape = meta[:, 1:4]
    image_shape = meta[:, 4:7]
    window = meta[:, 7:11]  # (y1, x1, y2, x2) window of image in in pixels
    scale = meta[:, 11]
    active_class_ids = meta[:, 12:]
    return {
        "image_id": image_id,
        "original_image_shape": original_image_shape,
        "image_shape": image_shape,
        "window": window,
        "scale": scale,
        "active_class_ids": active_class_ids,
    }


def mold_image(images, config):
    """Expects an RGB image (or array of images) and subtracts
    the mean pixel and converts it to float. Expects image
    colors in RGB order.
    """
    return images.astype(np.float32) - config.MEAN_PIXEL


def unmold_image(normalized_images, config):
    """Takes a image normalized with mold() and returns the original."""
    return (normalized_images + config.MEAN_PIXEL).astype(np.uint8)


############################################################
#  Miscellenous Graph Functions
############################################################

def trim_zeros_graph(boxes, name='trim_zeros'):
    """Often boxes are represented with matrices of shape [N, 4] and
    are padded with zeros. This removes zero boxes.

    boxes: [N, 4] matrix of boxes.
    non_zeros: [N] a 1D boolean mask identifying the rows to keep
    """
    non_zeros = tf.cast(tf.reduce_sum(tf.abs(boxes), axis=1), tf.bool)
    boxes = tf.boolean_mask(boxes, non_zeros, name=name)
    return boxes, non_zeros


def batch_pack_graph(x, counts, num_rows):
    """Picks different number of values from each row
    in x depending on the values in counts.
    """
    outputs = []
    for i in range(num_rows):
        outputs.append(x[i, :counts[i]])
    return tf.concat(outputs, axis=0)


def norm_boxes_graph(boxes, shape):
    """Converts boxes from pixel coordinates to normalized coordinates.
    boxes: [..., (y1, x1, y2, x2)] in pixel coordinates
    shape: [..., (height, width)] in pixels

    Note: In pixel coordinates (y2, x2) is outside the box. But in normalized
    coordinates it's inside the box.

    Returns:
        [..., (y1, x1, y2, x2)] in normalized coordinates
    """
    print("shape_norm_boxes_graph: {}".format(shape))
    h, w = tf.split(tf.cast(shape, tf.float32), 2)
    scale = tf.concat([h, w, h, w], axis=-1) - tf.constant(1.0)
    shift = tf.constant([0., 0., 1., 1.])
    return tf.divide(boxes - shift, scale)


def denorm_boxes_graph(boxes, shape):
    """Converts boxes from normalized coordinates to pixel coordinates.
    boxes: [..., (y1, x1, y2, x2)] in normalized coordinates
    shape: [..., (height, width)] in pixels

    Note: In pixel coordinates (y2, x2) is outside the box. But in normalized
    coordinates it's inside the box.

    Returns:
        [..., (y1, x1, y2, x2)] in pixel coordinates
    """
    h, w = tf.split(tf.cast(shape, tf.float32), 2)
    scale = tf.concat([h, w, h, w], axis=-1) - tf.constant(1.0)
    shift = tf.constant([0., 0., 1., 1.])
    return tf.cast(tf.round(tf.multiply(boxes, scale) + shift), tf.int32)
