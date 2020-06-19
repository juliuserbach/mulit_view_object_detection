from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import keras.backend as K
import keras.layers as KL
import keras.activations
import keras.initializers
import keras.regularizers
import keras.constraints
from keras.layers.recurrent import _generate_dropout_mask
from keras.layers.recurrent import _standardize_args

import numpy as np
import warnings
from keras.engine.base_layer import InputSpec, Layer
from keras.utils import conv_utils
from keras.legacy import interfaces
from keras.legacy.layers import Recurrent, ConvRecurrent2D
from keras.layers.recurrent import RNN
from keras.utils.generic_utils import has_arg
from keras.utils.generic_utils import to_list
from keras.utils.generic_utils import transpose_shape

class ConvRNN3D(KL.ConvRNN2D):
    def __init__(self, cell,
                 return_sequences=False,
                 return_state=False,
                 go_backwards=False,
                 stateful=False,
                 unroll=False,
                 **kwargs):
        super(ConvRNN3D, self).__init__(cell,
                                        return_sequences,
                                        return_state,
                                        go_backwards,
                                        stateful,
                                        unroll,
                                        **kwargs)
        self.input_spec = [InputSpec(ndim=6)]
        
        
    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, list):
            input_shape = input_shape[0]

        cell = self.cell
        if cell.data_format == 'channels_first':
            rows = input_shape[3]
            cols = input_shape[4]
            depth = input_shape[5]
        elif cell.data_format == 'channels_last':
            rows = input_shape[2]
            cols = input_shape[3]
            depth = input_shape[4]
#         rows = conv_utils.conv_output_length(rows,
#                                              cell.kernel_size[0],
#                                              padding=cell.padding,
#                                              stride=cell.strides[0],
#                                              dilation=cell.dilation_rate[0])
#         cols = conv_utils.conv_output_length(cols,
#                                              cell.kernel_size[1],
#                                              padding=cell.padding,
#                                              stride=cell.strides[1],
#                                              dilation=cell.dilation_rate[1])
#         depth = conv_utils.conv_output_length(depth,
#                                              cell.kernel_size[2],
#                                              padding=cell.padding,
#                                              stride=cell.strides[2],
#                                              dilation=cell.dilation_rate[2])

        output_shape = input_shape[:2] + (rows, cols, depth, cell.filters)
        output_shape = transpose_shape(output_shape, cell.data_format,
                                       spatial_axes=(2, 3, 4))

        if not self.return_sequences:
            output_shape = output_shape[:1] + output_shape[2:]

        if self.return_state:
            output_shape = [output_shape]
            base = (input_shape[0], rows, cols, depth, cell.filters)
            base = transpose_shape(base, cell.data_format, spatial_axes=(1, 2, 3))
            output_shape += [base[:] for _ in range(2)]
        return output_shape
    
    
    def build(self, input_shape):
        # Note input_shape will be list of shapes of initial states and
        # constants if these are passed in __call__.
        if self._num_constants is not None:
            constants_shape = input_shape[-self._num_constants:]
        else:
            constants_shape = None

        if isinstance(input_shape, list):
            input_shape = input_shape[0]

        batch_size = input_shape[0] if self.stateful else None
        self.input_spec[0] = InputSpec(shape=(batch_size, None) + input_shape[2:6])

        # allow cell (if layer) to build before we set or validate state_spec   ##### maybe has to be built even if not layer??
        if isinstance(self.cell, Layer):
            step_input_shape = (input_shape[0],) + input_shape[2:]
            if constants_shape is not None:
                self.cell.build([step_input_shape] + constants_shape)
            else:
                self.cell.build(step_input_shape)

        # set or validate state_spec
        if hasattr(self.cell.state_size, '__len__'):
            state_size = list(self.cell.state_size)
        else:
            state_size = [self.cell.state_size]

        if self.state_spec is not None:
            # initial_state was passed in call, check compatibility
            if self.cell.data_format == 'channels_first':
                ch_dim = 1
            elif self.cell.data_format == 'channels_last':
                ch_dim = 4
            if not [spec.shape[ch_dim] for spec in self.state_spec] == state_size:
                raise ValueError(
                    'An initial_state was passed that is not compatible with '
                    '`cell.state_size`. Received `state_spec`={}; '
                    'However `cell.state_size` is '
                    '{}'.format([spec.shape for spec in self.state_spec],
                                self.cell.state_size))
        else:
            if self.cell.data_format == 'channels_first':
                self.state_spec = [InputSpec(shape=(None, dim, None, None, None))
                                   for dim in state_size]
            elif self.cell.data_format == 'channels_last':
                self.state_spec = [InputSpec(shape=(None, None, None, None, dim))
                                   for dim in state_size]
        if self.stateful:
            self.reset_states()
        self.built = True
        
    
    def get_initial_state(self, inputs):
        # (samples, timesteps, rows, cols, depth, filters)
        initial_state = K.zeros_like(inputs)
        # (samples, rows, cols, depth, filters)
        initial_state = K.sum(initial_state, axis=1)
        shape = list(self.cell.kernel_shape)
        shape[-1] = self.cell.filters
#         initial_state = self.cell.input_conv(initial_state,
#                                              K.zeros(tuple(shape)),
#                                              padding=self.cell.padding)
        # Fix for Theano because it needs
        # K.int_shape to work in call() with initial_state.
        keras_shape = list(K.int_shape(inputs))
        keras_shape.pop(1)
        if K.image_data_format() == 'channels_first':
            indices = 2, 3, 4
        else:
            indices = 1, 2, 3
#         for i, j in enumerate(indices):
#             keras_shape[j] = conv_utils.conv_output_length(
#                 keras_shape[j],
#                 shape[i],
#                 padding=self.cell.padding,
#                 stride=self.cell.strides[i],
#                 dilation=self.cell.dilation_rate[i])
        initial_state._keras_shape = keras_shape

        if hasattr(self.cell.state_size, '__len__'):
            return [initial_state for _ in self.cell.state_size]
        else:
            return [initial_state]
        
        
    def __call__(self, inputs, initial_state=None, constants=None, **kwargs):
        inputs, initial_state, constants = _standardize_args(
            inputs, initial_state, constants, self._num_constants)

        if initial_state is None and constants is None:
            return super(ConvRNN3D, self).__call__(inputs, **kwargs)

        # If any of `initial_state` or `constants` are specified and are Keras
        # tensors, then add them to the inputs and temporarily modify the
        # input_spec to include them.

        additional_inputs = []
        additional_specs = []
        if initial_state is not None:
            kwargs['initial_state'] = initial_state
            additional_inputs += initial_state
            self.state_spec = []
            for state in initial_state:
                try:
                    shape = K.int_shape(state)
                # Fix for Theano
                except TypeError:
                    shape = tuple(None for _ in range(K.ndim(state)))
                self.state_spec.append(InputSpec(shape=shape))

            additional_specs += self.state_spec
        if constants is not None:
            kwargs['constants'] = constants
            additional_inputs += constants
            self.constants_spec = [InputSpec(shape=K.int_shape(constant))
                                   for constant in constants]
            self._num_constants = len(constants)
            additional_specs += self.constants_spec
        # at this point additional_inputs cannot be empty
        for tensor in additional_inputs:
            if K.is_keras_tensor(tensor) != K.is_keras_tensor(additional_inputs[0]):
                raise ValueError('The initial state or constants of an RNN'
                                 ' layer cannot be specified with a mix of'
                                 ' Keras tensors and non-Keras tensors')

        if K.is_keras_tensor(additional_inputs[0]):
            # Compute the full input spec, including state and constants
            full_input = [inputs] + additional_inputs
            full_input_spec = self.input_spec + additional_specs
            # Perform the call with temporarily replaced input_spec
            original_input_spec = self.input_spec
            self.input_spec = full_input_spec
            output = super(ConvRNN3D, self).__call__(full_input, **kwargs)
            self.input_spec = original_input_spec
            return output
        else:
            return super(ConvRNN3D, self).__call__(inputs, **kwargs)
        
    
    def call(self,
             inputs,
             mask=None,
             training=None,
             initial_state=None,
             constants=None):
        # note that the .build() method of subclasses MUST define
        # self.input_spec and self.state_spec with complete input shapes.
        if isinstance(inputs, list):
            inputs = inputs[0]
        if initial_state is not None:
            pass
        elif self.stateful:
            initial_state = self.states
        else:
            initial_state = self.get_initial_state(inputs)

        if isinstance(mask, list):
            mask = mask[0]

        if len(initial_state) != len(self.states):
            raise ValueError('Layer has ' + str(len(self.states)) +
                             ' states but was passed ' +
                             str(len(initial_state)) +
                             ' initial states.')
        timesteps = K.int_shape(inputs)[1]

        kwargs = {}
        if has_arg(self.cell.call, 'training'):
            kwargs['training'] = training

        if constants:
            if not has_arg(self.cell.call, 'constants'):
                raise ValueError('RNN cell does not support constants')

            def step(inputs, states):
                constants = states[-self._num_constants:]
                states = states[:-self._num_constants]
                return self.cell.call(inputs, states, constants=constants,
                                      **kwargs)
        else:
            def step(inputs, states):
                return self.cell.call(inputs, states, **kwargs)

        last_output, outputs, states = K.rnn(step,
                                             inputs,
                                             initial_state,
                                             constants=constants,
                                             go_backwards=self.go_backwards,
                                             mask=mask,
                                             input_length=timesteps)
        if self.stateful:
            updates = []
            for i in range(len(states)):
                updates.append((self.states[i], states[i]))
            self.add_update(updates, inputs)

        if self.return_sequences:
            output = outputs
        else:
            output = last_output

        # Properly set learning phase
        if getattr(last_output, '_uses_learning_phase', False):
            output._uses_learning_phase = True

        if self.return_state:
            states = to_list(states, allow_tuple=True)
            return [output] + states
        else:
            return output
        
        
    def reset_states(self, states=None):
        if not self.stateful:
            raise AttributeError('Layer must be stateful.')
        input_shape = self.input_spec[0].shape
        state_shape = self.compute_output_shape(input_shape)
        if self.return_state:
            state_shape = state_shape[0]
        if self.return_sequences:
            state_shape = state_shape[:1] + state_shape[2:]
        if None in state_shape:
            raise ValueError('If a RNN is stateful, it needs to know '
                             'its batch size. Specify the batch size '
                             'of your input tensors: \n'
                             '- If using a Sequential model, '
                             'specify the batch size by passing '
                             'a `batch_input_shape` '
                             'argument to your first layer.\n'
                             '- If using the functional API, specify '
                             'the time dimension by passing a '
                             '`batch_shape` argument to your Input layer.\n'
                             'The same thing goes for the number of rows '
                             'and columns.')

        # helper function
        def get_tuple_shape(nb_channels):
            result = list(state_shape)
            if self.cell.data_format == 'channels_first':
                result[1] = nb_channels
            elif self.cell.data_format == 'channels_last':
                result[4] = nb_channels
            else:
                raise KeyError
            return tuple(result)

        # initialize state if None
        if self.states[0] is None:
            if hasattr(self.cell.state_size, '__len__'):
                self.states = [K.zeros(get_tuple_shape(dim))
                               for dim in self.cell.state_size]
            else:
                self.states = [K.zeros(get_tuple_shape(self.cell.state_size))]
        elif states is None:
            if hasattr(self.cell.state_size, '__len__'):
                for state, dim in zip(self.states, self.cell.state_size):
                    K.set_value(state, np.zeros(get_tuple_shape(dim)))
            else:
                K.set_value(self.states[0],
                            np.zeros(get_tuple_shape(self.cell.state_size)))
        else:
            states = to_list(states, allow_tuple=True)
            if len(states) != len(self.states):
                raise ValueError('Layer ' + self.name + ' expects ' +
                                 str(len(self.states)) + ' states, '
                                 'but it received ' + str(len(states)) +
                                 ' state values. Input received: ' +
                                 str(states))
            for index, (value, state) in enumerate(zip(states, self.states)):
                if hasattr(self.cell.state_size, '__len__'):
                    dim = self.cell.state_size[index]
                else:
                    dim = self.cell.state_size
                if value.shape != get_tuple_shape(dim):
                    raise ValueError('State ' + str(index) +
                                     ' is incompatible with layer ' +
                                     self.name + ': expected shape=' +
                                     str(get_tuple_shape(dim)) +
                                     ', found shape=' + str(value.shape))
                # TODO: consider batch calls to `set_value`.
                K.set_value(state, value)