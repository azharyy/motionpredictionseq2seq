
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function



import torch
from torch import nn
from torch.nn import RNNCell


import collections
import math

class ResidualWrapper(RNNCell):
  """Operator adding residual connections to a given cell."""

  def __init__(self, cell):
    if not isinstance(cell, RNNCell):
      raise TypeError("The parameter cell is not a RNNCell.")

    self._cell = cell

  def forward(self, inputs, state, scope=None):
    from IPython import embed
    embed()

    # Run the rnn as usual
    output, new_state = self._cell(inputs, state)

    output = output + inputs    
    return output, new_state

class LinearSpaceDecoderWrapper(RNNCell):
  """Operator adding a linear encoder to an RNN cell"""

  def __init__(self, cell, output_size):
    if not isinstance(cell, RNNCell):
      raise TypeError("The parameter cell is not a RNNCell.")

    from IPython import embed
    embed()

    self._cell = cell

    print( 'output_size = {0}'.format(output_size) )
    print( ' state_size = {0}'.format(self._cell.state_size) )

    self.fc1 = nn.Linear(self._cell.state_size, output_size_o_output_size[cell -1])

    self.linear_output_size = output_size


  def forward(self, inputs, state, scope=None):
    from IPython import embed
    embed()

    output, new_state = self._cell(inputs, state)

#    output, new_state = self._cell(inputs, state, scope)

    # Apply the multiplication to everything
#    output = tf.matmul(output, self.w_out) + self.b_out
    output = self.fc1(output)

    return output, new_state
