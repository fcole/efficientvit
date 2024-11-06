# Copyright 2024 The Jaxonnxruntime Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Define ONNX Tile operator."""
from collections.abc import Callable, Sequence
import functools
from typing import Any
import jax
from jax import numpy as jnp
# from jaxonnxruntime.core import config_class
from jaxonnxruntime import config

#config = config_class.config
from jaxonnxruntime.core import handler
from jaxonnxruntime.core import onnx_node
import numpy as np


@handler.register_op('Tile')
class Tile(handler.Handler):
  """Implementation of the ONNX Tile operator."""

  @classmethod
  def _prepare(
      cls, node: onnx_node.OnnxNode, inputs: Sequence[Any], onnx_jax_impl: Any
  ):
    # For jit optimization, if repeats is a constant, we store it in attrs
    if config.jaxort_only_allow_initializers_as_static_args:
      if node.inputs[1] not in node.context_graph.get_constant_dict():
        raise ValueError(
            f'{node.inputs[1]} is not constant but used as `repeats` of Tile'
            ' static argument during `jax.jit`. The jitted function gives'
            ' wrong results if its value changes in another input. If you know'
            ' what you are doing, set'
            ' `config.update("jaxort_only_allow_initializers_as_static_args",'
            ' False)` to remove this constraint.'
        )
      node.attrs_dict['repeats'] = tuple(
          node.context_graph.get_constant_dict()[node.inputs[1]].tolist()
      )
    else:
      node.attrs_dict['repeats'] = tuple(inputs[1].tolist())

  @classmethod
  def _validate(cls, input_shape, repeats):
    """Validates inputs and repeats have compatible shapes."""
    if len(repeats) != len(input_shape):
      raise ValueError(
          f'Repeats length ({len(repeats)}) must match input rank ({len(input_shape)})'
      )
    if len(input_shape) < 1:
      raise ValueError('Input tensor must be at least 1-dimensional')

  @classmethod
  def version_6(
      cls, node: onnx_node.OnnxNode, inputs: Sequence[Any]
  ) -> Callable[..., Any]:
    """ONNX version_6 Tile op."""
    cls._prepare(node, inputs, onnx_tile)
    return onnx_tile

  @classmethod
  def version_13(
      cls, node: onnx_node.OnnxNode, inputs: Sequence[Any]
  ) -> Callable[..., Any]:
    """ONNX version_13 Tile op."""
    cls._prepare(node, inputs, onnx_tile)
    return onnx_tile


def is_tile_memcpy(input_shape, repeats):
  """Determines if tiling operation can be optimized as a memory copy.
  
  This is a JAX-friendly version of the C++ IsTileMemcpy optimization check.
  Returns True if the tiling operation is essentially copying the input buffer
  multiple times, which happens when:
  1) All dims to the left of the first non-1 repeat are 1's, or
  2) At most one non-1 dim value to the left (batch dimension case)
  
  Args:
    input_shape: Shape of input tensor
    repeats: Repeat values for each dimension
    
  Returns:
    is_optimizable: Whether the operation can be optimized
    is_batched: Whether it's a batched operation
    batch_size: Size of batch dimension if batched
    copies_per_batch: Number of copies per batch
    batch_copies: Number of times to copy batches
  """
  rank = len(input_shape)
  
  # Look for first non-1 repeat from the right
  for i in range(rank - 1, -1, -1):
    if repeats[i] != 1:
      # Check if all dims to the left are 1
      if np.prod(input_shape[:i]) == 1:
        # Simple copy case
        return (True, False, 1, int(np.prod(repeats[:i + 1])), 1)
      
      # Check batched case - only position 1 can have non-1 dim
      elif i == 1:
        batch_size = input_shape[0]
        elements_per_batch = int(np.prod(input_shape[1:]))
        return (True, True, elements_per_batch, repeats[i], repeats[0])
      
      # Any other case can't be optimized
      else:
        break
        
  return (False, False, 1, 1, 1)


@functools.partial(jax.jit, static_argnames=('repeats',))
def onnx_tile(data, repeats_tensor, *, repeats):
  """Implementation of ONNX Tile operator.
  
  Args:
    data: Input tensor to tile
    repeats_tensor: Tensor containing repeat counts for each dim (unused but required by ONNX)
    repeats: Tuple of repeat counts for each dimension (static argument)
    
  Returns:
    Tiled output tensor
  """
  # Validate inputs
  input_shape = data.shape
  if len(repeats) != len(input_shape):
    raise ValueError(
        f'Repeats length ({len(repeats)}) must match input rank ({len(input_shape)})'
    )

  # Early return for empty tensor
  if 0 in input_shape or 0 in repeats:
    return jnp.empty([d * r for d, r in zip(input_shape, repeats)])

  # No tiling case
  if all(r == 1 for r in repeats):
    return data

  # Check if we can optimize
  is_optimizable, is_batched, elems_per_batch, copies_per_batch, batch_copies = (
      is_tile_memcpy(input_shape, repeats)
  )

  if is_optimizable:
    if not is_batched:
      # Simple repeat case
      return jnp.tile(data, repeats)
    else:
      # Batched case
      # First tile each batch
      batched_shape = list(input_shape)
      batched_repeats = [1] + [copies_per_batch if i == 1 else 1 for i in range(1, len(input_shape))]
      result = jnp.tile(data, batched_repeats)
      
      # Then tile the batches if needed
      if batch_copies > 1:
        result = jnp.tile(result, [batch_copies] + [1] * (len(input_shape) - 1))
      
      return result

  # General case using reshape and tile
  # JAX's tile operation handles the generic case efficiently
  return jnp.tile(data, repeats)