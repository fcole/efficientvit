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

"""Define ONNX SpaceToDepth and DepthToSpace operators."""
from collections.abc import Callable, Sequence
import functools
from typing import Any
import jax
from jax import numpy as jnp
# from jaxonnxruntime.core import config_class

# config = config_class.config
from jaxonnxruntime import config
from jaxonnxruntime.core import handler
from jaxonnxruntime.core import onnx_node


@handler.register_op('SpaceToDepth')
class SpaceToDepth(handler.Handler):
  """Implementation of the ONNX SpaceToDepth operator."""

  @classmethod
  def _prepare(
      cls, node: onnx_node.OnnxNode, inputs: Sequence[Any], onnx_jax_impl: Any
  ):
    node.attrs_dict['blocksize'] = node.attrs.get('blocksize')
    if not node.attrs_dict['blocksize']:
      raise ValueError('Attribute blocksize is not set.')

  @classmethod
  def version_1(
      cls, node: onnx_node.OnnxNode, inputs: Sequence[Any]
  ) -> Callable[..., Any]:
    """ONNX version_1 SpaceToDepth op."""
    cls._prepare(node, inputs, onnx_space_to_depth)
    return onnx_space_to_depth

  @classmethod
  def version_13(
      cls, node: onnx_node.OnnxNode, inputs: Sequence[Any]
  ) -> Callable[..., Any]:
    """ONNX version_13 SpaceToDepth op."""
    cls._prepare(node, inputs, onnx_space_to_depth)
    return onnx_space_to_depth


@handler.register_op('DepthToSpace')
class DepthToSpace(handler.Handler):
  """Implementation of the ONNX DepthToSpace operator."""

  @classmethod
  def _prepare(
      cls, node: onnx_node.OnnxNode, inputs: Sequence[Any], onnx_jax_impl: Any
  ):
    node.attrs_dict['blocksize'] = node.attrs.get('blocksize')
    if not node.attrs_dict['blocksize']:
      raise ValueError('Attribute blocksize is not set.')
    
    # Default mode is "DCR"
    mode = node.attrs.get('mode', 'DCR')
    if mode not in ['DCR', 'CRD']:
      raise ValueError("DepthToSpace op: only 'DCR' and 'CRD' modes are supported")
    node.attrs_dict['mode'] = mode

  @classmethod
  def version_1(
      cls, node: onnx_node.OnnxNode, inputs: Sequence[Any]
  ) -> Callable[..., Any]:
    """ONNX version_1 DepthToSpace op."""
    cls._prepare(node, inputs, onnx_depth_to_space)
    return onnx_depth_to_space

  @classmethod
  def version_11(
      cls, node: onnx_node.OnnxNode, inputs: Sequence[Any]
  ) -> Callable[..., Any]:
    """ONNX version_11 DepthToSpace op."""
    cls._prepare(node, inputs, onnx_depth_to_space)
    return onnx_depth_to_space

  @classmethod
  def version_13(
      cls, node: onnx_node.OnnxNode, inputs: Sequence[Any]
  ) -> Callable[..., Any]:
    """ONNX version_13 DepthToSpace op."""
    cls._prepare(node, inputs, onnx_depth_to_space)
    return onnx_depth_to_space


@functools.partial(jax.jit, static_argnames=('blocksize',))
def onnx_space_to_depth(data, *, blocksize):
  """Implementation of ONNX SpaceToDepth operator.
  
  Args:
    data: Input tensor of shape [N,C,H,W]
    blocksize: Size of blocks to use for space to depth transformation
    
  Returns:
    Output tensor of shape [N,C*(blocksize*blocksize),H/blocksize,W/blocksize]
  """
  N, C, H, W = data.shape
  
  if H % blocksize != 0 or W % blocksize != 0:
    raise ValueError(
        f'Height ({H}) and width ({W}) must be multiples of blocksize ({blocksize})'
    )

  # Reshape to [N, C, H/blocksize, blocksize, W/blocksize, blocksize]
  x = jnp.reshape(data, (N, C, H // blocksize, blocksize, W // blocksize, blocksize))
  
  # Transpose to [N, C, blocksize, blocksize, H/blocksize, W/blocksize]
  x = jnp.transpose(x, (0, 1, 3, 5, 2, 4))
  
  # Reshape to [N, C*(blocksize*blocksize), H/blocksize, W/blocksize]
  return jnp.reshape(x, (N, C * blocksize * blocksize, H // blocksize, W // blocksize))


@functools.partial(jax.jit, static_argnames=('blocksize', 'mode'))
def onnx_depth_to_space(data, *, blocksize, mode='DCR'):
  """Implementation of ONNX DepthToSpace operator.
  
  Args:
    data: Input tensor of shape [N,C,H,W]
    blocksize: Size of blocks to use for depth to space transformation
    mode: Either 'DCR' or 'CRD' for different data format interpretations
    
  Returns:
    Output tensor of shape [N,C/(blocksize*blocksize),H*blocksize,W*blocksize]
  """
  N, C, H, W = data.shape
  
  if C % (blocksize * blocksize) != 0:
    raise ValueError(
        f'Channel size ({C}) must be divisible by blocksize*blocksize ({blocksize*blocksize})'
    )
  
  new_C = C // (blocksize * blocksize)
  
  if mode == 'DCR':
    # DCR mode: Depth, Column, Row
    # First reshape to [N, blocksize, blocksize, new_C, H, W]
    x = jnp.reshape(data, (N, blocksize, blocksize, new_C, H, W))
    # Transpose to [N, new_C, H, blocksize, W, blocksize]
    x = jnp.transpose(x, (0, 3, 4, 1, 5, 2))
  else:  # CRD mode
    # CRD mode: Column, Row, Depth
    # First reshape to [N, new_C, blocksize, blocksize, H, W]
    x = jnp.reshape(data, (N, new_C, blocksize, blocksize, H, W))
    # Transpose to [N, new_C, H, blocksize, W, blocksize]
    x = jnp.transpose(x, (0, 1, 4, 2, 5, 3))
  
  # Finally reshape to [N, new_C, H*blocksize, W*blocksize]
  return jnp.reshape(x, (N, new_C, H * blocksize, W * blocksize))