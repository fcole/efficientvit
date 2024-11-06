import jax
import jax.numpy as jnp
from jaxonnxruntime import backend as jax_backend
from jaxonnxruntime import config as jaxonnxruntime_config
import jaxonnxruntime.call_onnx as call_onnx
import jax_tile_op
import jax_depthtospace_op

from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from efficientvit.apps.utils.image import DMCrop
import onnx

jax.config.update("jax_enable_x64", True)
jaxonnxruntime_config.update("jaxort_only_allow_initializers_as_static_args", False)

# Set up transform (same as in dcae_test.py)
transform = transforms.Compose([
    DMCrop(512),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# Load and prepare image
image = Image.open("assets/fig/girl.png")
x = transform(image).numpy()  # Convert to numpy array
x = jnp.array(x)[None]  # Add batch dimension and convert to JAX array

# Load and JIT compile ONNX models
encoder_onnx = onnx.load("onnx/encoder.onnx")
decoder_onnx = onnx.load("onnx/decoder.onnx")

encoder_model_func, encoder_model_params = call_onnx.call_onnx_model(
    encoder_onnx, {'x': jnp.zeros((1, 3, 512, 512), dtype=jnp.float32)}
)

decoder_model_func, decoder_model_params = call_onnx.call_onnx_model(
    decoder_onnx, {'z': jnp.zeros((1, 128, 8, 8), dtype=jnp.float32)}
)

# Perform inference
@jax.jit
def encode(x, params):
    latent = encoder_model_func(params, {'x': x})
    return latent[0]

@jax.jit
def decode(z, params):
    output = decoder_model_func(params, {'z': z})
    return output[0]

# Convert output back to image
latent = encode(x, encoder_model_params)
output = decode(latent, decoder_model_params)

output = output.squeeze(0)  # Remove batch dimension
output = (output * 0.5 + 0.5)  # Denormalize
output = np.clip(output, 0, 1)  # Clip values to valid range
output = (output * 255).astype(np.uint8)  # Convert to 8-bit format

# Convert from CHW to HWC format for PIL
output = output.transpose(1, 2, 0)

# Save result
Image.fromarray(output).save("demo_dc_ae_jax.png")

print("Reconstruction complete! Check demo_dc_ae_jax.png") 