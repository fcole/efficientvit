import onnxruntime
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from efficientvit.apps.utils.image import DMCrop

# Initialize ONNX Runtime sessions
encoder_session = onnxruntime.InferenceSession(
    "onnx/encoder.onnx",
    providers=['CPUExecutionProvider']  # Use 'CUDAExecutionProvider' for GPU
)
decoder_session = onnxruntime.InferenceSession(
    "onnx/decoder.onnx",
    providers=['CPUExecutionProvider']  # Use 'CUDAExecutionProvider' for GPU
)

# Set up transform (same as in dcae_test.py)
transform = transforms.Compose([
    DMCrop(512),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# Load and prepare image
image = Image.open("assets/fig/girl.png")
x = transform(image).numpy()  # Convert to numpy array
x = x[None]  # Add batch dimension

# Perform inference
latent = encoder_session.run(
    ['z'], 
    {'x': x.astype(np.float32)}
)[0]

output = decoder_session.run(
    ['x'], 
    {'z': latent.astype(np.float32)}
)[0]

# Convert output back to image
output = output.squeeze(0)  # Remove batch dimension
output = (output * 0.5 + 0.5)  # Denormalize
output = np.clip(output, 0, 1)  # Clip values to valid range
output = (output * 255).astype(np.uint8)  # Convert to 8-bit format

# Convert from CHW to HWC format for PIL
output = output.transpose(1, 2, 0)

# Save result
Image.fromarray(output).save("demo_dc_ae_onnx.png")

print("Reconstruction complete! Check demo_dc_ae_onnx.png") 