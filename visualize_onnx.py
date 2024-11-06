import netron
import os

# Get the absolute path to the ONNX file
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, 'onnx', 'encoder.onnx')

# Start Netron server and open in browser
print(f"Opening {model_path} in Netron...")
netron.start(model_path) 