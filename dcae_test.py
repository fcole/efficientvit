# build DC-AE models
# full DC-AE model list: https://huggingface.co/collections/mit-han-lab/dc-ae-670085b9400ad7197bb1009b
from efficientvit.ae_model_zoo import DCAE_HF

dc_ae = DCAE_HF.from_pretrained(f"mit-han-lab/dc-ae-f64c128-in-1.0")

# encode
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image
from efficientvit.apps.utils.image import DMCrop
import torch.jit

EXPORT_ENCODER = True
EXPORT_DECODER = True

# build DC-AE models and move to device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
dc_ae = dc_ae.to(device).eval()

with torch.inference_mode():   
    # Create traced encoder and decoder
    dummy_input = torch.randn(1, 3, 512, 512).to(device)  # Match expected input dimensions
    dummy_latent = dc_ae.encode(dummy_input)  # Create appropriate latent shape for decoder
    traced_model = torch.jit.trace_module(
        dc_ae,
        inputs={
            'encode': dummy_input,
            'decode': dummy_latent
        }
    )

    # Set up transform and load image
    transform = transforms.Compose([
        DMCrop(512),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    image = Image.open("assets/fig/girl.png")
    x = transform(image)[None].to(device)

    # Use traced model instead of original
    latent = traced_model.encode(x)
    y = traced_model.decode(latent)

    save_image(y * 0.5 + 0.5, "demo_dc_ae.png")

    # Create ONNX export directories
    import os
    os.makedirs("onnx", exist_ok=True)

    # Create wrapper modules for ONNX export
    class EncoderWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.model.encode(x)

    class DecoderWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.model.decode(x)

    # Create wrapper instances with original model
    encoder_wrapper = EncoderWrapper(dc_ae).to(device).eval()
    decoder_wrapper = DecoderWrapper(dc_ae).to(device).eval()
    # Trace the wrapper modules
    encoder_input = torch.randn(1, 3, 512, 512).to(device)
    traced_encoder = torch.jit.trace(encoder_wrapper, encoder_input)

    decoder_input = dc_ae.encode(encoder_input)
    traced_decoder = torch.jit.trace(decoder_wrapper, decoder_input)

    if EXPORT_ENCODER:
        # Export traced encoder to ONNX
        dummy_input = torch.randn(1, 3, 512, 512).to(device)  # Create fresh input tensor
        dynamic_axes = {
            'x': {0: 'batch_size'},
            'z': {0: 'batch_size'}
        }
        torch.onnx.export(
            traced_encoder,
            dummy_input,  # Use fresh input tensor instead of reusing encoder_input
            "onnx/encoder.onnx",
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=['x'],
            output_names=['z'],
            dynamic_axes=dynamic_axes
        )

    # Export traced decoder to ONNX
    if EXPORT_DECODER:
        dummy_latent = torch.randn(1, 128, 8, 8).to(device)
        dynamic_axes = {
            'z': {0: 'batch_size'},
            'x': {0: 'batch_size'}
        }
        torch.onnx.export(
            traced_decoder,
            dummy_latent,  # Use fresh latent tensor instead of reusing decoder_input
            "onnx/decoder.onnx",
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=['z'],
            output_names=['x'],
            dynamic_axes=dynamic_axes
        )

