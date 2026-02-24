from mapper import get_model
import torch
import qai_hub as hub
import numpy as np

model = get_model(
    encoder='vitl',
    dataset='hypersim',
    max_depth=20
)

model = model.to('cpu')
model.eval()

dummy_input = torch.randn(1, 3, 518, 518, device='cpu')  # H/W must be multiples of 14
traced_model = torch.jit.trace(model, dummy_input)
traced_model.save("depth_anything_v2_slam_rover.pt")
print("Model exported successfully as depth_anything_v2_slam_rover.pt")

# export onnx
# torch.onnx.export(
#     model,
#     dummy_input,
#     "depth_anything_v2_slam_rover.onnx",
#     opset_version=17,
#     input_names=["input"],
#     output_names=["depth"],
# )
# print("Model exported successfully as depth_anything_v2_slam_rover.onnx")

device = hub.Device("Dragonwing RB3 Gen 2 Vision Kit")

# submit to qai hub
calib_data = {
    "input": [np.random.rand(1, 3, 518, 518).astype(np.float32) for _ in range(5)]
}

# job = hub.submit_compile_and_profile_jobs(
#     model_path="depth_anything_v2_slam_rover.onnx",
#     device=device,
#     calibration_data=calib_data,
#     weights_dtype=hub.INT8,
#     activations_dtype=hub.INT8,
#     options="--target_runtime qnn_context_binary"
# )

# Compile your model to ONNX
compile_job = hub.submit_compile_job(
    		model=traced_model,
    		device=device,
    		input_specs=dict(image=dummy_input.shape),
    		options="--target_runtime onnx"
)
unquantized_onnx_model = compile_job.get_target_model()
print("Model compiled to ONNX successfully.")

import os
from PIL import Image

mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
std = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))
sample_inputs = []

images_dir = "images_surroundings"  # Directory containing calibration images

for image_path in os.listdir(images_dir):
    image = Image.open(os.path.join(images_dir, image_path))
    image = image.convert("RGB").resize(dummy_input.shape[2:])
    sample_input = np.array(image).astype(np.float32) / 255.0
    sample_input = np.expand_dims(np.transpose(sample_input, (2, 0, 1)), 0)
    sample_inputs.append(((sample_input - mean) / std).astype(np.float32))
calibration_data = dict(image=sample_inputs)

# Submit quantize job
quantize_job = hub.submit_quantize_job(
    model=unquantized_onnx_model,
    calibration_data=calibration_data,
    weights_dtype=hub.QuantizeDtype.INT8,
    activations_dtype=hub.QuantizeDtype.INT8,
)
quantized_onnx_model = quantize_job.get_target_model()
print("Model quantized successfully.")

# Optimize model for the chosen device
compile_job = hub.submit_compile_job(
        model=quantized_onnx_model,
        device=device,
        input_specs=dict(image=dummy_input.shape),
        options="--target_runtime tflite"
)
target_model = compile_job.get_target_model()
print("Model optimized successfully.")

target_model.download("depth_anything_v2_slam_rover.tflite")
quantized_onnx_model.download("depth_anything_v2_slam_rover_quantized.onnx")