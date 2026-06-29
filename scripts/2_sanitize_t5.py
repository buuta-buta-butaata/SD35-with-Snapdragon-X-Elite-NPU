import onnx
import os
import gc
import our_onnx_utils as utils

for i in range(1, 7):
    model_path = f"onnx_models/t5_split_fixed/sd35_t5_part{i}.onnx"
    utils.sanitize_value_info(model_path)
    utils.generate_specs(i, model_path, "t5")
