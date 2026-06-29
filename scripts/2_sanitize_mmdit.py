import onnx
import os
import gc
import our_onnx_utils as utils

model_path = "onnx_models/split/sd35_mmdit_part1.onnx"
utils.sanitize_value_info(model_path)
utils.generate_specs(1, model_path, "mmdit")

model_path = "onnx_models/split/sd35_mmdit_part2.onnx"
utils.sanitize_value_info(model_path)
utils.generate_specs(2, model_path, "mmdit")

model_path = "onnx_models/split/sd35_mmdit_part3.onnx"
utils.sanitize_value_info(model_path)
utils.generate_specs(3, model_path, "mmdit")

model_path = "onnx_models/split/sd35_mmdit_part4.onnx"
utils.sanitize_value_info(model_path)
utils.generate_specs(4, model_path, "mmdit")
