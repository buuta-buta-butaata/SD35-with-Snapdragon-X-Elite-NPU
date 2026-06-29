input_names = ["add_60", "attention_mask"]
input_specs = dict(add_60=((1, 77, 4096), "float32"), attention_mask=((1, 77), "int32"))
output_names = "add_88"
compile_options = "--truncate_64bit_io --output_names add_88"