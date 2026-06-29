input_names = ["add_116", "attention_mask"]
input_specs = dict(add_116=((1, 77, 4096), "float32"), attention_mask=((1, 77), "int32"))
output_names = "add_144"
compile_options = "--truncate_64bit_io --output_names add_144"