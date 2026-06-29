input_names = ["add_144", "attention_mask"]
input_specs = dict(add_144=((1, 77, 4096), "float32"), attention_mask=((1, 77), "int32"))
output_names = "hidden_states"
compile_options = "--truncate_64bit_io --output_names hidden_states"