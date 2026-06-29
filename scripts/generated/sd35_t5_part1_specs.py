input_names = ["input_ids", "attention_mask"]
input_specs = dict(input_ids=((1, 77), "int32"), attention_mask=((1, 77), "int32"))
output_names = "add_32"
compile_options = "--truncate_64bit_io --output_names add_32"