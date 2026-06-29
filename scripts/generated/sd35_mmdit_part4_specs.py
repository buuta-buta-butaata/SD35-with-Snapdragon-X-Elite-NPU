input_names = ["add_334", "add_338", "pooled_projections", "timestep"]
input_specs = dict(add_334=((1, 4096, 1536), "float16"), add_338=((1, 154, 1536), "float16"), pooled_projections=((1, 2048), "float16"), timestep=((1, ), "float16"))
output_names = "sample"
compile_options = "--truncate_64bit_io --output_names sample"