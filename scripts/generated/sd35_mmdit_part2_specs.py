input_names = ["add_144", "add_148", "pooled_projections", "timestep"]
input_specs = dict(add_144=((1, 4096, 1536), "float16"), add_148=((1, 154, 1536), "float16"), pooled_projections=((1, 2048), "float16"), timestep=((1, ), "float16"))
output_names = "add_207,add_211"
compile_options = "--truncate_64bit_io --output_names add_207,add_211"