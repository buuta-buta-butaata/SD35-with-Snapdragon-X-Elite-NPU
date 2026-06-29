input_names = ["hidden_states", "encoder_hidden_states", "pooled_projections", "timestep"]
input_specs = dict(hidden_states=((1, 16, 128, 128), "float16"), encoder_hidden_states=((1, 154, 4096), "float16"), pooled_projections=((1, 2048), "float16"), timestep=((1, ), "float16"))
output_names = "add_144,add_148"
compile_options = "--truncate_64bit_io --output_names add_144,add_148"