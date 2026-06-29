import onnx
import os
import gc

def sanitize_value_info(model_path):
    # モデルの読み込み
    model = onnx.load(model_path, load_external_data=False)
    graph = model.graph

    # 1. 現在の「入力（inputs）」と「出力（outputs）」の名前をすべて収集
    io_names = set()
    for inp in graph.input:
        io_names.add(inp.name)
    for out in graph.output:
        io_names.add(out.name)

    # 2. value_info を走査し、入出力に含まれている名前のものを除外する
    new_value_info = []
    removed_count = 0
    
    for info in graph.value_info:
        if info.name in io_names:
            # 入出力と重複している場合はスキップ（削除）
            removed_count += 1
            continue
        new_value_info.append(info)

    # 3. グラフの value_info をクレンジングされた新しいリストに置き換える
    del graph.value_info[:]
    graph.value_info.extend(new_value_info)

    # 保存
    onnx.save_model(
        model,
        model_path,
    )
    print(f"\n[{model_path}] クレンジング完了: {removed_count} 個の重複した value_info を削除しました。")



def zatsu_type2str(inp):
    if inp == 1:
        return "float32"
    if inp == 6:
        return "int32"
    elif inp == 10:
        return "float16"
    return f"unknown {inp}"

def generate_specs(part_num, path, model_name):
    print(f"\nFor {path} " + "=" * (75 - len(f"For {path} ")))
    model = onnx.load(path, load_external_data=False)
    graph = model.graph

    inputs = graph.input
    outputs = graph.output

    input_specs = []
    input_names = []
    for inp in inputs:
        name = inp.name
        type = inp.type.tensor_type
        dim = type.shape.dim

        input_names.append(f"\"{name}\"")
        if len(dim) == 1:
            # print(f"{name}=(({dim[0].dim_value}, ), \"{zatsu_type2str(type.elem_type)}\")")
            dim_values = f"({dim[0].dim_value}, )"
            # input_specs.append(f"{name}=(({dim[0].dim_value}, ), \"{zatsu_type2str(type.elem_type)}\")")
        else:
            # print(f"{name}=(({', '.join(map(lambda x: str(x.dim_value), dim))}), \"{zatsu_type2str(type.elem_type)}\")")
            dim_values = f"({', '.join(map(lambda x: str(x.dim_value), dim))})"
        # print(f"{name}=({dim_values}, \"{zatsu_type2str(type.elem_type)}\")")
        input_specs.append(f"{name}=({dim_values}, \"{zatsu_type2str(type.elem_type)}\")")

    output_names = []
    for out in outputs:
        output_names.append(out.name)

    specs = f"input_names = [{', '.join(input_names)}]"
    specs += f"\ninput_specs = dict({', '.join(input_specs)})"
    specs += f"\noutput_names = \"{','.join(output_names)}\""
    specs += f"\ncompile_options = \"--truncate_64bit_io --output_names {','.join(output_names)}\""
    print(specs)

    temp_dir = "./generated"
    os.makedirs(temp_dir, exist_ok=True)
    output_IO_specs(f"{temp_dir}/sd35_{model_name}_part{part_num}_specs.py", specs)

def output_IO_specs(file_name, text):
    with open(file_name, "w") as f:
        f.write(text)
    

# print("\nプリコンパイル時に利用する入力情報、コンパイルオプションを出力")
# generate_specs(common_path)
# generate_specs(down_path)
# generate_specs(mid_path)
# generate_specs(up_1_path)
# generate_specs(up_2_path)

if __name__ == "__main__":
    exit()

