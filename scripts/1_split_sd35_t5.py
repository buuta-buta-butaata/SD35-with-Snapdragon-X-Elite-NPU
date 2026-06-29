import onnx
import onnx.utils
import os

onnx_path = "./onnx_models/t5xxl.onnx"
output_dir = "./onnx_models/t5_split_fixed"
os.makedirs(output_dir, exist_ok=True)

model = onnx.load(onnx_path, load_external_data=False)
graph = model.graph

print("=== T5の【最初の】LayerNormから境界テンソルを自動検出中... ===")

boundary_b3 = None
boundary_b7 = None
boundary_b11 = None
boundary_b15 = None
boundary_b19 = None

for node in graph.node:
    scopes = ""
    for prop in node.metadata_props:
        if prop.key == "pkg.torch.onnx.name_scopes":
            scopes = prop.value
            break
    if not scopes:
        continue

    # --- block 4 の入り口（Block 3 の出口） ---
    if "encoder.block.3" in scopes:
        boundary_b3 = node.output[0]
            
    if "encoder.block.7" in scopes:
        boundary_b7 = node.output[0]
        
    if "encoder.block.11" in scopes:
        boundary_b11 = node.output[0]
        
    if "encoder.block.15" in scopes:
        boundary_b15 = node.output[0]
        
    if "encoder.block.19" in scopes:
        boundary_b19 = node.output[0]
        
        
print("\n【検出結果】")
print(f"■ 境界:")
print(f"  -1: {boundary_b3}")
print(f"  -2: {boundary_b7}")
print(f"  -3: {boundary_b11}")
print(f"  -4: {boundary_b15}")
print(f"  -5: {boundary_b19}")

if not boundary_b3 or not boundary_b7:
    print("\n[エラー] 境界テンソルをロックできませんでした。処理を中断します。")
    exit()

# ----------------------------------------------------
# ステップ 2: extract_model で切り出し
# ----------------------------------------------------
orig_inputs = ["input_ids", "attention_mask"]
orig_outputs = ["hidden_states"]


print("\n--- Part1 を切り出し中... ---")
part1_path = os.path.join(output_dir, "sd35_t5_part1.onnx")
onnx.utils.extract_model(
    input_path=onnx_path,
    output_path=part1_path,
    input_names=orig_inputs,
    output_names=[boundary_b3]
)
print("Part1 成功!")

print("\n--- Part2 を切り出し中... ---")
part2_path = os.path.join(output_dir, "sd35_t5_part2.onnx")
part2_inputs = [boundary_b3, "attention_mask"]
onnx.utils.extract_model(
    input_path=onnx_path,
    output_path=part2_path,
    input_names=part2_inputs,
    output_names=[boundary_b7]
)
print("Part2 成功!")

print("\n--- Part3 を切り出し中... ---")
part3_path = os.path.join(output_dir, "sd35_t5_part3.onnx")
part3_inputs = [boundary_b7, "attention_mask"]
onnx.utils.extract_model(
    input_path=onnx_path,
    output_path=part3_path,
    input_names=part3_inputs,
    output_names=[boundary_b11]
)
print("Part3 成功!")

print("\n--- Part4 を切り出し中... ---")
part4_path = os.path.join(output_dir, "sd35_t5_part4.onnx")
part4_inputs = [boundary_b11, "attention_mask"]
onnx.utils.extract_model(
    input_path=onnx_path,
    output_path=part4_path,
    input_names=part4_inputs,
    output_names=[boundary_b15]
)
print("Part4 成功!")

print("\n--- Part5 を切り出し中... ---")
part5_path = os.path.join(output_dir, "sd35_t5_part5.onnx")
part5_inputs = [boundary_b15, "attention_mask"]
onnx.utils.extract_model(
    input_path=onnx_path,
    output_path=part5_path,
    input_names=part5_inputs,
    output_names=[boundary_b19]
)
print("Part5 成功!")

print("\n--- Part6 を切り出し中... ---")
part6_path = os.path.join(output_dir, "sd35_t5_part6.onnx")
part6_inputs = [boundary_b19, "attention_mask"]
onnx.utils.extract_model(
    input_path=onnx_path,
    output_path=part6_path,
    input_names=part6_inputs,
    output_names=orig_outputs
)
print("Part6 成功!")

print("\n=== 6分割がすべて成功しました！サイズを確認してください ===")
