import onnx
import onnx.utils
import os

onnx_path = "./onnx_models/sd35_mmdit_full.onnx"
output_dir = "./onnx_models/split"
os.makedirs(output_dir, exist_ok=True)

model = onnx.load(onnx_path, load_external_data=False)
graph = model.graph

print("=== MMDiTの【最初の】LayerNormから境界テンソルを自動検出中... ===")

boundary_b7 = {"img_tensor": None, "txt_tensor": None}
boundary_b10 = {"img_tensor": None, "txt_tensor": None}
boundary_b17 = {"img_tensor": None, "txt_tensor": None}

for node in graph.node:
    scopes = ""
    for prop in node.metadata_props:
        if prop.key == "pkg.torch.onnx.name_scopes":
            scopes = prop.value
            break
    if not scopes:
        continue

    # --- Block 7 の入り口（Block 6 の出口） ---
    if "transformer_blocks.7.norm1" in scopes and "norm1_context" not in scopes:
        # すでに記録済みならスキップすることで「最初に出現したノード」をロックします
        if boundary_b7["img_tensor"] is None:
            if node.op_type in ["LayerNormalization", "Cast", "Add"] and len(node.input) > 0:
                boundary_b7["img_tensor"] = node.input[0] # 最初の入力テンソル名を取得
            
    if "transformer_blocks.7.norm1_context" in scopes:
        if boundary_b7["txt_tensor"] is None:
            if node.op_type in ["LayerNormalization", "Cast", "Add"] and len(node.input) > 0:
                boundary_b7["txt_tensor"] = node.input[0]

    # --- Block 10 の入り口（Block 9 の出口） ---
    if "transformer_blocks.10.norm1" in scopes and "norm1_context" not in scopes:
        if boundary_b10["img_tensor"] is None:
            if node.op_type in ["LayerNormalization", "Cast", "Add"] and len(node.input) > 0:
                boundary_b10["img_tensor"] = node.input[0]
            
    if "transformer_blocks.10.norm1_context" in scopes:
        if boundary_b10["txt_tensor"] is None:
            if node.op_type in ["LayerNormalization", "Cast", "Add"] and len(node.input) > 0:
                boundary_b10["txt_tensor"] = node.input[0]

    # --- Block 17 の入り口（Block 16 の出口） ---
    if "transformer_blocks.17.norm1" in scopes and "norm1_context" not in scopes:
        if boundary_b17["img_tensor"] is None:
            if node.op_type in ["LayerNormalization", "Cast", "Add"] and len(node.input) > 0:
                boundary_b17["img_tensor"] = node.input[0]
            
    if "transformer_blocks.17.norm1_context" in scopes:
        if boundary_b17["txt_tensor"] is None:
            if node.op_type in ["LayerNormalization", "Cast", "Add"] and len(node.input) > 0:
                boundary_b17["txt_tensor"] = node.input[0]

print("\n【検出結果（修正版）】")
print(f"■ Part1 ➔ Part2 の境界 (Block 7 の本当の入り口):")
print(f"  - 画像用 (img_tensor): {boundary_b7['img_tensor']}")
print(f"  - 文字用 (txt_tensor): {boundary_b7['txt_tensor']}")

print(f"\n■ Part2 ➔ Part3 の境界 (Block 10 の本当の入り口):")
print(f"  - 画像用 (img_tensor): {boundary_b10['img_tensor']}")
print(f"  - 文字用 (txt_tensor): {boundary_b10['txt_tensor']}")

print(f"\n■ Part3 ➔ Part4 の境界 (Block 17 の本当の入り口):")
print(f"  - 画像用 (img_tensor): {boundary_b17['img_tensor']}")
print(f"  - 文字用 (txt_tensor): {boundary_b17['txt_tensor']}")

if not all(boundary_b7.values()) or not all(boundary_b10.values()):
    print("\n[エラー] 境界テンソルをロックできませんでした。処理を中断します。")
    exit()

# ----------------------------------------------------
# ステップ 2: extract_model で切り出し
# ----------------------------------------------------
orig_inputs = ["hidden_states", "encoder_hidden_states", "pooled_projections", "timestep"]
orig_outputs = ["sample"]

print("\n--- Part1 を切り出し中... ---")
part1_path = os.path.join(output_dir, "sd35_mmdit_part1.onnx")
onnx.utils.extract_model(
    input_path=onnx_path,
    output_path=part1_path,
    input_names=orig_inputs,
    output_names=[boundary_b7["img_tensor"], boundary_b7["txt_tensor"]]
)
print("Part1 成功!")

print("\n--- Part2 を切り出し中... ---")
part2_path = os.path.join(output_dir, "sd35_mmdit_part2.onnx")
part2_inputs = [boundary_b7["img_tensor"], boundary_b7["txt_tensor"], "pooled_projections", "timestep"]
onnx.utils.extract_model(
    input_path=onnx_path,
    output_path=part2_path,
    input_names=part2_inputs,
    output_names=[boundary_b10["img_tensor"], boundary_b10["txt_tensor"]]
)
print("Part2 成功!")

print("\n--- Part3 を切り出し中... ---")
part3_path = os.path.join(output_dir, "sd35_mmdit_part3.onnx")
part3_inputs = [boundary_b10["img_tensor"], boundary_b10["txt_tensor"], "pooled_projections", "timestep"]
onnx.utils.extract_model(
    input_path=onnx_path,
    output_path=part3_path,
    input_names=part3_inputs,
    output_names=[boundary_b17["img_tensor"], boundary_b17["txt_tensor"]]
)
print("Part3 成功!")

print("\n--- Part4 を切り出し中... ---")
part4_path = os.path.join(output_dir, "sd35_mmdit_part4.onnx")
part4_inputs = [boundary_b17["img_tensor"], boundary_b17["txt_tensor"], "pooled_projections", "timestep"]
onnx.utils.extract_model(
    input_path=onnx_path,
    output_path=part4_path,
    input_names=part4_inputs,
    output_names=orig_outputs
)
print("Part4 成功!")

print("\n=== 4分割がすべて成功しました！サイズを確認してください ===")
