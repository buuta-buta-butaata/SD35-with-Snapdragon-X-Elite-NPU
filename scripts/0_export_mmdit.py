import argparse
import torch
from diffusers import SD3Transformer2DModel
import os

def parse_args():
    parser = argparse.ArgumentParser(description="export SD3.5 MMDiT-X")
    parser.add_argument("--model_path", type=str, default="../safetensors/your_model.safetensors", help="元モデルのパス")
    return parser.parse_args()

args = parse_args()

# 1. 保存先とモデルの準備
output_dir = "./onnx_models"
os.makedirs(output_dir, exist_ok=True)
onnx_path = os.path.join(output_dir, "sd35_mmdit_full.onnx")

model_path = args.model_path
print("=== MMDiT モデルをロード中... ===")
# FP16、CPU（または可能ならCUDA）で読み込みます
# model = SD3Transformer2DModel.from_pretrained(
#    model_id, 
#    subfolder="transformer", 
#    torch_dtype=torch.float16
# )
model = SD3Transformer2DModel.from_single_file(
    model_path,
    torch_dtype=torch.float16,
)
model.eval()

# 2. ダミーデータの作成（SD3.5-Mediumの標準サイズ）
print("=== ダミーデータを生成中... ===")
batch_size = 1

# 画像の潜在変数 (Latent)
# SD3.5のチャンネル数は16、1024x1024生成時は128x128サイズになります
hidden_states = torch.randn(batch_size, 16, 128, 128, dtype=torch.float16)

# タイムステップ (Denoisingの進み具合)
timestep = torch.tensor([1.0], dtype=torch.float16)

# CLIP-Lからのテキスト入力
# SD3.5のシーケンス長は77、次元数は4096です
encoder_hidden_states = torch.randn(batch_size, 77+77, 4096, dtype=torch.float16)

# CLIPのプールされた出力 (ポインティングベクトル)
pooled_projections = torch.randn(batch_size, 2048, dtype=torch.float16)

# 引数を辞書にまとめます
inputs = {
    "hidden_states": hidden_states,
    "timestep": timestep,
    "encoder_hidden_states": encoder_hidden_states,
    "pooled_projections": pooled_projections
}

# 3. ONNXへエクスポート
print(f"=== ONNXへのエクスポートを開始します... ===")
print("※これには数分かかり、一時的にPCのメモリを多く消費します。")

# inputsの順番を正しく渡すためのタプル化
input_tuple = (
    inputs["hidden_states"],
    inputs["encoder_hidden_states"],
    inputs["pooled_projections"],
    inputs["timestep"]
)

with torch.no_grad():
    torch.onnx.export(
        model,
        input_tuple,
        onnx_path,
        export_params=True,
        opset_version=22,
        do_constant_folding=True,
        input_names=["hidden_states", "encoder_hidden_states", "pooled_projections", "timestep"],
        output_names=["sample"],  # MMDiTの最終出力名
        # スコープ（名前）をメタデータに残す設定（これが分割時に超重要になります！）
        keep_initializers_as_inputs=False
    )

print(f"=== エクスポート完了！ ===\n保存先: {onnx_path}")
