import copy
import gc
import os
from datetime import datetime

import numpy as np
import onnxruntime as ort
import torch

from tqdm.auto import tqdm

import qnn_ep_helper as qnn
from schedulers import Scheduler
#from calib_data_collector import CalibrationDataCollector

import time

import threading
import queue

class NpuMMDiTLoop:
    def __init__(self, config):
        self.model_dir = config.dirs.mmdit_dir
        self.session_p1 = ort.InferenceSession(f"{self.model_dir}/Part1/model.onnx", sess_options=qnn.session_options)
        self.session_p2 = None
        self.session_p3 = None
        self.session_p4 = None

    def find_model_path(self, mmdit_dir, part, width, height):
        model_dir = os.path.join(mmdit_dir, part)
        single_graph_model = os.path.join(model_dir, "model.onnx")
        # part0 はwidth、heightに依存しないので、model.onnx１つだけ
        if os.path.exists(single_graph_model):
            return single_graph_model
        return os.path.join(model_dir, f"mmdit_{part}_{width}x{height}.onnx")

    def inference(self, config, prompt_embeds, pooled_prompt_embeds,
                  uncond_embeds, uncond_pooled_embeds):
        # --------------------------------------------------
        # Step 1: スケジューラーの初期化
        # --------------------------------------------------
        # ローカルの設定ファイルを読み込んでスケジューラーを準備
        #scheduler = DPMSolverSinglestepScheduler.from_pretrained(
        #scheduler = DPMSolverMultistepScheduler.from_pretrained(
        #    config.dirs.scheduler_dir)
        #scheduler = EulerAncestralDiscreteScheduler.from_pretrained(
        #    config.dirs.scheduler_dir)
        # scheduler = Scheduler.get(config.scheduler_type)
        scheduler = Scheduler.get(0)
        scheduler.set_timesteps(config.steps)

        # --------------------------------------------------
        # Step 3: MMDiT デノイズループの準備
        # --------------------------------------------------
        # 初期ノイズ (Latents) の生成
        # ※ 再現性を持たせるため、必要に応じて torch.manual_seed を設定してください
        if config.seed == -1:
            config.seed = np.random.randint(np.iinfo(np.uint32).max, dtype=np.uint32)
        generator = torch.manual_seed(config.seed)
        print(f"seed: {config.seed}")
        
        latents = torch.randn(1, 16, config.height // 8, config.width // 8,
                              dtype=torch.float32, generator=generator)
        # latents = latents * scheduler.init_noise_sigma
    
        # --------------------------------------------------
        # Step 4: デノイズ ループ実行
        # --------------------------------------------------
        print("\n--- デノイズループ開始 ---")
        # NumPy配列に変換してループに投入
        latents_np = latents.numpy()

        encoder_hidden_states = prompt_embeds.astype(np.float16)
        pooled_projections = pooled_prompt_embeds.astype(np.float16)
        if config.cfg_scale != 1:
            uncond_hidden_states = uncond_embeds.astype(np.float16)
            uncond_pooled_projections = uncond_pooled_embeds.astype(np.float16)
        latents_torch = torch.from_numpy(latents_np)
    
        # import tensors_io as tio
        # for i, t in enumerate(scheduler.timesteps):
        # for t in tqdm(scheduler.timesteps):
        for i, t in enumerate(tqdm(scheduler.timesteps)):
            # 現在の進捗を表示
            # print(f"🔄 Step {i+1}/{steps} (Timestep: {t.item():.1f})")
        
            # scaled_latents_torch = scheduler.scale_model_input(latents_torch, t)
            scaled_latents_torch = latents_torch
            scaled_latents_np = scaled_latents_torch.numpy().astype(np.float16)

            # タイムステップをMMDiTが要求する形状 [1] のfloat16配列にする
            timestep_np = np.array([t.item()], dtype=np.float16)
        
            # tio.save("./", "latents.npy", scaled_latents_np)
            # tio.save("./", "timestep.npy", timestep_np)
            # tio.save("./", "encoder_hidden_states.npy", encoder_hidden_states)
            # tio.save("./", "pooled_projections.npy", pooled_projections)

            # 常駐しているMMDiTのforwardを実行
            # 前回の修正（内部でのNHWC変換、float32統一）が施されたMMDiTが動きます
            if config.cfg_scale != 1:
                layer_skipping = i > 0 and i / config.steps < 0.20 and config.slg_scale > 0

                noise_pred_text, noise_pred_skip = self.forward(
                    scaled_latents_np, timestep_np,
                    encoder_hidden_states, pooled_projections,
                    save_calibration_data=False,
                    step=i, layer_skipping=layer_skipping)
                
                noise_pred_uncond, _ = self.forward(
                    scaled_latents_np, timestep_np,
                    uncond_hidden_states,  uncond_pooled_projections,
                    save_calibration_data=False,
                    step=i)

                noise_pred = noise_pred_uncond + config.cfg_scale * (
                    noise_pred_text - noise_pred_uncond
                )

                if layer_skipping:
                    noise_pred = noise_pred + config.slg_scale * (
                        noise_pred_text - noise_pred_skip
                    )
            else:
                layer_skipping = i > 0 and i / config.steps < 0.20 and config.slg_scale > 0

                noise_pred_text, noise_pred_skip = self.forward(
                    scaled_latents_np, timestep_np,
                    encoder_hidden_states, pooled_projections,
                    save_calibration_data=False,
                    step=i, layer_skipping=layer_skipping)

                if layer_skipping:
                    noise_pred = noise_pred_text + config.slg_scale * (
                        noise_pred_text - noise_pred_skip
                    )
                else:
                    noise_pred = noise_pred_text
                    
            # スケジューラーを使ってノイズを除去し、次のlatentsを計算
            # ※ diffusersのschedulerはtorch.Tensorを期待するため、一時的に戻して処理
            latents_torch = scheduler.step(
                torch.from_numpy(noise_pred), 
                t, 
                torch.from_numpy(latents_np),
                return_dict=False,
            )[0]
        
            # 次のステップのためにNumPyに変換
            latents_np = latents_torch.numpy()
            """
            out_latents_np = latents_np / 0.13025
                
            out_image_tensor = vae_decoder.decode_latents(out_latents_np, auto_mem_free=False)
            output_image(out_image_tensor, f"{t.item():.1f}.png")
            """
        
        print("--- デノイズループ完了 ---")
        # del base_inputs
        if config.cfg_scale != 1:
            pass
            # del uncond_base_inputs

        return latents_np
    
    def forward(self, latents, timestep, hidden, pooled, save_calibration_data=False, step = 0, layer_skipping = False):
        p1_img_out = "add_144"
        p1_txt_out = "add_148"

        p2_img_out = "add_207"
        p2_txt_out = "add_211"

        p3_img_out = "add_334"
        p3_txt_out = "add_338"
        
        def print_stats(name, tensor):
            """テンソルの健全性をチェックする補助関数"""
            # ndarrayでない（リストやタプルの）場合は最初の要素を対象にする
            if isinstance(tensor, (list, tuple)):
                tensor = tensor[0]
            arr = np.array(tensor, dtype=np.float32)
            has_nan = np.isnan(arr).any()
            has_inf = np.isinf(arr).any()
            print(f"  [{name}] Shape: {arr.shape} | Mean: {arr.mean():.4f} | Max: {arr.max():.4f} | Min: {arr.min():.4f} | NaN: {has_nan} | Inf: {has_inf}")

        # print(f"\n=== MMDiT 内部統計チェック (Timestep: {timestep}) ===")
        # print_stats("Input: latents", latents)

        # ------------------------------------------------
        # 【Part 1】の処理：ロード ➔ 実行 ➔ 即解放
        # ------------------------------------------------
        # print("MMDiT 処理時間計測開始")
        # start_time = time.perf_counter()
        
        # elapsed_time = time.perf_counter()
        # print(f"Part1 モデルのロードにかかった時間: {elapsed_time - start_time:.3f} 秒")
        # start_time = elapsed_time

        out_p1_pos = self.session_p1.run([p1_img_out, p1_txt_out], {
            "hidden_states": latents, "encoder_hidden_states": hidden,
            "pooled_projections": pooled, "timestep": timestep
        })
        
        # ★ Part1の出力をチェック
        # print_stats("Part1 Out (Pos)[0] img", out_p1_pos[0])
        # print_stats("Part1 Out (Pos)[1] txt", out_p1_pos[1])

        # elapsed_time = time.perf_counter()
        # print(f"Part1 NPUでの処理にかかった時間: {elapsed_time - start_time:.3f} 秒")
        # start_time = elapsed_time

        # ------------------------------------------------
        # 【Part 2】の処理：ロード ➔ 実行 ➔ 即解放
        # ------------------------------------------------
        if self.session_p2 is None:
            self.session_p2 = ort.InferenceSession(f"{self.model_dir}/Part2/model.onnx", sess_options=qnn.session_options)
        
        # elapsed_time = time.perf_counter()
        # print(f"Part2 モデルのロードにかかった時間: {elapsed_time - start_time:.3f} 秒")
        # start_time = elapsed_time
        
        out_p2_pos = self.session_p2.run([p2_img_out, p2_txt_out], {
            p1_img_out: out_p1_pos[0], p1_txt_out: out_p1_pos[1],
            "pooled_projections": pooled, "timestep": timestep
        })
    
        # ★ Part2の出力をチェック
        # print_stats("Part2 Out (Pos)[0] img", out_p2_pos[0])
        # print_stats("Part2 Out (Pos)[1] txt", out_p2_pos[1])

        # out_p2_pos[1] = np.clip(out_p2_pos[1], -8000, 8000) # 値を強制的に安全圏に縛る
        # out_p2_neg[1] = np.clip(out_p2_neg[1], -8000, 8000) # 値を強制的に安全圏に縛る
        
        # elapsed_time = time.perf_counter()
        # print(f"Part2 NPUでの処理にかかった時間: {elapsed_time - start_time:.3f} 秒")
        # start_time = elapsed_time

        # ------------------------------------------------
        # 【Part 3】の処理：ロード ➔ 実行 ➔ 即解放
        # ------------------------------------------------
        if self.session_p3 is None:
            self.session_p3 = ort.InferenceSession(f"{self.model_dir}/Part3/model.onnx", sess_options=qnn.session_options)
        
        # elapsed_time = time.perf_counter()
        # print(f"Part3 モデルのロードにかかった時間: {elapsed_time - start_time:.3f} 秒")
        # start_time = elapsed_time

        out_p3_pos = self.session_p3.run([p3_img_out, p3_txt_out], {
            p2_img_out: out_p2_pos[0], p2_txt_out: out_p2_pos[1],
            "pooled_projections": pooled, "timestep": timestep
        })
        if layer_skipping:
            out_p3_pos_skip = self.session_p3.run([p3_img_out, p3_txt_out], {
                p2_img_out: out_p1_pos[0], p2_txt_out: out_p1_pos[1],
                "pooled_projections": pooled, "timestep": timestep
            })
        del out_p2_pos, out_p1_pos
            
        # ------------------------------------------------
        # 【Part 4】の処理：ロード ➔ 実行 ➔ 即解放
        # ------------------------------------------------
        if self.session_p4 is None:
            self.session_p4 = ort.InferenceSession(f"{self.model_dir}/Part4/model.onnx", sess_options=qnn.session_options)
        
        # elapsed_time = time.perf_counter()
        # print(f"Part4 モデルのロードにかかった時間: {elapsed_time - start_time:.3f} 秒")
        # start_time = elapsed_time
        noise_pred_pos = self.session_p4.run(["sample"], {
            p3_img_out: out_p3_pos[0], p3_txt_out: out_p3_pos[1],
            "pooled_projections": pooled, "timestep": timestep
        })
        if layer_skipping:
            noise_pred_pos_skip = self.session_p4.run(["sample"], {
                p3_img_out: out_p3_pos_skip[0], p3_txt_out: out_p3_pos_skip[1],
                "pooled_projections": pooled, "timestep": timestep
            })
        del out_p3_pos

        # ★ Part3（最終出力）のチェック
        # print_stats("Part3 Out (Pos) Final", noise_pred_pos[0])
        # print_stats("Part3 Out (Neg) Final", noise_pred_neg[0])

        # elapsed_time = time.perf_counter()
        # print(f"Part3 NPUでの処理にかかった時間: {elapsed_time - start_time:.3f} 秒")
        # start_time = elapsed_time

        if layer_skipping:
            return noise_pred_pos[0].astype(np.float32), noise_pred_pos_skip[0].astype(np.float32)
        else:
            return noise_pred_pos[0].astype(np.float32), None

if __name__ == "__main__":
    import torch
    # 単体テスト用
    mmdit = NpuMMDiTLoop()
    pos_hidden = np.random.randn(1, 77, 4096).astype(np.float16)
    pos_pooled = np.random.randn(1, 2048).astype(np.float16)

    neg_hidden = np.random.randn(1, 77, 4096).astype(np.float16)
    neg_pooled = np.random.randn(1, 2048).astype(np.float16)
                
    latents = torch.randn(1, 16, 128, 128, dtype=torch.float16).numpy()
    timestep = torch.randn(1, dtype=torch.float32).numpy()
    res1, res2 = mmdit.forward(latents, timestep, pos_hidden, pos_pooled,
                        neg_hidden, neg_pooled)
    print(f"MMDiT 出力形状: {res1.shape}") # 想定: (1, 16, 128, 128)
    
