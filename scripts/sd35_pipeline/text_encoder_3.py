import gc
import os
import numpy as np
import onnxruntime as ort
import qnn_ep_helper as qnn
from transformers import AutoTokenizer

import time

class TextEncoder3:
    def __init__(self, text_encoder_3_dir, tokenizer_3_dir):
        self.tokenizer_3 = None
        # self.model_3 = None
        self.model_path = os.path.join(text_encoder_3_dir)
        self.tokenizer_3_path = tokenizer_3_dir

    def get_tokens(self, prompt):
        # 1. ローカルフォルダからトークナイザー3をロード
        if self.tokenizer_3 is None:
            self.tokenizer_3 = AutoTokenizer.from_pretrained(
                self.tokenizer_3_path)
        
        # 2. トークンIDに変換
        text_inputs_3 = self.tokenizer_3(
            prompt,
            padding="max_length",
            # max_length=self.tokenizer_3.model_max_length,
            max_length=77,
            truncation=True,
            return_tensors="np"
        )
        # print(text_inputs_3)
        return text_inputs_3.input_ids.astype(np.int32), text_inputs_3.attention_mask.astype(np.int32)

    def get_text_embeddings_3(self, prompt: str, uncond_prompt: str, auto_mem_free=True):
        print("--- Text Encoder 3 処理開始 ---")
        
        input_ids_3, attention_mask = self.get_tokens(prompt)
        uncond_input_ids_3, uncond_attention_mask = self.get_tokens(uncond_prompt)
        
        # 3. NPUにロードして推論
        # if self.model_3 is None:
        #     self.model_3 = ort.InferenceSession(self.model_path, qnn.session_options)
        # output_names = list(map(lambda x: x.name,  self.model_3.get_outputs()))
        # output_list = self.model_3.run(output_names, {"input_ids": input_ids_3})
        prompt_embeds_3, uncond_prompt_embeds_3 = self.t5_pipeline(input_ids_3, attention_mask, uncond_input_ids_3, uncond_attention_mask)

        if auto_mem_free:
            self.free_memory()
     
        del input_ids_3, uncond_input_ids_3
        
        return prompt_embeds_3, uncond_prompt_embeds_3

    def free_memory(self):
        # del self.tokenizer_3, self.model_3
        del self.tokenizer_3
        gc.collect()

    #def t5_pipeline(self, input_ids_3, attention_mask, uncond_input_ids_3, uncond_attention_mask):
    def t5_pipeline(self, input_ids, attention_mask, uncond_input_ids, uncond_attention_mask):
        # NPUが求める int32 形式に型を整えます
        # input_ids = input_ids_3.astype(np.int32)
        # uncond_input_ids = uncond_input_ids_3.astype(np.int32)
        t5_dir = self.model_path
         
        print("=== 6分割 T5-XXL によるテキスト処理を開始します ===")
         
        # print("T5 処理時間計測開始")
        # start_time = time.perf_counter()
        
        # --- [Part 1] ロード ➔ 実行 ➔ 即解放 ---
        print("-> Part 1 を処理中...")
        session_t5 = ort.InferenceSession(f"{t5_dir}/Part1/model.onnx", sess_options=qnn.session_options)

        # elapsed_time = time.perf_counter()
        # print(f"モデルのロードにかかった時間: {elapsed_time - start_time:.3f} 秒")
        # start_time = elapsed_time
        
        hidden_states = session_t5.run(["add_32"], {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        })[0]
        uncond_hidden_states = session_t5.run(["add_32"], {
            "input_ids": uncond_input_ids,
            "attention_mask": uncond_attention_mask,
        })[0]
        
        # elapsed_time = time.perf_counter()
        # print(f"NPUでの処理にかかった時間: {elapsed_time - start_time:.3f} 秒")
        # start_time = elapsed_time
        
        del session_t5
        gc.collect()

        # --- [Part 2] ---
        print("-> Part 2 を処理中...")
        session_t5 = ort.InferenceSession(f"{t5_dir}/Part2/model.onnx", sess_options=qnn.session_options)
        
        # elapsed_time = time.perf_counter()
        # print(f"モデルのロードにかかった時間: {elapsed_time - start_time:.3f} 秒")
        # start_time = elapsed_time

        hidden_states = session_t5.run(["add_60"], {
            "add_32": hidden_states,
            "attention_mask": attention_mask,
        })[0]
        uncond_hidden_states = session_t5.run(["add_60"], {
            "add_32": uncond_hidden_states,
            "attention_mask": uncond_attention_mask,
        })[0]
        
        # elapsed_time = time.perf_counter()
        # print(f"NPUでの処理にかかった時間: {elapsed_time - start_time:.3f} 秒")
        # start_time = elapsed_time
        
        del session_t5
        gc.collect()
         
        # --- [Part 3] ---
        print("-> Part 3 を処理中...")
        session_t5 = ort.InferenceSession(f"{t5_dir}/Part3/model.onnx", sess_options=qnn.session_options)
        hidden_states = session_t5.run(["add_88"], {
            "add_60": hidden_states,
            "attention_mask": attention_mask,
        })[0]
        uncond_hidden_states = session_t5.run(["add_88"], {
            "add_60": uncond_hidden_states,
            "attention_mask": uncond_attention_mask,
        })[0]
        del session_t5
        gc.collect()
         
        # --- [Part 4] ---
        print("-> Part 4 を処理中...")
        session_t5 = ort.InferenceSession(f"{t5_dir}/Part4/model.onnx", sess_options=qnn.session_options)
        hidden_states = session_t5.run(["add_116"], {
            "add_88": hidden_states,
            "attention_mask": attention_mask,
        })[0]
        uncond_hidden_states = session_t5.run(["add_116"], {
            "add_88": uncond_hidden_states,
            "attention_mask": uncond_attention_mask,
        })[0]
        del session_t5
        gc.collect()
         
        # --- [Part 5] ---
        print("-> Part 5 を処理中...")
        session_t5 = ort.InferenceSession(f"{t5_dir}/Part5/model.onnx", sess_options=qnn.session_options)
        hidden_states = session_t5.run(["add_144"], {
            "add_116": hidden_states,
            "attention_mask": attention_mask,
        })[0]
        uncond_hidden_states = session_t5.run(["add_144"], {
            "add_116": uncond_hidden_states,
            "attention_mask": uncond_attention_mask,
        })[0]
        del session_t5
        gc.collect()

        # --- [Part 6] ---
        print("-> Part 6 を処理中...")
        session_t5 = ort.InferenceSession(f"{t5_dir}/Part6/model.onnx", sess_options=qnn.session_options)
        t5_outputs = session_t5.run(["hidden_states"], {
            "add_144": hidden_states,
            "attention_mask": attention_mask,
        })[0]
        uncond_t5_outputs = session_t5.run(["hidden_states"], {
            "add_144": uncond_hidden_states,
            "attention_mask": uncond_attention_mask,
        })[0]
        del session_t5
        gc.collect()

        return t5_outputs.astype(np.float16), uncond_t5_outputs.astype(np.float16)


if __name__ == "__main__":
    # 単体テスト用
    te = TextEncoder3()
    res_dict = te.get_text_embeddings_3("A beautiful cyberpunk city, 8k resolution")
    for k, v in res_dict.items():
        print(f"入力名: {k}, 出力形状: {v.shape}")
        # 想定される出力:
        # 1. prompt_embeds_3 相当 -> (1, 77, 4096)
