import gc
import os
import numpy as np
import onnxruntime as ort
import qnn_ep_helper as qnn
import qnn_ep_helper_use_bfloat as qnn_bfloat
from transformers import AutoTokenizer

import time

from utils import print_stats

class TextEncoder3:
    def __init__(self, text_encoder_3_dir, tokenizer_3_dir, use_cpu=False, use_bfloat=False):
        self.tokenizer_3 = None
        # self.model_3 = None
        self.model_path = os.path.join(text_encoder_3_dir)
        self.tokenizer_3_path = tokenizer_3_dir
        self.use_cpu = use_cpu
        self.use_bfloat = use_bfloat

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

    def get_t5_session(self, onnx_path, is_cpu_session=False, use_bfloat=False):
        if is_cpu_session:
            session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
            return session
        options = qnn_bfloat.session_options if use_bfloat else qnn.session_options
        session = ort.InferenceSession(onnx_path, sess_options=options)
        return session
        
    #def t5_pipeline(self, input_ids_3, attention_mask, uncond_input_ids_3, uncond_attention_mask):
    def t5_pipeline(self, input_ids, attention_mask, uncond_input_ids, uncond_attention_mask):
        print("=== 6分割 T5-XXL によるテキスト処理を開始します ===")
         
        hidden_states, uncond_hidden_states, output_name = self.run_part(1, "input_ids", input_ids, attention_mask,
                                                                         uncond_input_ids, uncond_attention_mask)
        hidden_states, uncond_hidden_states, output_name = self.run_part(2, output_name, hidden_states, attention_mask,
                                                                         uncond_hidden_states, uncond_attention_mask)
        hidden_states, uncond_hidden_states, output_name = self.run_part(3, output_name, hidden_states, attention_mask,
                                                                         uncond_hidden_states, uncond_attention_mask)
        hidden_states, uncond_hidden_states, output_name = self.run_part(4, output_name, hidden_states, attention_mask,
                                                                         uncond_hidden_states, uncond_attention_mask)
        hidden_states, uncond_hidden_states, output_name = self.run_part(5, output_name, hidden_states, attention_mask,
                                                                         uncond_hidden_states, uncond_attention_mask)
        t5_outputs, uncond_t5_outputs, _ = self.run_part(6, output_name, hidden_states, attention_mask,
                                                         uncond_hidden_states, uncond_attention_mask)

        return t5_outputs.astype(np.float16), uncond_t5_outputs.astype(np.float16)

    def run_part(self, part_num, input_name, hidden_states, attention_mask, uncond_hidden_states, uncond_attention_mask):
        t5_dir = self.model_path
        print(f"-> Part {part_num} を処理中...")
        if self.use_cpu:
            model_path = rf"{t5_dir}/part{part_num}.onnx"
        else:
            model_path = rf"{t5_dir}/Part{part_num}/model.onnx"
        session_t5 = self.get_t5_session(model_path, self.use_cpu, self.use_bfloat)
        output_name = list(map(lambda x: x.name, session_t5.get_outputs()))[0]

        hidden_states = session_t5.run([output_name], {
            input_name: hidden_states,
            "attention_mask": attention_mask,
        })[0]
        uncond_hidden_states = session_t5.run([output_name], {
            input_name: uncond_hidden_states,
            "attention_mask": uncond_attention_mask,
        })[0]
        del session_t5
        gc.collect()
        
        print_stats(f"Part{part_num} Pos Out", hidden_states)
        print_stats(f"Part{part_num} Neg Out", uncond_hidden_states)

        return hidden_states, uncond_hidden_states, output_name


if __name__ == "__main__":
    # 単体テスト用
    te = TextEncoder3()
    res_dict = te.get_text_embeddings_3("A beautiful cyberpunk city, 8k resolution")
    for k, v in res_dict.items():
        print(f"入力名: {k}, 出力形状: {v.shape}")
        # 想定される出力:
        # 1. prompt_embeds_3 相当 -> (1, 77, 4096)
