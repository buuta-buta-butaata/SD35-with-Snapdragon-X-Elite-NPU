import gc
import numpy as np
from text_encoder import TextEncoder
from text_encoder_2 import TextEncoder2
from text_encoder_3 import TextEncoder3

class TextProcessing:
    def __init__(self, text_encoder_dir, tokenizer_dir, text_encoder_2_dir, tokenizer_2_dir,
                 text_encoder_3_dir, tokenizer_3_dir):
        self.text_encoder = TextEncoder(text_encoder_dir, tokenizer_dir)
        self.text_encoder_2 = TextEncoder2(text_encoder_2_dir, tokenizer_2_dir)
        self.text_encoder_3 = TextEncoder3(text_encoder_3_dir, tokenizer_3_dir)
        return

    def _encode_text(self, prompt, prompt_2):
        # Text Encoder 1 の処理 & 即時解放
        pooled_prompt_embeds_1, prompt_embeds_1 = self.text_encoder.get_text_embeddings(prompt, auto_mem_free=False)
    
        # Text Encoder 2 の処理 & 即時解放
        pooled_prompt_embeds_2, prompt_embeds_2 = self.text_encoder_2.get_text_embeddings_2(
            prompt_2, auto_mem_free=False)

        # SD3.5仕様に合わせて2つのエンコーダーの出力をドッキング
        prompt_embeds = np.concatenate([prompt_embeds_1, prompt_embeds_2], axis=-1)
        pooled_prompt_embeds = np.concatenate([pooled_prompt_embeds_1, pooled_prompt_embeds_2], axis=-1)
    
        # 不要になった中間変数を徹底的に掃除
        #del prompt_embeds_1, prompt_embeds_2
        gc.collect()
        return prompt_embeds, pooled_prompt_embeds

    def pad_for_sd35(self, hidden, hidden_3):
        pad_hidden = np.zeros((1, 77, 4096), dtype=np.float16)
        pad_hidden[:, :, :2048] = hidden
        pad_hidden = np.concatenate([pad_hidden, hidden_3], axis = -2)
        return pad_hidden
    
    def encode_text(self, config):
        prompt_embeds, pooled_prompt_embeds = self._encode_text(
            config.prompt, config.prompt_2)
        # prompt_embeds, pooled_prompt_embeds = self._encode_text(
        #     "", "")
        
        if config.cfg_scale != 1:
            uncond_embeds, uncond_pooled_embeds = self._encode_text(
                config.negative_prompt, config.negative_prompt_2)
        else:
            uncond_embeds, uncond_pooled_embeds = None, None

        # prompt_embeds_3 = np.zeros((1, 77, 4096), dtype=np.float16)
        # uncond_embeds_3 = np.zeros((1, 77, 4096), dtype=np.float16)
        # Text Encoder 3 の処理 & 即時解放
        # prompt_embeds = np.zeros((1, 77, 2048), dtype=np.float16)
        # uncond_embeds = np.zeros((1, 77, 2048), dtype=np.float16)
        if config.t5_cache != "load":
            prompt_embeds_3, uncond_embeds_3 = self.text_encoder_3.get_text_embeddings_3(
                config.prompt, config.negative_prompt, auto_mem_free=False)

        if config.t5_cache == "load":
            import tensors_io as tio
            prompt_embeds_3 = tio.load(r"prompt_embeds_3.npy")
            uncond_embeds_3 = tio.load(r"uncond_embeds_3.npy")
        elif config.t5_cache == "save":
            import tensors_io as tio
            tio.save(r"./", "prompt_embeds_3.npy", prompt_embeds_3)
            tio.save(r"./", "uncond_embeds_3.npy", uncond_embeds_3)

        prompt_embeds = self.pad_for_sd35(prompt_embeds, prompt_embeds_3)
        uncond_embeds = self.pad_for_sd35(uncond_embeds, uncond_embeds_3)

        # print(prompt_embeds)

        # prompt_embeds = np.zeros((1, 154, 4096), dtype=np.float16)
        # uncond_embeds = np.zeros((1, 154, 4096), dtype=np.float16)
        # pooled_prompt_embeds = np.zeros((1, 2048), dtype=np.float16)
        # uncond_pooled_embeds = np.zeros((1, 2048), dtype=np.float16)

        self.text_encoder.free_memory()
        self.text_encoder_2.free_memory()
        self.text_encoder_3.free_memory()
        return prompt_embeds, pooled_prompt_embeds, uncond_embeds, uncond_pooled_embeds
