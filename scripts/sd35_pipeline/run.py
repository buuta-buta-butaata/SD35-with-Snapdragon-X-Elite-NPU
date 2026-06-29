import argparse
import numpy as np
import os
import random
import sys
import onnxruntime_qnn as qnn

from diffusers.schedulers.scheduling_utils import KarrasDiffusionSchedulers

from pipeline import SD35Pipeline, SD35Config, SD35Dirs

if qnn.__version__ < "2.2.0":
    print(f"ERROR: The version of onnxruntime-qnn is outdated, so please upgrade it to 2.3.0.")
    print(f"Command: pip install onnxruntime-qnn==2.3.0")
    exit()
    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt", type=str, help="prompt")
    parser.add_argument("--prompt_2", type=str, help="prompt_2")
    parser.add_argument("--steps", type=int, default=8, help="steps")
    parser.add_argument("--negative_prompt", type=str, default="", help="negative prompt")
    parser.add_argument("--negative_prompt_2", type=str, default="", help="negative prompt 2")
    parser.add_argument("--seed", type=int, default=-1, help="seed")
    # parser.add_argument("--layout", choices=["P", "L", "Portrait", "Landscape"], default="", help="画像の形状を指定する。指定がない場合、1024x1024)")
    parser.add_argument("--cfg", type=float, default=1.5, help="classifier free guidance scale")
    parser.add_argument("--slg", type=float, default=0, help="skip layer guidance scale")
    parser.add_argument("--t5_cache", choices=["save", "load"], default="", help="t5 cache")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="画像出力先のディレクトリ")
    return parser.parse_args()

def main():
    args = parse_args()
    if args.steps > 50:
        args.steps = 20
        print(f"ステップ数({args.steps})が大きすぎるため、20に修正")

    print(f"steps: {args.steps}")

    dirs = SD35Dirs()
    dirs.output_dir = args.output_dir

    scheduler_type = KarrasDiffusionSchedulers.EulerAncestralDiscreteScheduler

    config = SD35Config(args.prompt,
                        prompt_2 = args.prompt_2,
                        negative_prompt = args.negative_prompt,
                        negative_prompt_2 = args.negative_prompt_2,
                        seed = args.seed,
                        steps = args.steps,
                        cfg_scale = args.cfg,
                        slg_scale = args.slg,
                        scheduler_type = scheduler_type,
                        t5_cache = args.t5_cache,
                        sd35_dirs = dirs)

    # if args.layout == "P" or args.layout == "Portrait":
    #     config.width = 832
    #     config.height = 1216
    # elif args.layout == "L" or args.layout == "Landscape":
    #     config.width = 1344
    #     config.height = 768
    # else:
    #     config.width = 1024
    #     config.height = 1024
    config.width = 1024
    config.height = 1024

    # config.calib_data_collector=collector
    # pipeline.negative_prompt = get_random_negative_prompt()
    pipe = SD35Pipeline(config)
    pipe.run()

if __name__ == "__main__":
    main()

