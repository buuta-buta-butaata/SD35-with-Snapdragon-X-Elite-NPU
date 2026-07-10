# SD3.5-Medium with Snapdragon X Elite NPU

## Project Objectives

The primary objective of this project is to run Stable Diffusion 3.5 Medium (SD3.5-Medium) series models utilizing the Snapdragon X Elite NPU. (This has been successfully achieved using FP16 precision models.)
Previous project: [SDXL version](https://github.com/buuta-buta-butaata/SDXL-with-Snapdragon-X-Elite-NPU)

*Note: This project is strictly a Proof of Concept (PoC) focused on making the models operational on the NPU. It currently offers very little practical utility due to the significant amount of time required for image generation.*

![Generated Image (Hello SD3.5 Medium with Snapdragon X Elite)](/Hello.png)

*Please forgive the spelling mistakes in the generated text (´・ω・｀). I regenerated the image about 30 times, and this was the best result. Interestingly, the model struggled to render the text properly when combined as "snapdragon", so I split it into two words: "snap dragon".*

## Features

- **Full NPU Execution:** Both T5xxl and the Transformer (MMDiT-X) components run entirely on the NPU.
- **Performance Benchmarks:** Using a Turbo model with 8 inference steps, the total execution time—including model loading—is **144.479 seconds**.
- **Low Memory Footprint:** The peak RAM usage during inference is only **5.82 GB**.
- **Accurate Text-to-Image Alignment:** Since T5xxl is fully functional, spatial relationships, materials, and other detailed instructions are accurately reflected.

**Prompt Example:** *"A studio product shot on a clean gray background. On the left is a golden metallic cube with a small wooden dog sculpture on top. On the right is a vibrant red glass sphere."*

![Power of T5xxl](/T5xxl_power.png)

*Prompt Reference: [SD1.5,SDXL,SD3(Medium),SD3.5(Medium, Large)を雑に比較 (Rough comparison of SD1.5, SDXL, SD3, and SD3.5)](https://note.com)*

## How It Was Achieved

Similar to the previous SDXL project, this was achieved by partitioning and pre-compiling the models into segments under 2GB to ensure compatibility and successful execution on the NPU.

## Execution Notes

- **Required ONNX Runtime Version:** Ensure that your `onnxruntime-qnn` version is strictly set to **2.3.0**. 
- *Using version 2.1.1 causes an abnormal and massive spike in RAM consumption.*

## Limitations

- **Fixed Resolution:** The output image size is strictly limited to **1024x1024** pixels.

## Usage

### Prerequisites

#### System Requirements

- **Processor:** Snapdragon X Elite (Strictly required as this project is optimized specifically for this SoC)
- **RAM:** 16 GB or higher (An available 8 GB of free RAM at execution is ideal, but it will run as long as you have sufficient virtual memory—8 GB or more allocated)
- **OS:** Windows 11
- **Python:** Python 3.13.3 (Arm64) (Any Python 3.13.X Arm64 version should work)

#### Required Skills & Knowledge

- Ability to run and manage Python environments on Windows 11

*Note: The following setup instructions assume a basic proficiency with Python.*

*I have put together a simple execution script. It is quite a crude, "bare minimum" script just to get things running, but it gets the job done.*

### Setup

#### 1. Clone or Download This Repository
Download the project files using Git or by downloading the repository directly.

**Command Example:**
```bash
git clone https://github.com/buuta-buta-butaata/SD35-with-Snapdragon-X-Elite-NPU.git
cd SD35-with-Snapdragon-X-Elite-NPU
```

#### 2. Download the Models
You can download the models either via your web browser or using the provided Python script.
*Note: The total file size for the models is approximately 16.1 GB, so downloading will take some time.*

##### Option A: Via Web Browser

Download the compiled models from the Hugging Face repository:  
[sd-3.5-medium-turbo-for-Snapdragon-X-Elite](https://huggingface.co/Buuta/sd-3.5-medium-turbo-for-Snapdragon-X-Elite/tree/main)

Place all the downloaded model files into the following directory:  
`compiled_models\sd-3.5-medium-turbo-for-Snapdragon-X-Elite`

##### Option B: Via Python Script

A dedicated download script is included in the repository. Run the following commands to download the models automatically:

```cmd
cd compiled_models\sd-3.5-medium-turbo-for-Snapdragon-X-Elite
pip install -r requirements_download.txt
python download.py
cd ..\..
```

#### 3. Install Python Dependencies

Install the required packages using the provided `requirements.txt`:

```cmd
pip install -r requirements.txt
```

### Inference (Running the Model)

Once the setup is complete, you can generate images by executing `image_gen.bat`. Simply pass your prompt as an argument.

**Execution Example (The "Hello SD3.5 Medium" image shown at the top):**
```cmd
image_gen.bat "a photo of a cat holding a sign that says “Hello SD3.5 with Snap dragon X Elite”" --steps 8 --cfg 1.3 --slg 1 --seed 1996858601
```

By the way, running:
```cmd
image_gen.bat --help
```
will display a rather unenthusiastic, bare-bones explanation of the available options.


## Known Issues

### Slow Image Generation

#### Transformer (MMDiT-X) takes approximately 12 seconds per step.
While it would be interesting to see how much faster it could run with quantization, the entire model is currently 16.1 GB. Even with INT8 quantization, it will not fit within an 8 GB footprint. Therefore, keeping the entire model in RAM within a 16 GB system environment is highly impractical, and honestly, I lack the motivation to pursue it further under these constraints. Things might be different if I had a 32 GB RAM environment.

#### T5xxl bottlenecked by loading times.
Loading the T5xxl model takes **23.9 seconds**, while the subsequent inference processing takes only **0.18 seconds**. The inference itself is plenty fast, but the loading time is just way too slow (´・ω・｀).

## Future Outlook

Due to the issues mentioned above, I honestly thought there was no real room for further development. However, Qualcomm's announcement of GenieX has made me reconsider the practical potential of running SD3.5-Medium on the Snapdragon X Elite.

GenieX supports GGUF format models. If we can utilize GGUF for T5xxl, it opens up the possibility of using an INT4 model, which would slash the file size from 9 GB down to 2.9 GB. Furthermore, if we quantize the Transformer (MMDiT-X) to INT8, its file size should drop to roughly 2.3 GB (about half its current size). 

This makes it highly plausible that the entire model could fit comfortably within RAM. Even with a 16 GB RAM system, it might just barely fall into a usable range—depending on how much memory GGUF consumes during inference. At the very least, any system with 24 GB of RAM or more would have plenty of headroom, making this setup genuinely practical. GenieX definitely feels like it holds a lot of promise.

## Acknowledgments

This achievement was made possible once again thanks to the **Qualcomm AI Hub Workbench** and **Google Gemini**. 
My sincere gratitude goes out to both companies for providing such incredible tools and services.

## Project Contributors & Context

* **The Developer** (Human)
* **Google Gemini** (AI Collaborator)

This entire project was brought to life through a tag-team effort between a human engineer determined to squeeze every drop of performance out of the NPU, and an AI co-pilot handling the heavy lifting of debugging and documentation. It stands as a testament to what human-AI synergy can achieve under tight hardware constraints.
