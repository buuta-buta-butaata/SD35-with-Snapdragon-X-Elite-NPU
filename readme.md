# SD3.5-Medium with Snapdragon X Elite NPU

## Project Objective
The main goal of this project is to run Stable Diffusion 3.5 Medium (SD3.5-Medium) models natively and entirely on the Snapdragon X Elite NPU using the QNN Execution Provider (Achieved).

* **Previous Project**: [SDXL Version](https://github.com/buuta-buta-butaata/SDXL-with-Snapdragon-X-Elite-NPU)

---

### 🎨 Showcase

#### Hello World from NPU!
Our very first successful text-rendering test on the Hexagon NPU:

![Generated Image (Hello)](/Hello.png)

*Note: Please forgive the minor spelling mistakes in the generated text! (´・ω・｀) It took about 30 prompt iterations to get this result. Interestingly, the model struggled to spell "Snapdragon" as a single word, so splitting it into two words ("snap dragon") significantly improved the NPU's text-rendering accuracy.*

---

## 🔥 Key Features

* **100% NPU Acceleration**: Both the massive **T5xxl text encoder** and the core **MMDiT-X Transformer block** run fully natively on the NPU!
* **Ultra-Low Memory Footprint**: Peak RAM usage during inference is optimized down to just **5.82 GB**.
* **Impressive Lightning Speed**: Utilizing the Turbo version, a full generation takes only **144.479 seconds total** (8 steps, including the heavy T5xxl text encoding process).
* **True Prompt Adherence**: Thanks to the NPU-driven T5xxl encoder, complex spatial relationships, textures, and material prompts are accurately understood and rendered.

#### 📐 Spatial & Material Prompt Example:
> **Prompt**: *"A studio product shot on a clean gray background. On the left is a golden metallic cube with a small wooden dog sculpture on top. On the right is a vibrant red glass sphere."*

![T5xxl Power Demonstration](/T5xxl_power.png)

*(Prompt reference/inspiration from: [Comparing SD1.5, SDXL, SD3, and SD3.5](https://note.com/redrayz/n/n6688a681635c))*

---

## 🛠️ Technical Architecture: How It Works

Consistent with our previous SDXL project, the core breakthrough relies on structural model partitioning. To bypass the QNN compiler constraints and Google Protobuf limits, the massive SD3.5-Medium layers were **strategically split into sub-models under 2 GB each** before undergoing pre-compilation for the Snapdragon NPU.

---

## ⚠️ Critical Dependency Requirement

* **`onnxruntime-qnn==2.3.0` is STRICTLY REQUIRED.**
* **Do NOT use version 2.1.1.** Keeping the older runtime version causes an abnormal, critical spike in RAM consumption that will crash the pipeline.

---

## 🛑 Current Limitations

* **Fixed Image Resolution**: The pipeline is strictly hardcoded to **1024x1024** output resolution due to pre-compilation constraints. Dynamic resizing is not supported.

## Getting Started

### Prerequisites

#### System Requirements

* **SoC**: Snapdragon X Elite (Strictly required, as the project is optimized specifically for this architecture).
* **OS**: Windows 11 (ARM64).
* **RAM**: 16 GB or higher. 
* **Storage**: At least **25 GB of free disk space** is highly recommended. 
  * *Reason: You need to account for both the 16 GB compiled model files and the additional allocation required for the Windows Virtual Memory (Pagefile).*
* **Python**: Python 3.13.x (ARM64 Native).
  * *Note: While it should theoretically run on Linux with minor script modifications, this repository currently only supports Windows.*

### Setup Instructions

#### 1. Clone the Repository
Clone this repository to your local machine using Git:
```bash
git clone https://github.com/buuta-buta-butaata/SD35-with-Snapdragon-X-Elite-NPU.git
cd SD35-with-Snapdragon-X-Elite-NPU
```

#### 2. Download the Model
You can download the pre-compiled models either via your browser or using the provided Python script.

*⚠️ **Important Note on Downloading:** The total model size is around **16 GB**. Depending on your network speed, the download may take some time (expect **15+ minutes**). Please be patient while the script fetches the files.*

##### Method A: Via Web Browser
1. Go to the Hugging Face repository: [sd-3.5-medium-turbo-for-Snapdragon-X-Elite](https://huggingface.co/Buuta/sd-3.5-medium-turbo-for-Snapdragon-X-Elite/tree/main)
2. Download the files and place them into the following directory:
   `compiled_models\sd-3.5-medium-turbo-for-Snapdragon-X-Elite`

##### Method B: Via Python Script
Run the built-in download script to automatically fetch the models:
```bash
cd compiled_models\sd-3.5-medium-turbo-for-Snapdragon-X-Elite
pip install -r requirements_download.txt
python download.py
cd ..\..
```
*(The script will automatically save the models to the correct directory).*

#### 3. Install Dependencies
Ensure all required Python packages are installed:
```bash
pip install -r requirements.txt
```

---

### Running the Text-to-Image Generation

Once the setup is complete, you can generate images by running `image_gen.bat` and passing your prompt as an argument. 

#### Execution Examples

image_gen.bat "A studio product shot on a clean gray background. On the left is a golden metallic cube with a small wooden dog sculpture on top. On the right is a vibrant red glass sphere."


