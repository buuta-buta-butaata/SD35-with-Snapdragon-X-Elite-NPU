# SD3.5-Medium with Snapdragon X Elite NPU

## 本プロジェクトの目的

Snapdragon X EliteのNPUを用いて、SD3.5-Medium系のモデルを動かすこと(達成済み)  
前のプロジェクト: [SDXL version](https://github.com/buuta-buta-butaata/SDXL-with-Snapdragon-X-Elite-NPU)

![生成した画像(Hello)](/Hello.png)

画像内の単語にスペルミスはあるが、許してほしい(´・ω・｀)  
30回くらい生成しなおして、一番ましな結果がこれだった  
snapdragonと1つの単語にすると、なかなかうまく文字を描画できなかったので、snap dragonと2つの単語にわけていたりする

## 特徴

T5xxl、transformer(MMDiT-X)もNPUで動かせる！  
turboモデルで、8steps、T5xxlの処理含めて合計処理時間が144.479秒
推論中のピークRAMが5.82 GB

T5xxlが動いているため、位置関係、材質などもきちんと指示可能

例 Prompt: "A studio product shot on a clean gray background. On the left is a golden metallic cube with a small wooden dog sculpture on top. On the right is a vibrant red glass sphere."

![T5xxlの力](/T5xxl_power.png)

プロンプトを参考にしたサイト [SD1.5,SDXL,SD3(Medium),SD3.5(Medium, Large)を雑に比較](https://note.com/redrayz/n/n6688a681635c)

## どうやって実現したか

前のSDXLのプロジェクトと同様に、NPU上で動かすためにモデルを2GB以内になるように分割、プリコンパイルしただけです。

## 実行時の注意

**onnxruntime-qnnのバージョンを必ず2.3.0にしておくこと**  
2.1.1のままだと、RAMの使用量が異常に増加するため

## 制限

- 画像サイズは1024x1024固定

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


