# SD3.5-Medium with Snapdragon X Elite NPU

## 本プロジェクトの目的

Snapdragon X EliteのNPUを用いて、SD3.5-Medium系のモデルを動かすこと(fp16のモデルで達成済み)  
前のプロジェクト: [SDXL version](https://github.com/buuta-buta-butaata/SDXL-with-Snapdragon-X-Elite-NPU)

注意: 本プロジェクトはあくまでNPU上で動かすことを目的とした概念実証なので、実用性はほとんどないです(画像の生成に時間がかかる)

![生成した画像(Hello SD3.5 Medium with Snapdragon X Elite)](/Hello.png)

画像内の単語にスペルミスはあるが、許してほしい(´・ω・｀)  
30回くらい生成しなおして、一番ましな結果がこれだった  
snapdragonと1つの単語にすると、なかなかうまく文字を描画できなかったので、snap dragonと2つの単語にわけていたりする

## 特徴

T5xxl、transformer(MMDiT-X)もNPUで動くよ！  
turboモデルで、8steps、モデルのロードの時間を含めて合計処理時間が144.479秒  
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

## モデルの動かし方

### 前提条件

#### 必要な環境

- Snapdragon X EliteでRAM16GB以上
- Windows11
- Python 3.13.3(Arm64)

Snapdragon X Elite用に最適化しているため、SoCは必須条件です。  
RAMは実行時点で空きが8GBあればうれしいですが、仮想メモリが十分(8GB以上)確保できれば動きます。  
Pythonのバージョンは3.13.X(Arm64)なら動くと思います。

#### 必要な知識・技能

- Pythonをwindows11上で動かせること

申し訳ないのですが、以降はPythonをある程度動かせる人向けに説明します。

簡単に動かすスクリプトを用意しました。  
とりあえず動く、雑なものです。

### 準備

#### 1.ここにあるスクリプトのダウンロード
gitを使うなどしてダウンロードしてください。

コマンド例:
`
git clone https://github.com/buuta-buta-butaata/SD35-with-Snapdragon-X-Elite-NPU.git
cd SD35-with-Snapdragon-X-Elite-NPU
`

#### 2.モデルのダウンロード
ブラウザかPythonのスクリプトを利用してダウンロードしてください。
注意: ファイルサイズがモデル全体で16.1GBあるので、ダウンロードには時間がかかります

##### ブラウザを使う場合

[sd-3.5-medium-turbo-for-Snapdragon-X-Elite](https://huggingface.co/Buuta/sd-3.5-medium-turbo-for-Snapdragon-X-Elite/tree/main)

上記のモデルを`compiled_models\sd-3.5-medium-turbo-for-Snapdragon-X-Elite`ディレクトリに入れてください。

##### Pythonを使う場合

ダウンロード用のスクリプトを用意してあるので、そちらを実行してください。
以下を実行してください。

```
cd compiled_models\sd-3.5-medium-turbo-for-Snapdragon-X-Elite
pip install -r requirements_download.txt
python download.py
cd ..\..
```

#### 3.Pythonの準備

`requirements.txt`を用意したので、ご利用ください。
```
pip install -r requirements.txt
```

### 実行

準備が完了したら、`image_gen.bat`を実行します。  
引数でプロンプトを与えればOKです。  

実行例(冒頭のHello SD3.5 Mediumの画像):
```
image_gen.bat "a photo of a cat holding a sign that says “Hello SD3.5 with Snap dragon X Elite”" --steps 8 --cfg 1.3 --slg 1 --seed 1996858601
```

ちなみに
```
image_gen.bat --help
```
で、やる気のないオプションの説明が表示されます。


## 問題点

### 生成が遅い

#### transfomer(MMDiT)が1stepあたり約12秒

量子化したらどの程度速くなるかは気になるところだが、モデル全体で16.1GBあるので、int8量子化しても8GBにはおさまらない  
そのため、RAM16GB環境でモデル全体をRAM上において処理することは現実的ではなく、やる気がおきない  
RAMが32GBとかあれば、また別なのだけど

#### T5xxlのボトルネックはモデルのロード時間

T5xxlのモデルのロードに23.9秒、その後の処理(推論)に0.18秒
推論は十分に速いのだが、ロードが遅すぎる(´・ω・｀)

## 今後の発展

前述の問題点があるため、正直なところ発展性はない……と思っていたが、
QualcommがGenieXを発表したので、改めてSnapdragon X Elite上でのSD3.5 Mediumの実用性を考えてみる

GenieXではggufを利用できる  
t5xxlでggufを利用すればint4のモデルを使えるらしいので、ファイルサイズが9GBから2.9GBに減る  
transfomer(MMDiT)をint8量子化すればファイルサイズが半分の2.3GBくらいになるはずなので、RAMにモデル全体がのっかる可能性が見えてくる
RAMが16GBでもギリギリ使える範囲になる……かな？ggufが推論時にどれだけRAMを使うかによるけど  
少なくとも24GB以上あれば余裕で動く範囲におさまるので、実用性が出てきそう  
GenieXは可能性を感じるね  

## 謝辞

今回の成果もQualcomm AI Hub WorkbenchとGoogle Geminiのおかげです。  
すばらしいサービスを提供する両社に改めて感謝を。

