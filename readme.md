# SD3.5-Medium with Snapdragon X Elite NPU

## 本プロジェクトの目的

Snapdragon X EliteのNPUを用いて、SD3.5-Medium系のモデルを動かすこと(達成済み)  

![Hello](/Hello.png)

スペルミスはあるが、許してほしい(´・ω・｀)  
30回くらい生成しなおして、一番ましな結果がこれだった  
snapdragonと1つの単語にすると、なかなかうまく文字を描画できなかったので、snap dragonと2つの単語にわけていたりする

readmeの記述や、モデルのアップロードはまだ途中  

## 特徴

T5xxl、transformer(MMDiT-X)もNPUで動かせる！  
turboモデルで、8steps、T5xxlの処理含めて合計処理時間が144.479秒

T5xxlが動いているため、位置関係、材質などもきちんと指示可能

Prompt: "A studio product shot on a clean gray background. On the left is a golden metallic cube with a small wooden dog sculpture on top. On the right is a vibrant red glass sphere."

![T5xxl](/T5xxl_power.png)


ref. [SD1.5,SDXL,SD3(Medium),SD3.5(Medium, Large)を雑に比較](https://note.com/redrayz/n/n6688a681635c)

## 実行時の注意

**onnxruntime-qnnのバージョンを必ず2.3.0にしておくこと**  
2.1.1のままだと、RAMの使用量が異常に増加するため

## 実行例

image_gen.bat "A studio product shot on a clean gray background. On the left is a golden metallic cube with a small wooden dog sculpture on top. On the right is a vibrant red glass sphere."

## これからやること

- モデルのアップロード
- readme.mdをちゃんと書く
- 可能ならばT5xxlをCPUで動かしたときの処理速度を検証しておきたい

## 制限

- 画像サイズは1024x1024固定
