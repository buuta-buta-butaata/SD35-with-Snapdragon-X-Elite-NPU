https://huggingface.co/Buuta/sd-3.5-medium-turbo-for-Snapdragon-X-Elite
上記からダウンロードしたモデルをここに置いてください

次のディレクトリが必要
- text_encoder
- text_encoder_2
- text_encoder_3
- tokenizer
- tokenizer_2
- tokenizer_3
- transformer
- vae_decoder

ダウンロード用のスクリプトも用意したので、そちらを利用してもOK
このファイルと同じディレクトリにあるものを使います
コマンドは以下の通り

pip install -r requirements_download.txt
python download.py

