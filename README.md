# conv_p2c
Pytorchで学習したパラメータをc言語の多次元配列形式に変換する。  
conv_p2c.pyを実行すると、model_sample.pthから変換されたnn_parameter.cが生成される。  
model_sample.pthは全結合2層。  
Conv2dレイヤー入りのCNNでも動作確認したが、どんなネットワークでも変換できることは保証できない。  
