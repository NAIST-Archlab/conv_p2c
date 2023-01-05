# conv_p2c
PyTorchで学習したニューラルネットワークのパラメータをC言語の多次元配列形式に変換することができる。  
conv_p2c.pyを実行すると、model_sample.pthから変換されたnn_parameter.cが生成される。  
model_sample.pthは全結合2層。  
Conv2dレイヤー入りのCNNでも動作確認したが、どんなネットワークでも変換できることは保証できない。   
パラメータの次元を問わず使えるはずである。  
学習はPyTorch + GPUで簡単に、推論は何らかのアクセラレータでやってみたい人に。  
参考：https://msyksphinz.hatenablog.com/entry/2018/03/24/040000
