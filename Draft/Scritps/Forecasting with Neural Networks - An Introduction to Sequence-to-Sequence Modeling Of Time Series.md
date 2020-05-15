title: Forecasting with Neural Networks - An Introduction to Sequence-to-Sequence Modeling Of Time Series
Date: 2019-10-14 13:01
Category: Time Series Forecasting, Neural Networks, CNN, RNN
Tags: Time Series Forecasting, Neural Networks, CNN, RNN
Slug: Time Series Forecasting


Author: Mohcine Madkour
Email:mohcine.madkour@gmail.com

# 1. Introcuction to Sequence to Dequence learning 


In traditional time series forecasting, series are often considered on an individual basis, and predictive models are then fit with series-specific parameters. An example of this style is the classic **Autoregressive Integrated Moving Average (ARIMA)** model. Series-specific models can often make quite good predictions, but unfortunately they do not scale well to problems where the number of series to forecast extends to thousands or even hundreds of thousands of series. Additionally, fitting series-specific models fails to capture the expressive general patterns that can be learned from studying many fundamentally related series. From the examples above, we can see that this challenging “high-dimensional” time series setting is faced by many companies.

Luckily, multi-step time series forecasting can be expressed as a sequence-to-sequence supervised prediction problem, a framework amenable to modern neural network models. At the cost of added complexity in constructing and tuning the model, it’s possible to capture the entire predictive problem across all the series with one model. Since neural networks are natural feature learners, it’s also possible to take a minimalistic approach to feature engineering when preparing the model. And when exogenous variables do need to be integrated into the model (e.g. product category, website language, day of week, etc.), it’s simple due to the flexibility of neural network architectures. If you’re not already sold on the potential power of this approach, check out the [DeepAR](https://arxiv.org/pdf/1704.04110.pdf) model that Amazon uses to forecast demand across a massive quantity of products.

# 2. How does seq2seq work exactly?

So how does seq2seq work exactly? I’ll give a high level description. Let’s first consider it in its original application domain as described by this [2014 paper](https://arxiv.org/abs/1409.3215), machine translation. Here’s a visualization summary, taken from [Fariz Rahman’s repo](https://github.com/farizrahman4u/seq2seq)

![](./images/seq2seq_lang.png)

The model uses an “encoder-decoder” framework, mapping an arbitrarily long input sequence to an arbitrarily long output sequence with an intermediate encoded state. You can think of the encoded state as a representation of the entire history of the sequence that provides the context the decoder needs to generate an output sequence. In this case, the encoded state stores the neural network’s “understanding” of the sentence it’s read. This understanding is produced by iteratively “reading” each of the input words using an [LSTM architecture](http://colah.github.io/posts/2015-08-Understanding-LSTMs/) (or similar type of recurrent neural network), which produces the final state vectors that give the encoding. The decoder LSTM then takes these encoded state vectors for its initial state, iteratively “writing” each output and updating its internal state.


Seq2Seq contains modular and reusable layers that you can use to build your own seq2seq models as well as built-in models that work out of the box. Seq2Seq models can be compiled as they are or added as layers to a bigger model. Every Seq2Seq model has 2 primary layers : the encoder and the decoder. Generally, the encoder encodes the input sequence to an internal representation called 'context vector' which is used by the decoder to generate the output sequence. The lengths of input and output sequences can be different, as there is no explicit one on one relation between the input and output sequences. In addition to the encoder and decoder layers, a Seq2Seq model may also contain layers such as the left-stack (Stacked LSTMs on the encoder side), the right-stack (Stacked LSTMs on the decoder side), resizers (for shape compatibility between the encoder and the decoder) and dropout layers to avoid overfitting. The source code is heavily documented, so lets go straight to the examples:

A simple Seq2Seq model:

	import seq2seq
	from seq2seq.models import SimpleSeq2Seq
	model = SimpleSeq2Seq(input_dim=5, hidden_dim=10, output_length=8, output_dim=8)
	model.compile(loss='mse', optimizer='rmsprop')

That's it! You have successfully compiled a minimal Seq2Seq model! Next, let's build a 6 layer deep Seq2Seq model (3 layers for encoding, 3 layers for decoding).

Note that only some of the language I just used was specific to NLP – we can replace “word” with “token” or “value”, and easily generalize to sequences from many problem domains. In particular, we can move from mapping between input and output sentences to mapping between time series. In time series forecasting, what we do is translate the past into the future. We encode the entire history of a series including useful patterns like seasonality and trend, conditioning on this encoding to make future predictions.

To clarify this further, here’s an excellent visual from Artur Suilin. I highly recommend checking out his [repo](https://github.com/Arturus/kaggle-web-traffic) with a state of the art time series seq2seq tensorflow model if you’re interested in this subject.

![](./images/seq2seq_ts.png)

As a proof of concept, let’s see a basic version of this model be applied to a real dataset - [daily wikipedia webpage traffic](https://www.kaggle.com/c/web-traffic-time-series-forecasting). If you’re interested in recreating this yourself, you can use the [accompanying notebook](https://github.com/mohcinemadkour/TimeSeries_Seq2Seq/blob/master/notebooks/TS_Seq2Seq_Intro.ipynb) I’ve written. The plot below shows predictions generated by a seq2seq model for an encoder/target series pair within a time range that the model was not trained on (shifted forward vs. training time range). Clearly these are not the best predictions, but the model is definitely able to pick up on trends in the data, without the use of any feature engineering.

![](./images/seq2seq_preds.png)




