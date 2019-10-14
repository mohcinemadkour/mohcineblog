title: High-Dimensional Time Series Forecasting with Convolutional Neural Networks: Full-Fledged WaveNet
Date: 2019-10-14 13:01
Category: Time Series Forecasting, Convolutional Neural Networks
Tags: Time Series Forecasting, Convolutional Neural Networks
Slug: Time Series Forecasting, Convolutional Neural Networks
Author: Mohcine Madkour
Email:mohcine.madkour@gmail.com



This notebook expands on the [previous notebook in this series](https://github.com/mohcinemadkour/TimeSeries_Seq2Seq/blob/master/notebooks/TS_Seq2Seq_Conv_Intro.ipynb), demonstrating in python/keras code how a **convolutional** sequence-to-sequence neural network modeled after WaveNet can be built for the purpose of high-dimensional time series forecasting. I assume working familiarity with **dilated causal convolutions** (WaveNet's core building block), and recommend referencing the 3rd section of the previous notebook if you need to review the concept.

For an introduction to neural network forecasting with an LSTM architecture, check out the [first notebook in this series](https://github.com/mohcinemadkour/TimeSeries_Seq2Seq/blob/master/notebooks/TS_Seq2Seq_Intro.ipynb).   

In this notebook I'll be using the daily wikipedia web page traffic dataset again, available [here on Kaggle](https://www.kaggle.com/c/web-traffic-time-series-forecasting/data). The corresponding competition called for forecasting 60 days into the future, which we'll now mirror in this demonstration of a full-fledged model. Once again we'll use all of the series history available in "train_1.csv" for the encoding stage of the model. 

Our goal here is to expand on the previous notebook's simple WaveNet implementation, adding additional architecture components from the [original model](https://arxiv.org/pdf/1609.03499.pdf). In particular, each convolutional block of our network will incorporate **gated activations**, **residual connections**, and **skip connections** in addition to the dilated causal convolutions we saw in the previous notebook. I'll explain how these three new mechanisms work in section 3. Feel free to skip ahead to that section if you're comfortable with the data setup and formatting steps (as in the previous notebooks), and want to get right into the neural network.    

**Note**: for a written overview on this topic, check out my two blog posts that walk through the core concepts behind WaveNet - [part 1](https://github.com/mohcinemadkour/TimeSeries_Seq2Seq/blob/master/notebooks/TS_Seq2Seq_Conv_Intro.ipynb), [part 2](https://github.com/mohcinemadkour/TimeSeries_Seq2Seq/blob/master/notebooks/TS_Seq2Seq_Conv_Full.ipynb).  

Here's a section breakdown of this notebook -- enjoy!

**1. Loading and Previewing the Data**   
**2. Formatting the Data for Modeling**  
**3. Building the Model - Training Architecture**  
**4. Building the Model - Inference Loop**  
**5. Generating and Plotting Predictions**

## 1. Loading and Previewing the Data 

First thing's first, let's load up the data and get a quick feel for it (reminder that the dataset is available [here](https://www.kaggle.com/c/web-traffic-time-series-forecasting/data)). 

Note that there are a good number of NaN values in the data that don't disambiguate missing from zero. For the sake of simplicity in this tutorial, we'll naively fill these with 0 later on.


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

import seaborn as sns
sns.set()

df = pd.read_csv('../data/train_1.csv')
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Page</th>
      <th>2015-07-01</th>
      <th>2015-07-02</th>
      <th>2015-07-03</th>
      <th>2015-07-04</th>
      <th>2015-07-05</th>
      <th>2015-07-06</th>
      <th>2015-07-07</th>
      <th>2015-07-08</th>
      <th>2015-07-09</th>
      <th>...</th>
      <th>2016-12-22</th>
      <th>2016-12-23</th>
      <th>2016-12-24</th>
      <th>2016-12-25</th>
      <th>2016-12-26</th>
      <th>2016-12-27</th>
      <th>2016-12-28</th>
      <th>2016-12-29</th>
      <th>2016-12-30</th>
      <th>2016-12-31</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2NE1_zh.wikipedia.org_all-access_spider</td>
      <td>18.0</td>
      <td>11.0</td>
      <td>5.0</td>
      <td>13.0</td>
      <td>14.0</td>
      <td>9.0</td>
      <td>9.0</td>
      <td>22.0</td>
      <td>26.0</td>
      <td>...</td>
      <td>32.0</td>
      <td>63.0</td>
      <td>15.0</td>
      <td>26.0</td>
      <td>14.0</td>
      <td>20.0</td>
      <td>22.0</td>
      <td>19.0</td>
      <td>18.0</td>
      <td>20.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2PM_zh.wikipedia.org_all-access_spider</td>
      <td>11.0</td>
      <td>14.0</td>
      <td>15.0</td>
      <td>18.0</td>
      <td>11.0</td>
      <td>13.0</td>
      <td>22.0</td>
      <td>11.0</td>
      <td>10.0</td>
      <td>...</td>
      <td>17.0</td>
      <td>42.0</td>
      <td>28.0</td>
      <td>15.0</td>
      <td>9.0</td>
      <td>30.0</td>
      <td>52.0</td>
      <td>45.0</td>
      <td>26.0</td>
      <td>20.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3C_zh.wikipedia.org_all-access_spider</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>4.0</td>
      <td>...</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>7.0</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>6.0</td>
      <td>3.0</td>
      <td>4.0</td>
      <td>17.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4minute_zh.wikipedia.org_all-access_spider</td>
      <td>35.0</td>
      <td>13.0</td>
      <td>10.0</td>
      <td>94.0</td>
      <td>4.0</td>
      <td>26.0</td>
      <td>14.0</td>
      <td>9.0</td>
      <td>11.0</td>
      <td>...</td>
      <td>32.0</td>
      <td>10.0</td>
      <td>26.0</td>
      <td>27.0</td>
      <td>16.0</td>
      <td>11.0</td>
      <td>17.0</td>
      <td>19.0</td>
      <td>10.0</td>
      <td>11.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>52_Hz_I_Love_You_zh.wikipedia.org_all-access_s...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>48.0</td>
      <td>9.0</td>
      <td>25.0</td>
      <td>13.0</td>
      <td>3.0</td>
      <td>11.0</td>
      <td>27.0</td>
      <td>13.0</td>
      <td>36.0</td>
      <td>10.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 551 columns</p>
</div>




```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 145063 entries, 0 to 145062
    Columns: 551 entries, Page to 2016-12-31
    dtypes: float64(550), object(1)
    memory usage: 609.8+ MB
    


```python
data_start_date = df.columns[1]
data_end_date = df.columns[-1]
print('Data ranges from %s to %s' % (data_start_date, data_end_date))
```

    Data ranges from 2015-07-01 to 2016-12-31
    

We can define a function that lets us visualize some random webpage series as below. For the sake of smoothing out the scale of traffic across different series, we apply a log1p transformation before plotting - i.e. take $\log(1+x)$ for each value $x$ in a series.


```python
def plot_random_series(df, n_series):
    
    sample = df.sample(n_series, random_state=8)
    page_labels = sample['Page'].tolist()
    series_samples = sample.loc[:,data_start_date:data_end_date]
    
    plt.figure(figsize=(10,6))
    
    for i in range(series_samples.shape[0]):
        np.log1p(pd.Series(series_samples.iloc[i]).astype(np.float64)).plot(linewidth=1.5)
    
    plt.title('Randomly Selected Wikipedia Page Daily Views Over Time (Log(views) + 1)')
    plt.legend(page_labels)
    
plot_random_series(df, 6)
```


![png](output_6_0d.png)


## 2. Formatting the Data for Modeling 

Sadly we can't just throw the dataframe we've created into keras and let it work its magic. Instead, we have to set up a few data transformation steps to extract nice numpy arrays that we can pass to keras. But even before doing that, we have to know how to appropriately partition the time series into encoding and prediction intervals for the purposes of training and validation. Note that for our simple convolutional model we won't use an encoder-decoder architecture like in the first notebook, but **we'll keep the "encoding" and "decoding" (prediction) terminology to be consistent** -- in this case, the encoding interval represents the entire series history that we will use for the network's feature learning, but not output any predictions on. 

We'll use a style of **walk-forward validation**, where our validation set spans the same time-range as our training set, but shifted forward in time (in this case by 60 days). This way, we simulate how our model will perform on unseen data that comes in the future. 

[Artur Suilin](https://github.com/Arturus/kaggle-web-traffic/blob/master/how_it_works.md) has created a very nice image that visualizes this validation style and contrasts it with traditional validation. I highly recommend checking out his entire repo, as he's implemented a truly state of the art (and competition winning) seq2seq model on this data set. 

![architecture](/images/ArturSuilin_validation.png)

### Train and Validation Series Partioning

We need to create 4 sub-segments of the data:

    1. Train encoding period
    2. Train decoding period (train targets, 60 days)
    3. Validation encoding period
    4. Validation decoding period (validation targets, 60 days)
    
We'll do this by finding the appropriate start and end dates for each segment. Starting from the end of the data we've loaded, we'll work backwards to get validation and training prediction intervals. Then we'll work forward from the start to get training and validation encoding intervals. 


```python
from datetime import timedelta

pred_steps = 60 
pred_length=timedelta(pred_steps)

first_day = pd.to_datetime(data_start_date) 
last_day = pd.to_datetime(data_end_date)

val_pred_start = last_day - pred_length + timedelta(1)
val_pred_end = last_day

train_pred_start = val_pred_start - pred_length
train_pred_end = val_pred_start - timedelta(days=1) 
```


```python
enc_length = train_pred_start - first_day

train_enc_start = first_day
train_enc_end = train_enc_start + enc_length - timedelta(1)

val_enc_start = train_enc_start + pred_length
val_enc_end = val_enc_start + enc_length - timedelta(1) 
```


```python
print('Train encoding:', train_enc_start, '-', train_enc_end)
print('Train prediction:', train_pred_start, '-', train_pred_end, '\n')
print('Val encoding:', val_enc_start, '-', val_enc_end)
print('Val prediction:', val_pred_start, '-', val_pred_end)

print('\nEncoding interval:', enc_length.days)
print('Prediction interval:', pred_length.days)
```

    Train encoding: 2015-07-01 00:00:00 - 2016-09-02 00:00:00
    Train prediction: 2016-09-03 00:00:00 - 2016-11-01 00:00:00 
    
    Val encoding: 2015-08-30 00:00:00 - 2016-11-01 00:00:00
    Val prediction: 2016-11-02 00:00:00 - 2016-12-31 00:00:00
    
    Encoding interval: 430
    Prediction interval: 60
    

### Keras Data Formatting

Now that we have the time segment dates, we'll define the functions we need to extract the data in keras friendly format. Here are the steps:

* Pull the time series into an array, save a date_to_index mapping as a utility for referencing into the array 
* Create function to extract specified time interval from all the series 
* Create functions to transform all the series. 
    - Here we smooth out the scale by taking log1p and de-meaning each series using the encoder series mean, then reshape to the **(n_series, n_timesteps, n_features) tensor format** that keras will expect. 
    - Note that if we want to generate true predictions instead of log scale ones, we can easily apply a reverse transformation at prediction time. 


```python
date_to_index = pd.Series(index=pd.Index([pd.to_datetime(c) for c in df.columns[1:]]),
                          data=[i for i in range(len(df.columns[1:]))])

series_array = df[df.columns[1:]].values

def get_time_block_series(series_array, date_to_index, start_date, end_date):
    
    inds = date_to_index[start_date:end_date]
    return series_array[:,inds]

def transform_series_encode(series_array):
    
    series_array = np.log1p(np.nan_to_num(series_array)) # filling NaN with 0
    series_mean = series_array.mean(axis=1).reshape(-1,1) 
    series_array = series_array - series_mean
    series_array = series_array.reshape((series_array.shape[0],series_array.shape[1], 1))
    
    return series_array, series_mean

def transform_series_decode(series_array, encode_series_mean):
    
    series_array = np.log1p(np.nan_to_num(series_array)) # filling NaN with 0
    series_array = series_array - encode_series_mean
    series_array = series_array.reshape((series_array.shape[0],series_array.shape[1], 1))
    
    return series_array
```

## 3. Building the Model - Architecture

This convolutional architecture is a full-fledged version of the [WaveNet model](https://deepmind.com/blog/wavenet-generative-model-raw-audio/), designed as a generative model for audio (in particular, for text-to-speech applications). The wavenet model can be abstracted beyond audio to apply to any time series forecasting problem, providing a nice structure for capturing long-term dependencies without an excessive number of learned weights.

The core building block of the wavenet model is the **dilated causal convolution layer**, discussed in detail in the [previous notebook of this series](https://github.com/JEddy92/TimeSeries_Seq2Seq/blob/master/notebooks/TS_Seq2Seq_Conv_Intro.ipynb) as well as the [accompanying blog post](https://jeddy92.github.io/JEddy92.github.io/ts_seq2seq_conv/). In summary, this style of convolution properly handles temporal flow and allows the receptive field of outputs to increase exponentially as a function of the number of layers. This structure is nicely visualized by the below diagram from the wavenet paper. 

![dilatedconv](/images/WaveNet_dilatedconv.png)

The model also utilizes some other key techniques: **gated activations**, **residual connections**, and **skip connections**. I'll introduce and explain these techniques, then show how to implement our full-fledged WaveNet architecture in keras. The WaveNet paper diagram below details how the model's components fit together block by block into a stack of operations, so we'll use it as a handy reference as we go (note that there are slight discrepancies between the diagram and what we implement, e.g. the original WaveNet has a softmax classification rather than regression output).   

![blocks](/images/WaveNet_residblock.png)

### **Gated Activations**

In the boxed portion of the architecture diagram, you'll notice that the dilated convolution output splits into two branches that are later recombined via element-wise multiplication. This depicts a *gated activation unit*, where we interpret the *tanh* activation branch as a learned filter and the *sigmoid* activation branch as a learned gate that regulates the information flow from the filter. If this reminds you of the gating mechanisms used in [LSTMs or GRUs](http://colah.github.io/posts/2015-08-Understanding-LSTMs/) you're on point, as those models use the same style of information gating to control adjustments to their cell states.

In mathematical notation, this means we map a convolutional block's input $x$ to output $z$ via the below, where $W_f$ and $W_g$ correspond to (learned) dilated causal convolution weights:

$$ z = tanh(W_f * x) \odot \sigma(W_g * x) $$

Why use gated activations instead of the more standard *ReLU* activation? The WaveNet designers found that gated activations saw stronger performance empirically than ReLU activations for audio data, and this outperformance may extend broadly to time series data. Perhaps the [sparsity induced by ReLU activations](http://proceedings.mlr.press/v15/glorot11a.html) is not as well suited to time series forecasting as it is to other problem domains, or gated activations allow for smoother information (gradient) flow over a many-layered WaveNet architecture. However, this choice of activation is certainly not set in stone and I'd be interested to see a results comparison when trying ReLU instead. With that caveat, we'll be sticking with the gated activations in the interest of learning about the full original architecture.            

### **Residual and Skip Connections**

In traditional neural network architectures, a neuron layer takes direct input only from the layer that precedes it, so early layers influence deeper layers via a heirarchy of intermediate computations. In theory, this heirarchy allows the network to properly build up high-level predictive features off of lower-level/raw signals. For example, in image classification problems, neural nets start from raw pixel values, find generic geometric and textural patterns, then combine these generic patterns to construct fine-grained representations of the features that identify specific object types.

But what if lower-level signals are actually immediately useful for prediction, and may be at risk of distortion as they're passed through a complex heirarchy of computations? We could always simplify the heirarchy by using fewer layers and units, but what if we want the best of both worlds: direct, unfiltered low-level signals and nuanced heirarchical representations? One avenue for addressing this problem is provided by **skip connections**, which act to preserve earlier feature layer outputs as the network passes forward signals for final prediction processing. To build intuition for why we would want a mix of feature complexities in our problem domain, consider the wide range of time series drivers - there are strong and direct autoregressive components, moderately more sophisticated trend and seasonality components, and idiosyncratic trajectories that are difficult to spot with the human eye.        

To leverage skip connections, we can simply store the tensor output of each convolutional block in addition to passing it through further blocks (or choose select blocks to store output from). At the end of the block heirarchy, we then have a collection of feature outputs at *all levels of the heirarchy*, rather than a singular set of maximally complex feature outputs. This collection of outputs is then combined for final processing, typically via concatenation or addition (we'll use the latter).

With this in mind, return to the WaveNet block diagram above, and notice how for each block in the stack, the post-convolution gated activations pass through to the set of skip connections. This visualizes the tensor output storage and eventual combination just described. Note that the frequency and structure of skip connections is fully customizable and can be chosen experimentally and via domain expertise - as an example of an alternate skip connection structure, check out this convolutional architecture from a [semantic segmentation paper](https://www.researchgate.net/publication/327330378_Semantic_Segmentation_Based_on_Deep_Convolution_Neural_Network).

![CNN_skips](/images/CNN_skips.png)

**Residual connections** are closely related to skip connections; in fact, they can be viewed as specialized, short skips further into the network (often and in our case just one layer). With residual connections, we think of mapping a network block's input to output via $x_{out} = f(x_{in}) + x_{in}$ instead of using the traditional direct mapping $x_{out} = f(x_{in})$, for some function $f$ that corresponds to the model's learned weights. This helps allow for the possibility that the model learns a mapping that acts almost as an identity function, with the input passing through nearly unchanged. In the diagram above, such connections are visualized by the rounded arrows grouped with each pair of convolutions.  

Why would this be beneficial? Well, the effectiveness of residual connections is still not fully understood, but a compelling explanation is that they facilitate the use of deeper networks by allowing for more direct gradient flow in backpropagation. It's often difficult to efficienctly train the early layers of a deep network due to the length of the backpropagation chain, but residual and skip connections create an easier information highway. Intuitively, perhaps you can think of both as mechanisms for guarding against overcomputation and intermediate signal loss. You can check out the [ResNet paper](https://arxiv.org/pdf/1512.03385.pdf) that originated the residual connection concept for more discussion and empirical results.

Though our architecture will be shallower than the original WaveNet (fewer convolutional blocks), we'll likely still see some benefit from introducing skip and residual connections at every block. Returning to the WaveNet architecture diagram again, you can see how the residual connection allows each block's input to bypass the convolution stage, and then adds that input to the convolution output. A final point to note is that the diagram's *1x1 convolutions* are really just equivalent to (time-distributed) fully connected layers, and serve in post-processing and standardization capacities. Our setup will use layers of this style (with different filter dimensions) for **post/pre-processing** to facilitate our skip and residual connections, as well as for generating final prediction outputs.           

### **Our Architecture**

With all of our components now laid out, here's what we'll use:

* 16 dilated causal convolutional blocks
    * Preprocessing and postprocessing (time distributed) fully connected layers (convolutions with filter width 1): 16 output units
    * 32 filters of width 2 per block
    * Exponentially increasing dilation rate with a reset (1, 2, 4, 8, ..., 128, 1, 2, ..., 128) 
    * Gated activations
    * Residual and skip connections
* 2 (time distributed) fully connected layers to map sum of skip outputs to final output 

We'll extract the last 60 steps from the output sequence as our predicted output for training. We'll use teacher forcing again during training. Similarly to the previous notebook, we'll have a separate function that runs an inference loop to generate predictions on unseen data, iteratively filling previous predictions into the history sequence (section 4). 


```python
from keras.models import Model
from keras.layers import Input, Conv1D, Dense, Activation, Dropout, Lambda, Multiply, Add, Concatenate
from keras.optimizers import Adam

# convolutional operation parameters
n_filters = 32 # 32 
filter_width = 2
dilation_rates = [2**i for i in range(8)] * 2 

# define an input history series and pass it through a stack of dilated causal convolution blocks. 
history_seq = Input(shape=(None, 1))
x = history_seq

skips = []
for dilation_rate in dilation_rates:
    
    # preprocessing - equivalent to time-distributed dense
    x = Conv1D(16, 1, padding='same', activation='relu')(x) 
    
    # filter convolution
    x_f = Conv1D(filters=n_filters,
                 kernel_size=filter_width, 
                 padding='causal',
                 dilation_rate=dilation_rate)(x)
    
    # gating convolution
    x_g = Conv1D(filters=n_filters,
                 kernel_size=filter_width, 
                 padding='causal',
                 dilation_rate=dilation_rate)(x)
    
    # multiply filter and gating branches
    z = Multiply()([Activation('tanh')(x_f),
                    Activation('sigmoid')(x_g)])
    
    # postprocessing - equivalent to time-distributed dense
    z = Conv1D(16, 1, padding='same', activation='relu')(z)
    
    # residual connection
    x = Add()([x, z])    
    
    # collect skip connections
    skips.append(z)

# add all skip connection outputs 
out = Activation('relu')(Add()(skips))

# final time-distributed dense layers 
out = Conv1D(128, 1, padding='same')(out)
out = Activation('relu')(out)
out = Dropout(.2)(out)
out = Conv1D(1, 1, padding='same')(out)

# extract the last 60 time steps as the training target
def slice(x, seq_length):
    return x[:,-seq_length:,:]

pred_seq_train = Lambda(slice, arguments={'seq_length':60})(out)

model = Model(history_seq, pred_seq_train)
model.compile(Adam(), loss='mean_absolute_error')
```

    /anaconda3/lib/python3.6/site-packages/h5py/__init__.py:34: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
      from ._conv import register_converters as _register_converters
    Using TensorFlow backend.
    


```python
model.summary()
```

    __________________________________________________________________________________________________
    Layer (type)                    Output Shape         Param #     Connected to                     
    ==================================================================================================
    input_1 (InputLayer)            (None, None, 1)      0                                            
    __________________________________________________________________________________________________
    conv1d_1 (Conv1D)               (None, None, 16)     32          input_1[0][0]                    
    __________________________________________________________________________________________________
    conv1d_2 (Conv1D)               (None, None, 32)     1056        conv1d_1[0][0]                   
    __________________________________________________________________________________________________
    conv1d_3 (Conv1D)               (None, None, 32)     1056        conv1d_1[0][0]                   
    __________________________________________________________________________________________________
    activation_1 (Activation)       (None, None, 32)     0           conv1d_2[0][0]                   
    __________________________________________________________________________________________________
    activation_2 (Activation)       (None, None, 32)     0           conv1d_3[0][0]                   
    __________________________________________________________________________________________________
    multiply_1 (Multiply)           (None, None, 32)     0           activation_1[0][0]               
                                                                     activation_2[0][0]               
    __________________________________________________________________________________________________
    conv1d_4 (Conv1D)               (None, None, 16)     528         multiply_1[0][0]                 
    __________________________________________________________________________________________________
    add_1 (Add)                     (None, None, 16)     0           conv1d_1[0][0]                   
                                                                     conv1d_4[0][0]                   
    __________________________________________________________________________________________________
    conv1d_5 (Conv1D)               (None, None, 16)     272         add_1[0][0]                      
    __________________________________________________________________________________________________
    conv1d_6 (Conv1D)               (None, None, 32)     1056        conv1d_5[0][0]                   
    __________________________________________________________________________________________________
    conv1d_7 (Conv1D)               (None, None, 32)     1056        conv1d_5[0][0]                   
    __________________________________________________________________________________________________
    activation_3 (Activation)       (None, None, 32)     0           conv1d_6[0][0]                   
    __________________________________________________________________________________________________
    activation_4 (Activation)       (None, None, 32)     0           conv1d_7[0][0]                   
    __________________________________________________________________________________________________
    multiply_2 (Multiply)           (None, None, 32)     0           activation_3[0][0]               
                                                                     activation_4[0][0]               
    __________________________________________________________________________________________________
    conv1d_8 (Conv1D)               (None, None, 16)     528         multiply_2[0][0]                 
    __________________________________________________________________________________________________
    add_2 (Add)                     (None, None, 16)     0           conv1d_5[0][0]                   
                                                                     conv1d_8[0][0]                   
    __________________________________________________________________________________________________
    conv1d_9 (Conv1D)               (None, None, 16)     272         add_2[0][0]                      
    __________________________________________________________________________________________________
    conv1d_10 (Conv1D)              (None, None, 32)     1056        conv1d_9[0][0]                   
    __________________________________________________________________________________________________
    conv1d_11 (Conv1D)              (None, None, 32)     1056        conv1d_9[0][0]                   
    __________________________________________________________________________________________________
    activation_5 (Activation)       (None, None, 32)     0           conv1d_10[0][0]                  
    __________________________________________________________________________________________________
    activation_6 (Activation)       (None, None, 32)     0           conv1d_11[0][0]                  
    __________________________________________________________________________________________________
    multiply_3 (Multiply)           (None, None, 32)     0           activation_5[0][0]               
                                                                     activation_6[0][0]               
    __________________________________________________________________________________________________
    conv1d_12 (Conv1D)              (None, None, 16)     528         multiply_3[0][0]                 
    __________________________________________________________________________________________________
    add_3 (Add)                     (None, None, 16)     0           conv1d_9[0][0]                   
                                                                     conv1d_12[0][0]                  
    __________________________________________________________________________________________________
    conv1d_13 (Conv1D)              (None, None, 16)     272         add_3[0][0]                      
    __________________________________________________________________________________________________
    conv1d_14 (Conv1D)              (None, None, 32)     1056        conv1d_13[0][0]                  
    __________________________________________________________________________________________________
    conv1d_15 (Conv1D)              (None, None, 32)     1056        conv1d_13[0][0]                  
    __________________________________________________________________________________________________
    activation_7 (Activation)       (None, None, 32)     0           conv1d_14[0][0]                  
    __________________________________________________________________________________________________
    activation_8 (Activation)       (None, None, 32)     0           conv1d_15[0][0]                  
    __________________________________________________________________________________________________
    multiply_4 (Multiply)           (None, None, 32)     0           activation_7[0][0]               
                                                                     activation_8[0][0]               
    __________________________________________________________________________________________________
    conv1d_16 (Conv1D)              (None, None, 16)     528         multiply_4[0][0]                 
    __________________________________________________________________________________________________
    add_4 (Add)                     (None, None, 16)     0           conv1d_13[0][0]                  
                                                                     conv1d_16[0][0]                  
    __________________________________________________________________________________________________
    conv1d_17 (Conv1D)              (None, None, 16)     272         add_4[0][0]                      
    __________________________________________________________________________________________________
    conv1d_18 (Conv1D)              (None, None, 32)     1056        conv1d_17[0][0]                  
    __________________________________________________________________________________________________
    conv1d_19 (Conv1D)              (None, None, 32)     1056        conv1d_17[0][0]                  
    __________________________________________________________________________________________________
    activation_9 (Activation)       (None, None, 32)     0           conv1d_18[0][0]                  
    __________________________________________________________________________________________________
    activation_10 (Activation)      (None, None, 32)     0           conv1d_19[0][0]                  
    __________________________________________________________________________________________________
    multiply_5 (Multiply)           (None, None, 32)     0           activation_9[0][0]               
                                                                     activation_10[0][0]              
    __________________________________________________________________________________________________
    conv1d_20 (Conv1D)              (None, None, 16)     528         multiply_5[0][0]                 
    __________________________________________________________________________________________________
    add_5 (Add)                     (None, None, 16)     0           conv1d_17[0][0]                  
                                                                     conv1d_20[0][0]                  
    __________________________________________________________________________________________________
    conv1d_21 (Conv1D)              (None, None, 16)     272         add_5[0][0]                      
    __________________________________________________________________________________________________
    conv1d_22 (Conv1D)              (None, None, 32)     1056        conv1d_21[0][0]                  
    __________________________________________________________________________________________________
    conv1d_23 (Conv1D)              (None, None, 32)     1056        conv1d_21[0][0]                  
    __________________________________________________________________________________________________
    activation_11 (Activation)      (None, None, 32)     0           conv1d_22[0][0]                  
    __________________________________________________________________________________________________
    activation_12 (Activation)      (None, None, 32)     0           conv1d_23[0][0]                  
    __________________________________________________________________________________________________
    multiply_6 (Multiply)           (None, None, 32)     0           activation_11[0][0]              
                                                                     activation_12[0][0]              
    __________________________________________________________________________________________________
    conv1d_24 (Conv1D)              (None, None, 16)     528         multiply_6[0][0]                 
    __________________________________________________________________________________________________
    add_6 (Add)                     (None, None, 16)     0           conv1d_21[0][0]                  
                                                                     conv1d_24[0][0]                  
    __________________________________________________________________________________________________
    conv1d_25 (Conv1D)              (None, None, 16)     272         add_6[0][0]                      
    __________________________________________________________________________________________________
    conv1d_26 (Conv1D)              (None, None, 32)     1056        conv1d_25[0][0]                  
    __________________________________________________________________________________________________
    conv1d_27 (Conv1D)              (None, None, 32)     1056        conv1d_25[0][0]                  
    __________________________________________________________________________________________________
    activation_13 (Activation)      (None, None, 32)     0           conv1d_26[0][0]                  
    __________________________________________________________________________________________________
    activation_14 (Activation)      (None, None, 32)     0           conv1d_27[0][0]                  
    __________________________________________________________________________________________________
    multiply_7 (Multiply)           (None, None, 32)     0           activation_13[0][0]              
                                                                     activation_14[0][0]              
    __________________________________________________________________________________________________
    conv1d_28 (Conv1D)              (None, None, 16)     528         multiply_7[0][0]                 
    __________________________________________________________________________________________________
    add_7 (Add)                     (None, None, 16)     0           conv1d_25[0][0]                  
                                                                     conv1d_28[0][0]                  
    __________________________________________________________________________________________________
    conv1d_29 (Conv1D)              (None, None, 16)     272         add_7[0][0]                      
    __________________________________________________________________________________________________
    conv1d_30 (Conv1D)              (None, None, 32)     1056        conv1d_29[0][0]                  
    __________________________________________________________________________________________________
    conv1d_31 (Conv1D)              (None, None, 32)     1056        conv1d_29[0][0]                  
    __________________________________________________________________________________________________
    activation_15 (Activation)      (None, None, 32)     0           conv1d_30[0][0]                  
    __________________________________________________________________________________________________
    activation_16 (Activation)      (None, None, 32)     0           conv1d_31[0][0]                  
    __________________________________________________________________________________________________
    multiply_8 (Multiply)           (None, None, 32)     0           activation_15[0][0]              
                                                                     activation_16[0][0]              
    __________________________________________________________________________________________________
    conv1d_32 (Conv1D)              (None, None, 16)     528         multiply_8[0][0]                 
    __________________________________________________________________________________________________
    add_8 (Add)                     (None, None, 16)     0           conv1d_29[0][0]                  
                                                                     conv1d_32[0][0]                  
    __________________________________________________________________________________________________
    conv1d_33 (Conv1D)              (None, None, 16)     272         add_8[0][0]                      
    __________________________________________________________________________________________________
    conv1d_34 (Conv1D)              (None, None, 32)     1056        conv1d_33[0][0]                  
    __________________________________________________________________________________________________
    conv1d_35 (Conv1D)              (None, None, 32)     1056        conv1d_33[0][0]                  
    __________________________________________________________________________________________________
    activation_17 (Activation)      (None, None, 32)     0           conv1d_34[0][0]                  
    __________________________________________________________________________________________________
    activation_18 (Activation)      (None, None, 32)     0           conv1d_35[0][0]                  
    __________________________________________________________________________________________________
    multiply_9 (Multiply)           (None, None, 32)     0           activation_17[0][0]              
                                                                     activation_18[0][0]              
    __________________________________________________________________________________________________
    conv1d_36 (Conv1D)              (None, None, 16)     528         multiply_9[0][0]                 
    __________________________________________________________________________________________________
    add_9 (Add)                     (None, None, 16)     0           conv1d_33[0][0]                  
                                                                     conv1d_36[0][0]                  
    __________________________________________________________________________________________________
    conv1d_37 (Conv1D)              (None, None, 16)     272         add_9[0][0]                      
    __________________________________________________________________________________________________
    conv1d_38 (Conv1D)              (None, None, 32)     1056        conv1d_37[0][0]                  
    __________________________________________________________________________________________________
    conv1d_39 (Conv1D)              (None, None, 32)     1056        conv1d_37[0][0]                  
    __________________________________________________________________________________________________
    activation_19 (Activation)      (None, None, 32)     0           conv1d_38[0][0]                  
    __________________________________________________________________________________________________
    activation_20 (Activation)      (None, None, 32)     0           conv1d_39[0][0]                  
    __________________________________________________________________________________________________
    multiply_10 (Multiply)          (None, None, 32)     0           activation_19[0][0]              
                                                                     activation_20[0][0]              
    __________________________________________________________________________________________________
    conv1d_40 (Conv1D)              (None, None, 16)     528         multiply_10[0][0]                
    __________________________________________________________________________________________________
    add_10 (Add)                    (None, None, 16)     0           conv1d_37[0][0]                  
                                                                     conv1d_40[0][0]                  
    __________________________________________________________________________________________________
    conv1d_41 (Conv1D)              (None, None, 16)     272         add_10[0][0]                     
    __________________________________________________________________________________________________
    conv1d_42 (Conv1D)              (None, None, 32)     1056        conv1d_41[0][0]                  
    __________________________________________________________________________________________________
    conv1d_43 (Conv1D)              (None, None, 32)     1056        conv1d_41[0][0]                  
    __________________________________________________________________________________________________
    activation_21 (Activation)      (None, None, 32)     0           conv1d_42[0][0]                  
    __________________________________________________________________________________________________
    activation_22 (Activation)      (None, None, 32)     0           conv1d_43[0][0]                  
    __________________________________________________________________________________________________
    multiply_11 (Multiply)          (None, None, 32)     0           activation_21[0][0]              
                                                                     activation_22[0][0]              
    __________________________________________________________________________________________________
    conv1d_44 (Conv1D)              (None, None, 16)     528         multiply_11[0][0]                
    __________________________________________________________________________________________________
    add_11 (Add)                    (None, None, 16)     0           conv1d_41[0][0]                  
                                                                     conv1d_44[0][0]                  
    __________________________________________________________________________________________________
    conv1d_45 (Conv1D)              (None, None, 16)     272         add_11[0][0]                     
    __________________________________________________________________________________________________
    conv1d_46 (Conv1D)              (None, None, 32)     1056        conv1d_45[0][0]                  
    __________________________________________________________________________________________________
    conv1d_47 (Conv1D)              (None, None, 32)     1056        conv1d_45[0][0]                  
    __________________________________________________________________________________________________
    activation_23 (Activation)      (None, None, 32)     0           conv1d_46[0][0]                  
    __________________________________________________________________________________________________
    activation_24 (Activation)      (None, None, 32)     0           conv1d_47[0][0]                  
    __________________________________________________________________________________________________
    multiply_12 (Multiply)          (None, None, 32)     0           activation_23[0][0]              
                                                                     activation_24[0][0]              
    __________________________________________________________________________________________________
    conv1d_48 (Conv1D)              (None, None, 16)     528         multiply_12[0][0]                
    __________________________________________________________________________________________________
    add_12 (Add)                    (None, None, 16)     0           conv1d_45[0][0]                  
                                                                     conv1d_48[0][0]                  
    __________________________________________________________________________________________________
    conv1d_49 (Conv1D)              (None, None, 16)     272         add_12[0][0]                     
    __________________________________________________________________________________________________
    conv1d_50 (Conv1D)              (None, None, 32)     1056        conv1d_49[0][0]                  
    __________________________________________________________________________________________________
    conv1d_51 (Conv1D)              (None, None, 32)     1056        conv1d_49[0][0]                  
    __________________________________________________________________________________________________
    activation_25 (Activation)      (None, None, 32)     0           conv1d_50[0][0]                  
    __________________________________________________________________________________________________
    activation_26 (Activation)      (None, None, 32)     0           conv1d_51[0][0]                  
    __________________________________________________________________________________________________
    multiply_13 (Multiply)          (None, None, 32)     0           activation_25[0][0]              
                                                                     activation_26[0][0]              
    __________________________________________________________________________________________________
    conv1d_52 (Conv1D)              (None, None, 16)     528         multiply_13[0][0]                
    __________________________________________________________________________________________________
    add_13 (Add)                    (None, None, 16)     0           conv1d_49[0][0]                  
                                                                     conv1d_52[0][0]                  
    __________________________________________________________________________________________________
    conv1d_53 (Conv1D)              (None, None, 16)     272         add_13[0][0]                     
    __________________________________________________________________________________________________
    conv1d_54 (Conv1D)              (None, None, 32)     1056        conv1d_53[0][0]                  
    __________________________________________________________________________________________________
    conv1d_55 (Conv1D)              (None, None, 32)     1056        conv1d_53[0][0]                  
    __________________________________________________________________________________________________
    activation_27 (Activation)      (None, None, 32)     0           conv1d_54[0][0]                  
    __________________________________________________________________________________________________
    activation_28 (Activation)      (None, None, 32)     0           conv1d_55[0][0]                  
    __________________________________________________________________________________________________
    multiply_14 (Multiply)          (None, None, 32)     0           activation_27[0][0]              
                                                                     activation_28[0][0]              
    __________________________________________________________________________________________________
    conv1d_56 (Conv1D)              (None, None, 16)     528         multiply_14[0][0]                
    __________________________________________________________________________________________________
    add_14 (Add)                    (None, None, 16)     0           conv1d_53[0][0]                  
                                                                     conv1d_56[0][0]                  
    __________________________________________________________________________________________________
    conv1d_57 (Conv1D)              (None, None, 16)     272         add_14[0][0]                     
    __________________________________________________________________________________________________
    conv1d_58 (Conv1D)              (None, None, 32)     1056        conv1d_57[0][0]                  
    __________________________________________________________________________________________________
    conv1d_59 (Conv1D)              (None, None, 32)     1056        conv1d_57[0][0]                  
    __________________________________________________________________________________________________
    activation_29 (Activation)      (None, None, 32)     0           conv1d_58[0][0]                  
    __________________________________________________________________________________________________
    activation_30 (Activation)      (None, None, 32)     0           conv1d_59[0][0]                  
    __________________________________________________________________________________________________
    multiply_15 (Multiply)          (None, None, 32)     0           activation_29[0][0]              
                                                                     activation_30[0][0]              
    __________________________________________________________________________________________________
    conv1d_60 (Conv1D)              (None, None, 16)     528         multiply_15[0][0]                
    __________________________________________________________________________________________________
    add_15 (Add)                    (None, None, 16)     0           conv1d_57[0][0]                  
                                                                     conv1d_60[0][0]                  
    __________________________________________________________________________________________________
    conv1d_61 (Conv1D)              (None, None, 16)     272         add_15[0][0]                     
    __________________________________________________________________________________________________
    conv1d_62 (Conv1D)              (None, None, 32)     1056        conv1d_61[0][0]                  
    __________________________________________________________________________________________________
    conv1d_63 (Conv1D)              (None, None, 32)     1056        conv1d_61[0][0]                  
    __________________________________________________________________________________________________
    activation_31 (Activation)      (None, None, 32)     0           conv1d_62[0][0]                  
    __________________________________________________________________________________________________
    activation_32 (Activation)      (None, None, 32)     0           conv1d_63[0][0]                  
    __________________________________________________________________________________________________
    multiply_16 (Multiply)          (None, None, 32)     0           activation_31[0][0]              
                                                                     activation_32[0][0]              
    __________________________________________________________________________________________________
    conv1d_64 (Conv1D)              (None, None, 16)     528         multiply_16[0][0]                
    __________________________________________________________________________________________________
    add_17 (Add)                    (None, None, 16)     0           conv1d_4[0][0]                   
                                                                     conv1d_8[0][0]                   
                                                                     conv1d_12[0][0]                  
                                                                     conv1d_16[0][0]                  
                                                                     conv1d_20[0][0]                  
                                                                     conv1d_24[0][0]                  
                                                                     conv1d_28[0][0]                  
                                                                     conv1d_32[0][0]                  
                                                                     conv1d_36[0][0]                  
                                                                     conv1d_40[0][0]                  
                                                                     conv1d_44[0][0]                  
                                                                     conv1d_48[0][0]                  
                                                                     conv1d_52[0][0]                  
                                                                     conv1d_56[0][0]                  
                                                                     conv1d_60[0][0]                  
                                                                     conv1d_64[0][0]                  
    __________________________________________________________________________________________________
    activation_33 (Activation)      (None, None, 16)     0           add_17[0][0]                     
    __________________________________________________________________________________________________
    conv1d_65 (Conv1D)              (None, None, 128)    2176        activation_33[0][0]              
    __________________________________________________________________________________________________
    activation_34 (Activation)      (None, None, 128)    0           conv1d_65[0][0]                  
    __________________________________________________________________________________________________
    dropout_1 (Dropout)             (None, None, 128)    0           activation_34[0][0]              
    __________________________________________________________________________________________________
    conv1d_66 (Conv1D)              (None, None, 1)      129         dropout_1[0][0]                  
    __________________________________________________________________________________________________
    lambda_1 (Lambda)               (None, None, 1)      0           conv1d_66[0][0]                  
    ==================================================================================================
    Total params: 48,657
    Trainable params: 48,657
    Non-trainable params: 0
    __________________________________________________________________________________________________
    

With our training architecture defined, we're ready to train the model! This will take quite a while if you're not running fancy hardware (read GPU). We'll leverage the transformer utility functions we defined earlier, and train using mean absolute error loss.

Note that for this full-fledged model, we have more than twice as many total parameters to train as we did with the simpler WaveNet model, explaining the slower training time (along with using more training data). From the loss curve you'll see plotted below, it also seems likely that the model can continue to improve with more than 10 training epochs -- the more complex model probably needs additional time to reach its full potential. That said, from the results plots (see section 5) we can see that this full-fledged model is very capable of handling the 60-day forecast horizon and often can generate very expressive predictions. 

This is only a starting point, and I would encourage you to play around with this architecture to see if you can get even better results! You could try using more data, adjusting the hyperparameters, tuning the learning rate and number of epochs, etc.  


```python
first_n_samples = 120000
batch_size = 2**11
epochs = 10

# sample of series from train_enc_start to train_enc_end  
encoder_input_data = get_time_block_series(series_array, date_to_index, 
                                           train_enc_start, train_enc_end)[:first_n_samples]
encoder_input_data, encode_series_mean = transform_series_encode(encoder_input_data)

# sample of series from train_pred_start to train_pred_end 
decoder_target_data = get_time_block_series(series_array, date_to_index, 
                                            train_pred_start, train_pred_end)[:first_n_samples]
decoder_target_data = transform_series_decode(decoder_target_data, encode_series_mean)

# we append a lagged history of the target series to the input data, 
# so that we can train with teacher forcing
lagged_target_history = decoder_target_data[:,:-1,:1]
encoder_input_data = np.concatenate([encoder_input_data, lagged_target_history], axis=1)

model.compile(Adam(), loss='mean_absolute_error')
history = model.fit(encoder_input_data, decoder_target_data,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_split=0.2)  
```

    Train on 96000 samples, validate on 24000 samples
    Epoch 1/10
    96000/96000 [==============================] - 1443s 15ms/step - loss: 0.4242 - val_loss: 0.2930
    Epoch 2/10
    96000/96000 [==============================] - 1408s 15ms/step - loss: 0.3065 - val_loss: 0.2724
    Epoch 3/10
    96000/96000 [==============================] - 1392s 14ms/step - loss: 0.2884 - val_loss: 0.2588
    Epoch 4/10
    96000/96000 [==============================] - 1399s 15ms/step - loss: 0.2800 - val_loss: 0.2535
    Epoch 5/10
    96000/96000 [==============================] - 1379s 14ms/step - loss: 0.2750 - val_loss: 0.2505
    Epoch 6/10
    96000/96000 [==============================] - 1401s 15ms/step - loss: 0.2719 - val_loss: 0.2483
    Epoch 7/10
    96000/96000 [==============================] - 1431s 15ms/step - loss: 0.2699 - val_loss: 0.2466
    Epoch 8/10
    96000/96000 [==============================] - 1428s 15ms/step - loss: 0.2684 - val_loss: 0.2457
    Epoch 9/10
    96000/96000 [==============================] - 1431s 15ms/step - loss: 0.2673 - val_loss: 0.2440
    Epoch 10/10
    96000/96000 [==============================] - 1434s 15ms/step - loss: 0.2662 - val_loss: 0.2431
    

It's typically a good idea to look at the convergence curve of train/validation loss.


```python
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])

plt.xlabel('Epoch')
plt.ylabel('Mean Absolute Error Loss')
plt.title('Loss Over Time')
plt.legend(['Train','Valid'])
```




    <matplotlib.legend.Legend at 0x1a2fc58d30>




![png](/images/output_19_1d.png)


## 4. Building the Model - Inference Loop

Like in the previous notebook, we'll generate predictions by running our model from section 3 in a loop, using each iteration to extract the prediction for the time step one beyond our current history then append it to our history sequence. With 60 iterations, this lets us generate predictions for the full interval we've chosen. 

Recall that we designed our model to output predictions for 60 time steps at once in order to use teacher forcing for training. So if we start from a history sequence and want to predict the first future time step, we can run the model on the history sequence and take the last time step of the output, which corresponds to one time step beyond the history sequence. 


```python
def predict_sequence(input_sequence):

    history_sequence = input_sequence.copy()
    pred_sequence = np.zeros((1,pred_steps,1)) # initialize output (pred_steps time steps)  
    
    for i in range(pred_steps):
        
        # record next time step prediction (last time step of model output) 
        last_step_pred = model.predict(history_sequence)[0,-1,0]
        pred_sequence[0,i,0] = last_step_pred
        
        # add the next time step prediction to the history sequence
        history_sequence = np.concatenate([history_sequence, 
                                           last_step_pred.reshape(-1,1,1)], axis=1)

    return pred_sequence
```

## 5. Generating and Plotting Predictions 

Now we have everything we need to generate predictions for encoder (history) /target series pairs that we didn't train on (note again we're using "encoder"/"decoder" terminology to stay consistent with notebook 1 -- here it's more like history/target). We'll pull out our set of validation encoder/target series (recall that these are shifted forward in time). Then using a plotting utility function, we can look at the tail end of the encoder series, the true target series, and the predicted target series. This gives us a feel for how our predictions are doing.  


```python
encoder_input_data = get_time_block_series(series_array, date_to_index, val_enc_start, val_enc_end)
encoder_input_data, encode_series_mean = transform_series_encode(encoder_input_data)

decoder_target_data = get_time_block_series(series_array, date_to_index, val_pred_start, val_pred_end)
decoder_target_data = transform_series_decode(decoder_target_data, encode_series_mean)
```


```python
def predict_and_plot(encoder_input_data, decoder_target_data, sample_ind, enc_tail_len=50):

    encode_series = encoder_input_data[sample_ind:sample_ind+1,:,:] 
    pred_series = predict_sequence(encode_series)
    
    encode_series = encode_series.reshape(-1,1)
    pred_series = pred_series.reshape(-1,1)   
    target_series = decoder_target_data[sample_ind,:,:1].reshape(-1,1) 
    
    encode_series_tail = np.concatenate([encode_series[-enc_tail_len:],target_series[:1]])
    x_encode = encode_series_tail.shape[0]
    
    plt.figure(figsize=(10,6))   
    
    plt.plot(range(1,x_encode+1),encode_series_tail)
    plt.plot(range(x_encode,x_encode+pred_steps),target_series,color='orange')
    plt.plot(range(x_encode,x_encode+pred_steps),pred_series,color='teal',linestyle='--')
    
    plt.title('Encoder Series Tail of Length %d, Target Series, and Predictions' % enc_tail_len)
    plt.legend(['Encoding Series','Target Series','Predictions'])
```

Generating some plots as below, we can see that our longer time horizon predictions (60 days) are often strong and expressive. Our full-fledged model is able to effectively capture weekly seasonality patterns and long term trends, and does a very nice job adapting to the varying levels of fluctuation in each series.   

Still, we can do even better! We'd benefit from increasing the sample size for training and fine-tuning our hyperparameters, but also by giving the model access to additional relevant information. So far we've only fed the model raw time series data, but it can likely benefit from the inclusion of **exogenous variables** such as the day of the week and the language of the webpage corresponding to each series. To see how these exogenous variables can be incorporated directly into the model **check out the next notebook in this series**. 

If you're interested in digging even deeper into state of the art WaveNet style architectures, I also highly recommend checking out [Sean Vasquez's model](https://github.com/sjvasquez/web-traffic-forecasting) that was designed for this data set. He implements a customized seq2seq WaveNet architecture in tensorflow.    


```python
predict_and_plot(encoder_input_data, decoder_target_data, 
                 sample_ind=16534, enc_tail_len=100)
```


![png](/images/output_26_0d.png)



```python
predict_and_plot(encoder_input_data, decoder_target_data, 
                 sample_ind=16555, enc_tail_len=100)
```


![png](/images/output_27_0d.png)



```python
predict_and_plot(encoder_input_data, decoder_target_data, 
                 sample_ind=4000, enc_tail_len=100)
```


![png](/images/output_28_0d.png)



```python
predict_and_plot(encoder_input_data, decoder_target_data, 
                 sample_ind=68000, enc_tail_len=100)
```


![png](/images/output_29_0d.png)



```python
predict_and_plot(encoder_input_data, decoder_target_data, 
                 sample_ind=6007, enc_tail_len=100)
```


![png](/images/output_30_0d.png)



```python
predict_and_plot(encoder_input_data, decoder_target_data, 
                 sample_ind=70450, enc_tail_len=100)
```


![png](/images/output_31_0d.png)



```python
predict_and_plot(encoder_input_data, decoder_target_data, 
                 sample_ind=16551, enc_tail_len=100)
```


![png](/images/output_32_0d.png)

