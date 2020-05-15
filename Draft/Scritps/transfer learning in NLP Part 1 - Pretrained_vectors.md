title: Transfer learning in NLP Part I : Pre-trained embeddings
Date: 2019-07-07 13:01
Category: Machine Learning, July 2019, Transfer learning
Tags: NLP, July 2019, Transfer learning
Slug: Machine Learning, July 2019, Transfer learning
Author: Mohcine Madkour
Email:mohcine.madkour@gmail.com

We are going to build some PyTorch models that are commonly used for text classification.  We also need to build out some infrastructure to run these models.

Once we have the models and the boilerplate stuff out of the way, we can see the impact of pre-trained embeddings for classification tasks. Pre-training methods like word2vec are context-limited language models whose goal is to predict a word given a fixed context, or a fixed context given a word. Pre-trained embeddings are particularly useful for smaller datasets.

Most of this code is inspired or derived from [Baseline](https://github.com/dpressel/baseline/) (Pressel et al, 2018), an open source project for building and evaluating NLP models across a variety of NLP tasks.  For this tutorial, we will only concern ourselves with Text Classification using a few useful models.

## Word Embeddings in NLP

We start our models with what are called "one-hot" vectors.  This is notionally a sparse vector with length `|V|` where V is our vocabulary, and where only the word representated at this temporal location is a 1.  The rest are zeros.

![onehot](https://www.tensorflow.org/images/feature_columns/categorical_column_with_vocabulary.jpg)

These vectors are not truly represented as a vector, but as an array of indices (in PyTorch, they are `torch.LongTensor`s), one for each word's index in the vocab.  This representation is not particularly helpful in DNNs since we want continuous representations for each word.

![indices](https://www.tensorflow.org/images/feature_columns/categorical_column_with_identity.jpg)

The general idea of an embedding is that we want to project from a large one-hot vector to a compact, distributed representation with smaller dimensionality.  We can look at this as a matrix multiply between a one-hot vector `|V|` and a weight matrix to a lower dimension of size `|D|`.  Since only a single vector value in the one-hot vector is on at a time, this matrix multiply is simplified to an address lookup in that matrix.

![LUT](https://cdn-images-1.medium.com/max/800/1*fZj1Hk1mhS5pIMv3ZrpLYw.png)


In PyTorch, this is called an `nn.Embedding`.  In fact, in Torch7, this was called a `nn.LookupTable` which may have actually been a better name, but which seems to have fallen out of favor in DNN toolkits.  In this tutorial we are going to refer to multiple types of embeddings, and in this case, we are referring to word vectors, which are typically lookup table embeddings.

Embeddings make up lowest layer of a typical DNN for text and we will feed their output to some pooling mechanism yielding a fixed length representation, followed by some number of fully connected layers.

### Pre-training with Word2Vec

There has been a large amount of research that has gone into building distributed representations for words through pre-training. Some widely used algorithms in NLP include Word2Vec, GloVe and fastText.  For instance, word2vec is actually 2 different algorithms with 2 different objectives.  They can be thought of a fixed context window non-causal LMs, but they are shallow models and extremely fast to train

![word2vec](https://deeplearning4j.org/img/word2vec_diagrams.png)


* **CBOW objective** given all words in a fixed context window except the middle word, predict the middle word
* **SkipGram objective**: given a word in a fixed context window, predict all other words in that window

Once we have trained these models, the learned distributed representation matrix can be plugged right in as our embedding weights and this often improves the model significantly.

Before we begin, we will download some data that can be used for our experiments


```python
!wget https://www.dropbox.com/s/7jyi4pi894bh2qh/sst2.tar.gz?dl=1
!tar -xzf 'sst2.tar.gz?dl=1'

!wget https://www.dropbox.com/s/08km2ean8bkt7p3/trec.tar.gz?dl=1
!tar -xzf 'trec.tar.gz?dl=1'

!wget https://www.dropbox.com/s/699kgut7hdb5tg9/GoogleNews-vectors-negative300.bin.gz?dl=1
!mv 'GoogleNews-vectors-negative300.bin.gz?dl=1' GoogleNews-vectors-negative300.bin.gz
!gunzip GoogleNews-vectors-negative300.bin.gz
```

    --2019-07-08 00:32:45--  https://www.dropbox.com/s/7jyi4pi894bh2qh/sst2.tar.gz?dl=1
    Resolving www.dropbox.com (www.dropbox.com)... 2620:100:6018:1::a27d:301, 162.125.3.1
    Connecting to www.dropbox.com (www.dropbox.com)|2620:100:6018:1::a27d:301|:443... connected.
    HTTP request sent, awaiting response... 301 Moved Permanently
    Location: /s/dl/7jyi4pi894bh2qh/sst2.tar.gz [following]
    --2019-07-08 00:32:45--  https://www.dropbox.com/s/dl/7jyi4pi894bh2qh/sst2.tar.gz
    Reusing existing connection to [www.dropbox.com]:443.
    HTTP request sent, awaiting response... 302 Found
    Location: https://uc3565d8839a10c5d53bd45de20c.dl.dropboxusercontent.com/cd/0/get/AkRLWd3A_OI7KBUn-VSMqeMgKz-UOPUTcEizbIpo0QNYHdzxKIhKRXKRLpO-YhOpuodmoJ-CvdeBgkzejDwGRE8E1NXaoVBNTDdXo9QK8pvgBw/file?dl=1# [following]
    --2019-07-08 00:32:45--  https://uc3565d8839a10c5d53bd45de20c.dl.dropboxusercontent.com/cd/0/get/AkRLWd3A_OI7KBUn-VSMqeMgKz-UOPUTcEizbIpo0QNYHdzxKIhKRXKRLpO-YhOpuodmoJ-CvdeBgkzejDwGRE8E1NXaoVBNTDdXo9QK8pvgBw/file?dl=1
    Resolving uc3565d8839a10c5d53bd45de20c.dl.dropboxusercontent.com (uc3565d8839a10c5d53bd45de20c.dl.dropboxusercontent.com)... 2620:100:6018:6::a27d:306, 162.125.3.6
    Connecting to uc3565d8839a10c5d53bd45de20c.dl.dropboxusercontent.com (uc3565d8839a10c5d53bd45de20c.dl.dropboxusercontent.com)|2620:100:6018:6::a27d:306|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 1759259 (1.7M) [application/binary]
    Saving to: ‘sst2.tar.gz?dl=1’
    
    sst2.tar.gz?dl=1    100%[===================>]   1.68M   935KB/s    in 1.8s    
    
    2019-07-08 00:32:48 (935 KB/s) - ‘sst2.tar.gz?dl=1’ saved [1759259/1759259]
    
    --2019-07-08 00:32:48--  https://www.dropbox.com/s/08km2ean8bkt7p3/trec.tar.gz?dl=1
    Resolving www.dropbox.com (www.dropbox.com)... 2620:100:6018:1::a27d:301, 162.125.3.1
    Connecting to www.dropbox.com (www.dropbox.com)|2620:100:6018:1::a27d:301|:443... connected.
    HTTP request sent, awaiting response... 301 Moved Permanently
    Location: /s/dl/08km2ean8bkt7p3/trec.tar.gz [following]
    --2019-07-08 00:32:49--  https://www.dropbox.com/s/dl/08km2ean8bkt7p3/trec.tar.gz
    Reusing existing connection to [www.dropbox.com]:443.
    HTTP request sent, awaiting response... 302 Found
    Location: https://uc249811a6e442bf7f679c39dfd1.dl.dropboxusercontent.com/cd/0/get/AkQGUYOHicMExjYIVGDFLGzAAyWTvTdF_5mVfLAakOE9VuAsn4ssIZVNEt087E2oL-OZHEquUp8ywHeCeAdyvinMSnPa6b4OIdJXfSFfS4E6lg/file?dl=1# [following]
    --2019-07-08 00:32:49--  https://uc249811a6e442bf7f679c39dfd1.dl.dropboxusercontent.com/cd/0/get/AkQGUYOHicMExjYIVGDFLGzAAyWTvTdF_5mVfLAakOE9VuAsn4ssIZVNEt087E2oL-OZHEquUp8ywHeCeAdyvinMSnPa6b4OIdJXfSFfS4E6lg/file?dl=1
    Resolving uc249811a6e442bf7f679c39dfd1.dl.dropboxusercontent.com (uc249811a6e442bf7f679c39dfd1.dl.dropboxusercontent.com)... 2620:100:6018:6::a27d:306, 162.125.3.6
    Connecting to uc249811a6e442bf7f679c39dfd1.dl.dropboxusercontent.com (uc249811a6e442bf7f679c39dfd1.dl.dropboxusercontent.com)|2620:100:6018:6::a27d:306|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 117253 (115K) [application/binary]
    Saving to: ‘trec.tar.gz?dl=1’
    
    trec.tar.gz?dl=1    100%[===================>] 114.50K  --.-KB/s    in 0.09s   
    
    2019-07-08 00:32:49 (1.21 MB/s) - ‘trec.tar.gz?dl=1’ saved [117253/117253]
    
    --2019-07-08 00:32:50--  https://www.dropbox.com/s/699kgut7hdb5tg9/GoogleNews-vectors-negative300.bin.gz?dl=1
    Resolving www.dropbox.com (www.dropbox.com)... 2620:100:6018:1::a27d:301, 162.125.3.1
    Connecting to www.dropbox.com (www.dropbox.com)|2620:100:6018:1::a27d:301|:443... connected.
    HTTP request sent, awaiting response... 301 Moved Permanently
    Location: /s/dl/699kgut7hdb5tg9/GoogleNews-vectors-negative300.bin.gz [following]
    --2019-07-08 00:32:50--  https://www.dropbox.com/s/dl/699kgut7hdb5tg9/GoogleNews-vectors-negative300.bin.gz
    Reusing existing connection to [www.dropbox.com]:443.
    HTTP request sent, awaiting response... 302 Found
    Location: https://uc63bc9bed52962b811b5cd41c55.dl.dropboxusercontent.com/cd/0/get/AkS8j2aeKHnhjxuH51sfVvN05jWUSJ6vP1i3_w4HSpcVFrf5oepeKw4n0BDXSQDZ_2yw0ksJFfFhmwpCT-EJF7kCiqMW3-8eAClkpqgkWkXXVw/file?dl=1# [following]
    --2019-07-08 00:32:50--  https://uc63bc9bed52962b811b5cd41c55.dl.dropboxusercontent.com/cd/0/get/AkS8j2aeKHnhjxuH51sfVvN05jWUSJ6vP1i3_w4HSpcVFrf5oepeKw4n0BDXSQDZ_2yw0ksJFfFhmwpCT-EJF7kCiqMW3-8eAClkpqgkWkXXVw/file?dl=1
    Resolving uc63bc9bed52962b811b5cd41c55.dl.dropboxusercontent.com (uc63bc9bed52962b811b5cd41c55.dl.dropboxusercontent.com)... 2620:100:6018:6::a27d:306, 162.125.3.6
    Connecting to uc63bc9bed52962b811b5cd41c55.dl.dropboxusercontent.com (uc63bc9bed52962b811b5cd41c55.dl.dropboxusercontent.com)|2620:100:6018:6::a27d:306|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 1743563840 (1.6G) [application/binary]
    Saving to: ‘GoogleNews-vectors-negative300.bin.gz?dl=1’
    
    GoogleNews-vectors-  12%[=>                  ] 209.10M   708KB/s    eta 24m 28s

## First, lets do some fun stuff

We will start by building out some models that we will reuse later in the tutorial.  First, we will build a convolutional neural network (CNN) that can classify text. Basically CNNs learn a kernel that can be used to filter images or text.  An example of 2D filtering*:

![2D filtering](https://cdn-images-1.medium.com/max/1600/0*9J3MK1gd2zrFDzDN.gif)

In the case of text filtering, we have a one-dimensional filter operation like this*:

![1D filtering](http://cs231n.github.io/assets/cnn/stride.jpeg)

This type of model has been used often in text, including by [Collobert et al 2011](https://ronan.collobert.com/pub/matos/2011_nlp_jmlr.pdf), but we will implement a multiple parallel filter variation of this introduced by [Kim 2014](https://www.aclweb.org/anthology/D14-1181).

### Convolutional Neural Network for Text Classification

We are using PyTorch, so every layer we have is going to inherit `nn.Module`.

#### Convolutions (actually cross correlations)

The first characteristic of this model is that we will have multiple convolutional filter lengths, and some number of filters associated with each length.  For each filter of length `K` convolved with a signal of length `T`, the output signal will be `T - K + 1`.  To handle the ends of the signal where the filter is hanging off (e.g. centered at 0), we will add some zero-padding.  So if we have a filter of length `K=3`, we want to zero-pad the temporal signal by a single pad value on both ends of the signal.

We are going to support multiple parallel filters, so we can add a `torch.nn.Conv1d` for each filter length, followed by a `torch.nn.ReLU` activation layer.  Since we have more than one of these, we will create a `nn.ModuleList` to track them.  When we call `forward()`, the data will be oriented as $$B \times C \times T$$ where `B` is the batch size,  `C` is the number of hidden units and `T` is the temporal length of the vector.

#### Pooling

Both of the papers mentioned above do max-over-time pooling over the features.  For each feature map in the vector, we simply select the maximum value for that feature map and concatenate all of these together.  Our $$B \times C \times T$$ vector is then going to be reduced along the time dimension to $$B \times C$$

*Images courtesy of http://cs231n.github.io/convolutional-networks/


```python
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple
import os
import io
import re
import codecs
from collections import Counter
from torch.utils.data import DataLoader, TensorDataset
```


```python
class ParallelConv(nn.Module):

    def __init__(self, input_dims, filters, dropout=0.5):
        super().__init__()
        convs = []        
        self.output_dims = sum([t[1] for t in filters])
        for (filter_length, output_dims) in filters:
            pad = filter_length//2
            conv = nn.Sequential(
                nn.Conv1d(input_dims, output_dims, filter_length, padding=pad),
                nn.ReLU()
            )
            convs.append(conv)
        # Add the module so its managed correctly
        self.convs = nn.ModuleList(convs)
        self.conv_drop = nn.Dropout(dropout)


    def forward(self, input_bct):
        mots = []
        for conv in self.convs:
            # In Conv1d, data BxCxT, max over time
            conv_out = conv(input_bct)
            mot, _ = conv_out.max(2)
            mots.append(mot)
        mots = torch.cat(mots, 1)
        return self.conv_drop(mots)

class ConvClassifier(nn.Module):

    def __init__(self, embeddings, num_classes, embed_dims,
                 filters=[(2, 100), (3, 100), (4, 100)],
                 dropout=0.5, hidden_units=[]):
        super().__init__()
        self.embeddings = embeddings
        self.dropout = nn.Dropout(dropout)
        self.convs = ParallelConv(embed_dims, filters, dropout)
        
        input_units = self.convs.output_dims
        output_units = self.convs.output_dims
        sequence = []
        for h in hidden_units:
            sequence.append(self.dropout(nn.Linear(input_units, h)))
            input_units = h
            output_units = h
            
        sequence.append(nn.Linear(output_units, num_classes))
        self.outputs = nn.Sequential(*sequence)

    def forward(self, inputs):
        one_hots, lengths = inputs
        embed = self.dropout(self.embeddings(one_hots))
        embed = embed.transpose(1, 2).contiguous()
        hidden = self.convs(embed)
        linear = self.outputs(hidden)
        return F.log_softmax(linear, dim=-1)

```

### LSTM Model

Our second model that we will explore uses Long Short-Term Memory (LSTM) units, which are a form of Recurrent Neural Networks.  These models tend to perform extremely well on NLP tasks.  Text classification is a simple case, where we give our inputs and take the final LSTM output as a form of pooling.  That looks like the **Many-to-One** image in this taxonomy from [Andrej Karpathy's 2015 blog post on using RNNs for character-level language modeling](https://karpathy.github.io/2015/05/21/rnn-effectiveness/)

![Many-to-one LSTM](https://karpathy.github.io/assets/rnn/diags.jpeg)


```python
class LSTMClassifier(nn.Module):

    def __init__(self, embeddings, num_classes, embed_dims, rnn_units, rnn_layers=1, dropout=0.5, hidden_units=[]):
        super().__init__()
        self.embeddings = embeddings
        self.dropout = nn.Dropout(dropout)
        self.rnn = torch.nn.LSTM(embed_dims,
                                 rnn_units,
                                 rnn_layers,
                                 dropout=dropout,
                                 bidirectional=False,
                                 batch_first=False)
        nn.init.orthogonal_(self.rnn.weight_hh_l0)
        nn.init.orthogonal_(self.rnn.weight_ih_l0)
        sequence = []
        input_units = rnn_units
        output_units = rnn_units
        for h in hidden_units:
            sequence.append(nn.Linear(input_units, h))
            input_units = h
            output_units = h
            
        sequence.append(nn.Linear(output_units, num_classes))
        self.outputs = nn.Sequential(*sequence)
        
        
    def forward(self, inputs):
        one_hots, lengths = inputs
        embed = self.dropout(self.embeddings(one_hots))
        embed = embed.transpose(0, 1)
        packed = torch.nn.utils.rnn.pack_padded_sequence(embed, lengths.tolist())
        _, hidden = self.rnn(packed)
        hidden = hidden[0].view(hidden[0].shape[1:])
        linear = self.outputs(hidden)
        return F.log_softmax(linear, dim=-1)

```

## Training our model

To set our model up for training (and evaluation), we need a loss function, some metrics, and an optimizer, along with some training data.

### Defining Metrics

For classification problems, most things we would like to know can be defined in terms of a confusion matrix.

![Confusion Matrix](https://scikit-learn.org/stable/_images/sphx_glr_plot_confusion_matrix_001.png)

The class below implements a confusion matrix and provides metrics associated using it.  This implementation is taken from verbatim from Baseline (https://github.com/dpressel/baseline/blob/master/python/baseline/confusion.py)


```python

class ConfusionMatrix:
    """Confusion matrix with metrics

    This class accumulates classification output, and tracks it in a confusion matrix.
    Metrics are available that use the confusion matrix
    """
    def __init__(self, labels):
        """Constructor with input labels

        :param labels: Either a dictionary (`k=int,v=str`) or an array of labels
        """
        if type(labels) is dict:
            self.labels = []
            for i in range(len(labels)):
                self.labels.append(labels[i])
        else:
            self.labels = labels
        nc = len(self.labels)
        self._cm = np.zeros((nc, nc), dtype=np.int)

    def add(self, truth, guess):
        """Add a single value to the confusion matrix based off `truth` and `guess`

        :param truth: The real `y` value (or ground truth label)
        :param guess: The guess for `y` value (or assertion)
        """

        self._cm[truth, guess] += 1

    def __str__(self):
        values = []
        width = max(8, max(len(x) for x in self.labels) + 1)
        for i, label in enumerate([''] + self.labels):
            values += ["{:>{width}}".format(label, width=width+1)]
        values += ['\n']
        for i, label in enumerate(self.labels):
            values += ["{:>{width}}".format(label, width=width+1)]
            for j in range(len(self.labels)):
                values += ["{:{width}d}".format(self._cm[i, j], width=width + 1)]
            values += ['\n']
        values += ['\n']
        return ''.join(values)

    def save(self, outfile):
        ordered_fieldnames = OrderedDict([("labels", None)] + [(l, None) for l in self.labels])
        with open(outfile, 'w') as f:
            dw = csv.DictWriter(f, delimiter=',', fieldnames=ordered_fieldnames)
            dw.writeheader()
            for index, row in enumerate(self._cm):
                row_dict = {l: row[i] for i, l in enumerate(self.labels)}
                row_dict.update({"labels": self.labels[index]})
                dw.writerow(row_dict)

    def reset(self):
        """Reset the matrix
        """
        self._cm *= 0

    def get_correct(self):
        """Get the diagonals of the confusion matrix

        :return: (``int``) Number of correct classifications
        """
        return self._cm.diagonal().sum()

    def get_total(self):
        """Get total classifications

        :return: (``int``) total classifications
        """
        return self._cm.sum()

    def get_acc(self):
        """Get the accuracy

        :return: (``float``) accuracy
        """
        return float(self.get_correct())/self.get_total()

    def get_recall(self):
        """Get the recall

        :return: (``float``) recall
        """
        total = np.sum(self._cm, axis=1)
        total = (total == 0) + total
        return np.diag(self._cm) / total.astype(float)

    def get_support(self):
        return np.sum(self._cm, axis=1)

    def get_precision(self):
        """Get the precision
        :return: (``float``) precision
        """

        total = np.sum(self._cm, axis=0)
        total = (total == 0) + total
        return np.diag(self._cm) / total.astype(float)

    def get_mean_precision(self):
        """Get the mean precision across labels

        :return: (``float``) mean precision
        """
        return np.mean(self.get_precision())

    def get_weighted_precision(self):
        return np.sum(self.get_precision() * self.get_support())/float(self.get_total())

    def get_mean_recall(self):
        """Get the mean recall across labels

        :return: (``float``) mean recall
        """
        return np.mean(self.get_recall())

    def get_weighted_recall(self):
        return np.sum(self.get_recall() * self.get_support())/float(self.get_total())

    def get_weighted_f(self, beta=1):
        return np.sum(self.get_class_f(beta) * self.get_support())/float(self.get_total())

    def get_macro_f(self, beta=1):
        """Get the macro F_b, with adjustable beta (defaulting to F1)

        :param beta: (``float``) defaults to 1 (F1)
        :return: (``float``) macro F_b
        """
        if beta < 0:
            raise Exception('Beta must be greater than 0')
        return np.mean(self.get_class_f(beta))

    def get_class_f(self, beta=1):
        p = self.get_precision()
        r = self.get_recall()

        b = beta*beta
        d = (b * p + r)
        d = (d == 0) + d

        return (b + 1) * p * r / d

    def get_f(self, beta=1):
        """Get 2 class F_b, with adjustable beta (defaulting to F1)

        :param beta: (``float``) defaults to 1 (F1)
        :return: (``float``) 2-class F_b
        """
        p = self.get_precision()[1]
        r = self.get_recall()[1]
        if beta < 0:
            raise Exception('Beta must be greater than 0')
        d = (beta*beta * p + r)
        if d == 0:
            return 0
        return (beta*beta + 1) * p * r / d

    def get_all_metrics(self):
        """Make a map of metrics suitable for reporting, keyed by metric name

        :return: (``dict``) Map of metrics keyed by metric names
        """
        metrics = {'acc': self.get_acc()}
        # If 2 class, assume second class is positive AKA 1
        if len(self.labels) == 2:
            metrics['precision'] = self.get_precision()[1]
            metrics['recall'] = self.get_recall()[1]
            metrics['f1'] = self.get_f(1)
        else:
            metrics['mean_precision'] = self.get_mean_precision()
            metrics['mean_recall'] = self.get_mean_recall()
            metrics['macro_f1'] = self.get_macro_f(1)
            metrics['weighted_precision'] = self.get_weighted_precision()
            metrics['weighted_recall'] = self.get_weighted_recall()
            metrics['weighted_f1'] = self.get_weighted_f(1)
        return metrics

    def add_batch(self, truth, guess):
        """Add a batch of data to the confusion matrix

        :param truth: The truth tensor
        :param guess: The guess tensor
        :return:
        """
        for truth_i, guess_i in zip(truth, guess):
            self.add(truth_i, guess_i)

```

Our `Trainer` is simple, but it gets the job done.  We will use PyTorch's `DataLoader` to feed our batches to the trainer.  The `run()` method cycles a single epoch. 
For every batch, we will do a stochastic gradient minibatch update, and we return the loss and the predictions and ground truth back to the `run()` method for tabulation


```python

class Trainer:
    def __init__(self, optimizer: torch.optim.Optimizer):
        self.optimizer = optimizer

    def run(self, model, labels, train, loss, batch_size): 
        model.train()       
        train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)

        cm = ConfusionMatrix(labels)

        for batch in train_loader:
            loss_value, y_pred, y_actual = self.update(model, loss, batch)
            _, best = y_pred.max(1)
            yt = y_actual.cpu().int().numpy()
            yp = best.cpu().int().numpy()
            cm.add_batch(yt, yp)

        print(cm.get_all_metrics())
        return cm
    
    def update(self, model, loss, batch):
        self.optimizer.zero_grad()
        x, lengths, y = batch
        lengths, perm_idx = lengths.sort(0, descending=True)
        x_sorted = x[perm_idx]
        y_sorted = y[perm_idx]
        y_sorted = y_sorted.to('cuda:0')
        inputs = (x_sorted.to('cuda:0'), lengths)
        y_pred = model(inputs)
        loss_value = loss(y_pred, y_sorted)
        loss_value.backward()
        self.optimizer.step()
        return loss_value.item(), y_pred, y_sorted
```

After training a epoch, we would like to test the validation performance.  Our evaluator class is similar to our `Trainer`, but it doesnt update our model -- it just gives us a way to evaluate the model on data


```python

class Evaluator:
    def __init__(self):
        pass

    def run(self, model, labels, dataset, batch_size=1):
        model.eval()
        valid_loader = DataLoader(dataset, batch_size=batch_size)
        cm = ConfusionMatrix(labels)
        for batch in valid_loader:
            y_pred, y_actual = self.inference(model, batch)
            _, best = y_pred.max(1)
            yt = y_actual.cpu().int().numpy()
            yp = best.cpu().int().numpy()
            cm.add_batch(yt, yp)
        return cm

    def inference(self, model, batch):
        with torch.no_grad():
            x, lengths, y = batch
            lengths, perm_idx = lengths.sort(0, descending=True)
            x_sorted = x[perm_idx]
            y_sorted = y[perm_idx]
            y_sorted = y_sorted.to('cuda:0')
            inputs = (x_sorted.to('cuda:0'), lengths)
            y_pred = model(inputs)
            return y_pred, y_sorted

```

We can encapsulate training multiple epochs and testing in a single function.   The best model is defined in terms of some metric -- here accuracy, and we only save the checkpoints when we improve on the model.  This is called early stopping, and is particularly helpful on smaller datasets


```python
def fit(model, labels, optimizer, loss, epochs, batch_size, train, valid, test):

    trainer = Trainer(optimizer)
    evaluator = Evaluator()
    best_acc = 0.0
    
    for epoch in range(epochs):
        print('EPOCH {}'.format(epoch + 1))
        print('=================================')
        print('Training Results')
        cm = trainer.run(model, labels, train, loss, batch_size)
        print('Validation Results')
        cm = evaluator.run(model, labels, valid)
        print(cm.get_all_metrics())
        if cm.get_acc() > best_acc:
            print('New best model {:.2f}'.format(cm.get_acc()))
            best_acc = cm.get_acc()
            torch.save(model.state_dict(), './checkpoint.pth')
    if test:
        model.load_state_dict(torch.load('./checkpoint.pth'))
        cm = evaluator.run(model, labels, test)
        print('Final result')
        print(cm.get_all_metrics())
    return cm.get_acc()
```

### A Reader for our Data

We need a reader to load our data from files and put it into a `Dataset`.

The reader needs to perform a few steps

* **read in sentences and labels**: it should convert the sentences into tokens and record a vocabulary of the labels
* **vectorize tokens**: it should convert tokens into tensors that comprise rows in our `TensorDataset`
* **tabulate the vocabulary**: if no vectorizer is provided, we need to build a vocab of attested words.  If a vectorizer is provided upfront, we dont need this step


```python

def whitespace_tokenizer(words: str) -> List[str]:
    return words.split() 

def sst2_tokenizer(words: str) -> List[str]:
    REPLACE = { "'s": " 's ",
                "'ve": " 've ",
                "n't": " n't ",
                "'re": " 're ",
                "'d": " 'd ",
                "'ll": " 'll ",
                ",": " , ",
                "!": " ! ",
                }
    words = words.lower()
    words = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", words)
    for k, v in REPLACE.items():
            words = words.replace(k, v)
    return [w.strip() for w in words.split()]


class Reader:

    def __init__(self, files, lowercase=True, min_freq=0,
                 tokenizer=sst2_tokenizer, vectorizer=None):
        self.lowercase = lowercase
        self.tokenizer = tokenizer
        build_vocab = vectorizer is None
        self.vectorizer = vectorizer if vectorizer else self._vectorizer
        x = Counter()
        y = Counter()
        for file_name in files:
            if file_name is None:
                continue
            with codecs.open(file_name, encoding='utf-8', mode='r') as f:
                for line in f:
                    words = line.split()
                    y.update(words[0])

                    if build_vocab:
                        words = self.tokenizer(' '.join(words[1:]))
                        words = words if not self.lowercase else [w.lower() for w in words]
                        x.update(words)
        self.labels = list(y.keys())

        if build_vocab:
            x = dict(filter(lambda cnt: cnt[1] >= min_freq, x.items()))
            alpha = list(x.keys())
            alpha.sort()
            self.vocab = {w: i+1 for i, w in enumerate(alpha)}
            self.vocab['[PAD]'] = 0

        self.labels.sort()

    def _vectorizer(self, words: List[str]) -> List[int]:
        return [self.vocab.get(w, 0) for w in words]

    def load(self, filename: str) -> TensorDataset:
        label2index = {l: i for i, l in enumerate(self.labels)}
        xs = []
        lengths = []
        ys = []
        with codecs.open(filename, encoding='utf-8', mode='r') as f:
            for line in f:
                words = line.split()
                ys.append(label2index[words[0]])
                words = self.tokenizer(' '.join(words[1:]))
                words = words if not self.lowercase else [w.lower() for w in words]
                vec = self.vectorizer(words)
                lengths.append(len(vec))
                xs.append(torch.tensor(vec, dtype=torch.long))
        x_tensor = torch.nn.utils.rnn.pad_sequence(xs, batch_first=True)
        lengths_tensor = torch.tensor(lengths, dtype=torch.long)
        y_tensor = torch.tensor(ys, dtype=torch.long)
        return TensorDataset(x_tensor, lengths_tensor, y_tensor)

```

### Pre-trained Embeddings

We would like to investigate how pre-training embeddings helps our models improve.  To do this, we need a mechanism to load in pre-trained embeddings and convert them into PyTorch's `nn.Embedding` object.  Specifically, we wish to support `word2vec`, `GloVe` and `fastText` embeddings. Rest-assured, these are simple file formats, and you do not need any 3rd party dependencies to read them in!  We will do it by hand.

For binary files, the first line contains 2 numbers delimited by a space.  The first number is the vocab size and the second is the embedding dimension.  We then read each line, splitting it by a space and reading the first portion as the vocabulary (token) and the second portion as a binary vector.

For text files, the first line may contain 2 numbers as in the binary file, but for `GloVe` files, this is omitted.  We can check if the first line contains the dimensions, and if it doesnt, we can just read in the first vector to figure out its dimension (again its space delimited, but the vector is also space delimited, so we split along the first space to find the token).

Notice that in this code, we have already created an alphabet that we will pass in for each key, so if that word is present in the embedding file, we will use its value, otherwise, we will initialize the vector randomly.


```python
def init_embeddings(vocab_size, embed_dim, unif):
    return np.random.uniform(-unif, unif, (vocab_size, embed_dim))
    

class EmbeddingsReader:

    @staticmethod
    def from_text(filename, vocab, unif=0.25):
        
        with io.open(filename, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.rstrip("\n ")
                values = line.split(" ")

                if i == 0:
                    # fastText style
                    if len(values) == 2:
                        weight = init_embeddings(len(vocab), values[1], unif)
                        continue
                    # glove style
                    else:
                        weight = init_embeddings(len(vocab), len(values[1:]), unif)
                word = values[0]
                if word in vocab:
                    vec = np.asarray(values[1:], dtype=np.float32)
                    weight[vocab[word]] = vec
        if '[PAD]' in vocab:
            weight[vocab['[PAD]']] = 0.0
        
        embeddings = nn.Embedding(weight.shape[0], weight.shape[1])
        embeddings.weight = nn.Parameter(torch.from_numpy(weight).float())
        return embeddings, weight.shape[1]
    
    @staticmethod
    def from_binary(filename, vocab, unif=0.25):
        def read_word(f):

            s = bytearray()
            ch = f.read(1)

            while ch != b' ':
                s.extend(ch)
                ch = f.read(1)
            s = s.decode('utf-8')
            # Only strip out normal space and \n not other spaces which are words.
            return s.strip(' \n')

        vocab_size = len(vocab)
        with io.open(filename, "rb") as f:
            header = f.readline()
            file_vocab_size, embed_dim = map(int, header.split())
            weight = init_embeddings(len(vocab), embed_dim, unif)
            if '[PAD]' in vocab:
                weight[vocab['[PAD]']] = 0.0
            width = 4 * embed_dim
            for i in range(file_vocab_size):
                word = read_word(f)
                raw = f.read(width)
                if word in vocab:
                    vec = np.fromstring(raw, dtype=np.float32)
                    weight[vocab[word]] = vec
        embeddings = nn.Embedding(weight.shape[0], weight.shape[1])
        embeddings.weight = nn.Parameter(torch.from_numpy(weight).float())
        return embeddings, embed_dim


```

### Now to run some stuff!

We did a lot of work to set things up, but its pretty boilerplate and we will reuse a lot of it.  So far, we made 2 classifiers we can run along with code to train and evaluate our models, and a reader to load our data.



```python
BASE = 'sst2'
TRAIN = os.path.join(BASE, 'stsa.binary.phrases.train')
VALID = os.path.join(BASE, 'stsa.binary.dev')
TEST = os.path.join(BASE, 'stsa.binary.test')
PRETRAINED_EMBEDDINGS_FILE = 'GoogleNews-vectors-negative300.bin'


```

Lets read in our datasets:


```python
r = Reader((TRAIN, VALID, TEST,))
train = r.load(TRAIN)
valid = r.load(VALID)
test = r.load(TEST)
```

## Model trained with randomly initialized embeddings

First, we are going to train a model without any pretrained embeddings for 10 epochs.  During training, we will see the training and validation performance, and after the final epoch, we will see the results from the best model trained on these epochs. 


```python
embed_dim = 300
embeddings = nn.Embedding(len(r.vocab), embed_dim)
model  = ConvClassifier(embeddings, len(r.labels), embed_dim)

num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Model has {num_params} parameters") 


model.to('cuda:0')
loss = torch.nn.NLLLoss()
loss = loss.to('cuda:0')

learnable_params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.Adadelta(learnable_params, lr=1.0)

fit(model, r.labels, optimizer, loss, 10, 50, train, valid, test)

```

    Model has 5442302 parameters
    EPOCH 1
    =================================
    Training Results
    {'acc': 0.5926248359558738, 'precision': 0.6179160630500119, 'recall': 0.6762583118388982, 'f1': 0.6457721335924437}
    Validation Results
    {'acc': 0.7110091743119266, 'precision': 0.7622950819672131, 'recall': 0.6283783783783784, 'f1': 0.6888888888888889}
    New best model 0.71
    EPOCH 2
    =================================
    Training Results
    {'acc': 0.6340224269435168, 'precision': 0.6503253333333333, 'recall': 0.7213611301734542, 'f1': 0.6840038593578207}
    Validation Results
    {'acc': 0.6089449541284404, 'precision': 0.5695006747638327, 'recall': 0.9504504504504504, 'f1': 0.7122362869198312}
    EPOCH 3
    =================================
    Training Results
    {'acc': 0.6637647639713621, 'precision': 0.6763477437133999, 'recall': 0.7433919401784236, 'f1': 0.7082868319298364}
    Validation Results
    {'acc': 0.7144495412844036, 'precision': 0.7506426735218509, 'recall': 0.6576576576576577, 'f1': 0.7010804321728692}
    New best model 0.71
    EPOCH 4
    =================================
    Training Results
    {'acc': 0.6873481373682775, 'precision': 0.6971548679277991, 'recall': 0.7613289476797842, 'f1': 0.7278300606279975}
    Validation Results
    {'acc': 0.7075688073394495, 'precision': 0.6556836902800659, 'recall': 0.8963963963963963, 'f1': 0.7573739295908659}
    EPOCH 5
    =================================
    Training Results
    {'acc': 0.7016800717246398, 'precision': 0.7109843018933928, 'recall': 0.7695165526870016, 'f1': 0.7390933781833471}
    Validation Results
    {'acc': 0.6490825688073395, 'precision': 0.597457627118644, 'recall': 0.9527027027027027, 'f1': 0.734375}
    EPOCH 6
    =================================
    Training Results
    {'acc': 0.7181819363054014, 'precision': 0.7250941083778342, 'recall': 0.783998674838496, 'f1': 0.7533967777512478}
    Validation Results
    {'acc': 0.7591743119266054, 'precision': 0.7378048780487805, 'recall': 0.8175675675675675, 'f1': 0.7756410256410255}
    New best model 0.76
    EPOCH 7
    =================================
    Training Results
    {'acc': 0.7320201140837567, 'precision': 0.7382341929658423, 'recall': 0.7932274781703306, 'f1': 0.7647434581251569}
    Validation Results
    {'acc': 0.7431192660550459, 'precision': 0.7370689655172413, 'recall': 0.7702702702702703, 'f1': 0.7533039647577091}
    EPOCH 8
    =================================
    Training Results
    {'acc': 0.7397253154194982, 'precision': 0.7452230704735008, 'recall': 0.7992380321351664, 'f1': 0.7712860095226134}
    Validation Results
    {'acc': 0.6788990825688074, 'precision': 0.6209439528023599, 'recall': 0.9481981981981982, 'f1': 0.7504456327985739}
    EPOCH 9
    =================================
    Training Results
    {'acc': 0.7485089850703603, 'precision': 0.7541725852272727, 'recall': 0.8040890697839513, 'f1': 0.7783313290958025}
    Validation Results
    {'acc': 0.7545871559633027, 'precision': 0.7281746031746031, 'recall': 0.8265765765765766, 'f1': 0.7742616033755274}
    EPOCH 10
    =================================
    Training Results
    {'acc': 0.7572016995621159, 'precision': 0.7618233111935491, 'recall': 0.8115431032442794, 'f1': 0.7858976121728768}
    Validation Results
    {'acc': 0.7591743119266054, 'precision': 0.7303149606299213, 'recall': 0.8355855855855856, 'f1': 0.7794117647058825}
    Final result
    {'acc': 0.7391543108182317, 'precision': 0.728421052631579, 'recall': 0.7612761276127613, 'f1': 0.7444862829478214}





    0.7391543108182317



Yikes, thats not very encouraging!  What about our LSTM?


```python
embed_dim = 300
embeddings = nn.Embedding(len(r.vocab), embed_dim)
model  = LSTMClassifier(embeddings, len(r.labels), embed_dim, 100, hidden_units=[100])

num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Model has {num_params} parameters") 


model.to('cuda:0')
loss = torch.nn.NLLLoss()
loss = loss.to('cuda:0')

learnable_params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.Adadelta(learnable_params, lr=1.0)

fit(model, r.labels, optimizer, loss, 10, 50, train, valid, test)

```

    /usr/local/lib/python3.6/dist-packages/torch/nn/modules/rnn.py:54: UserWarning: dropout option adds dropout after all but last recurrent layer, so non-zero dropout expects num_layers greater than 1, but got dropout=0.5 and num_layers=1
      "num_layers={}".format(dropout, num_layers))


    Model has 5342502 parameters
    EPOCH 1
    =================================
    Training Results
    {'acc': 0.6416496667143099, 'precision': 0.648109286089027, 'recall': 0.7600511133722994, 'f1': 0.6996307873269656}
    Validation Results
    {'acc': 0.7488532110091743, 'precision': 0.7300613496932515, 'recall': 0.8040540540540541, 'f1': 0.765273311897106}
    New best model 0.75
    EPOCH 2
    =================================
    Training Results
    {'acc': 0.7467158690765453, 'precision': 0.7576038743550285, 'recall': 0.7921862798457133, 'f1': 0.7745092368734602}
    Validation Results
    {'acc': 0.7786697247706422, 'precision': 0.747534516765286, 'recall': 0.8536036036036037, 'f1': 0.7970557308096741}
    New best model 0.78
    EPOCH 3
    =================================
    Training Results
    {'acc': 0.7794467327607489, 'precision': 0.7900387712496272, 'recall': 0.8149033342009986, 'f1': 0.8022784456248252}
    Validation Results
    {'acc': 0.783256880733945, 'precision': 0.7958236658932715, 'recall': 0.7725225225225225, 'f1': 0.7839999999999999}
    New best model 0.78
    EPOCH 4
    =================================
    Training Results
    {'acc': 0.7937786671171113, 'precision': 0.8048943938623654, 'recall': 0.8242267919259803, 'f1': 0.8144458863830334}
    Validation Results
    {'acc': 0.7901376146788991, 'precision': 0.7713097713097713, 'recall': 0.8355855855855856, 'f1': 0.8021621621621621}
    New best model 0.79
    EPOCH 5
    =================================
    Training Results
    {'acc': 0.8033159652291421, 'precision': 0.814348632359759, 'recall': 0.8313258714120069, 'f1': 0.8227496809096125}
    Validation Results
    {'acc': 0.7878440366972477, 'precision': 0.8018648018648019, 'recall': 0.7747747747747747, 'f1': 0.7880870561282932}
    EPOCH 6
    =================================
    Training Results
    {'acc': 0.8122165772274269, 'precision': 0.8227892183038098, 'recall': 0.8386379232826143, 'f1': 0.8306379787184175}
    Validation Results
    {'acc': 0.7889908256880734, 'precision': 0.8125, 'recall': 0.7612612612612613, 'f1': 0.786046511627907}
    EPOCH 7
    =================================
    Training Results
    {'acc': 0.8170501942542326, 'precision': 0.8279907814791536, 'recall': 0.841666863863319, 'f1': 0.8347728126173487}
    Validation Results
    {'acc': 0.8027522935779816, 'precision': 0.7995594713656388, 'recall': 0.8175675675675675, 'f1': 0.8084632516703786}
    New best model 0.80
    EPOCH 8
    =================================
    Training Results
    {'acc': 0.8234690297683243, 'precision': 0.8332597224482206, 'recall': 0.8482453441870371, 'f1': 0.8406857571706653}
    Validation Results
    {'acc': 0.7924311926605505, 'precision': 0.8065268065268065, 'recall': 0.7792792792792793, 'f1': 0.7926689576174113}
    EPOCH 9
    =================================
    Training Results
    {'acc': 0.824742401995816, 'precision': 0.8359920588578769, 'recall': 0.846991173477839, 'f1': 0.8414556738839128}
    Validation Results
    {'acc': 0.7981651376146789, 'precision': 0.8116279069767441, 'recall': 0.786036036036036, 'f1': 0.7986270022883294}
    EPOCH 10
    =================================
    Training Results
    {'acc': 0.8271332233209028, 'precision': 0.8384886956115125, 'recall': 0.8486476253579119, 'f1': 0.8435375749735388}
    Validation Results
    {'acc': 0.7993119266055045, 'precision': 0.799554565701559, 'recall': 0.8085585585585585, 'f1': 0.8040313549832027}
    Final result
    {'acc': 0.8050521691378364, 'precision': 0.7991360691144709, 'recall': 0.8140814081408141, 'f1': 0.8065395095367847}





    0.8050521691378364



## Same model with pre-trained word embeddings

The models below are identical to the ones above, the only difference is that we are going to initialize the embeddings using our previously defined `EmbeddingsReader`.  First lets take a look at our CNN model again.  Notice we only run 5 epochs here instead of 10!


```python
embeddings, embed_dim = EmbeddingsReader.from_binary(PRETRAINED_EMBEDDINGS_FILE, r.vocab)
model  = ConvClassifier(embeddings, len(r.labels), embed_dim)
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Model has {num_params} parameters") 


model.to('cuda:0')
loss = torch.nn.NLLLoss()
loss = loss.to('cuda:0')

learnable_params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.Adadelta(learnable_params, lr=1.0)

fit(model, r.labels, optimizer, loss, 5, 50, train, valid, test)
```

    /usr/local/lib/python3.6/dist-packages/ipykernel_launcher.py:60: DeprecationWarning: The binary mode of fromstring is deprecated, as it behaves surprisingly on unicode inputs. Use frombuffer instead


    Model has 5442302 parameters
    EPOCH 1
    =================================
    Training Results
    {'acc': 0.8329153727212484, 'precision': 0.8410831129054712, 'recall': 0.8577817742965995, 'f1': 0.8493503754817998}
    Validation Results
    {'acc': 0.8268348623853211, 'precision': 0.9080779944289693, 'recall': 0.7342342342342343, 'f1': 0.8119551681195517}
    New best model 0.83
    EPOCH 2
    =================================
    Training Results
    {'acc': 0.8798742220085498, 'precision': 0.8858578775128565, 'recall': 0.8967793842731726, 'f1': 0.8912851750373358}
    Validation Results
    {'acc': 0.8474770642201835, 'precision': 0.8373101952277657, 'recall': 0.8693693693693694, 'f1': 0.8530386740331491}
    New best model 0.85
    EPOCH 3
    =================================
    Training Results
    {'acc': 0.8955834773456686, 'precision': 0.9008221873462791, 'recall': 0.9100309993137556, 'f1': 0.9054031783402}
    Validation Results
    {'acc': 0.8486238532110092, 'precision': 0.8436123348017621, 'recall': 0.8626126126126126, 'f1': 0.8530066815144767}
    New best model 0.85
    EPOCH 4
    =================================
    Training Results
    {'acc': 0.9059523654838165, 'precision': 0.9103606664948091, 'recall': 0.9192361390473035, 'f1': 0.91477687507359}
    Validation Results
    {'acc': 0.841743119266055, 'precision': 0.9090909090909091, 'recall': 0.7657657657657657, 'f1': 0.8312958435207825}
    EPOCH 5
    =================================
    Training Results
    {'acc': 0.9139694130793519, 'precision': 0.9181293410925474, 'recall': 0.9258856101658818, 'f1': 0.9219911634756995}
    Validation Results
    {'acc': 0.8428899082568807, 'precision': 0.851258581235698, 'recall': 0.8378378378378378, 'f1': 0.844494892167991}
    Final result
    {'acc': 0.8725974739154311, 'precision': 0.8537095088819227, 'recall': 0.8987898789878987, 'f1': 0.8756698821007501}





    0.8725974739154311



Much better!  And now the LSTM!


```python
model  = LSTMClassifier(embeddings, len(r.labels), embed_dim, 100, hidden_units=[100])

num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Model has {num_params} parameters") 


model.to('cuda:0')
loss = torch.nn.NLLLoss()
loss = loss.to('cuda:0')

learnable_params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.Adadelta(learnable_params, lr=1.0)

fit(model, r.labels, optimizer, loss, 5, 50, train, valid, test)
```

    /usr/local/lib/python3.6/dist-packages/torch/nn/modules/rnn.py:54: UserWarning: dropout option adds dropout after all but last recurrent layer, so non-zero dropout expects num_layers greater than 1, but got dropout=0.5 and num_layers=1
      "num_layers={}".format(dropout, num_layers))


    Model has 5342502 parameters
    EPOCH 1
    =================================
    Training Results
    {'acc': 0.870012084042567, 'precision': 0.8834494400722794, 'recall': 0.8792683215409736, 'f1': 0.8813539220569748}
    Validation Results
    {'acc': 0.8371559633027523, 'precision': 0.8081632653061225, 'recall': 0.8918918918918919, 'f1': 0.8479657387580299}
    New best model 0.84
    EPOCH 2
    =================================
    Training Results
    {'acc': 0.904562050908902, 'precision': 0.9189545934530094, 'recall': 0.9061028419981543, 'f1': 0.9124834677755669}
    Validation Results
    {'acc': 0.8646788990825688, 'precision': 0.9137055837563451, 'recall': 0.8108108108108109, 'f1': 0.8591885441527446}
    New best model 0.86
    EPOCH 3
    =================================
    Training Results
    {'acc': 0.9171658372422395, 'precision': 0.9286806517895542, 'recall': 0.9197804018078989, 'f1': 0.9242090996635478}
    Validation Results
    {'acc': 0.8600917431192661, 'precision': 0.8340248962655602, 'recall': 0.9054054054054054, 'f1': 0.8682505399568036}
    EPOCH 4
    =================================
    Training Results
    {'acc': 0.9252738399968815, 'precision': 0.9348261076703192, 'recall': 0.9286542511654322, 'f1': 0.9317299588076782}
    Validation Results
    {'acc': 0.856651376146789, 'precision': 0.8444924406047516, 'recall': 0.8806306306306306, 'f1': 0.8621830209481808}
    EPOCH 5
    =================================
    Training Results
    {'acc': 0.9308091111082236, 'precision': 0.9387294497766796, 'recall': 0.9350197591045695, 'f1': 0.9368709321762635}
    Validation Results
    {'acc': 0.8635321100917431, 'precision': 0.9156010230179028, 'recall': 0.8063063063063063, 'f1': 0.8574850299401198}
    Final result
    {'acc': 0.8802855573860516, 'precision': 0.9157641395908543, 'recall': 0.8371837183718371, 'f1': 0.8747126436781609}





    0.8802855573860516



#### A quick note about these models on this data

Both of these models are surprisingly strong baselines and do fairly well on this dataset averaged over many runs.  Even with only 2-5 epochs of data it is quite common to see scores higher than in the Kim 2014 paper.

## Conclusions

### Its not hard to get good performance with a Deep Learning model for Text Classification

We saw above how to get good results on the SST-2 dataset using fairly simple deep learning models, even with very few training epochs.  This behavior is not limited to a single dataset -- these results have been shown over and over.  Also, using PyTorch, we were able to code an entire pipeline in this minimalistic notebook.

### Pre-trained embeddings often help

We can see that pre-trained embeddings can have a massive impact on the performance of our models, especially for smaller datasets.  The `word2vec` algorithm caused an explosion in the NLP community -- even though pre-training embeddings had been widely studied prior to that work, the results were reliable and fast.  `GloVe` and `fastText` embeddings came shortly thereafter, and all 3 models are in quite common use today.  The code above can load any of these flavors of embeddings and incorporate them into downstream models for large improvements.

For large datasets, like those used in Language Modeling and Neural Machine Translation, models are typically trained from random embeddings, which are sufficient in those cases.

### Incorporating pre-trained embeddings into your model is simple

The file formats are very simple to read and can be incorporated with only a few lines of code.  In some cases, memory-mapping the file can increase the loading speed.  This is implemented in [Baseline](https://github.com/dpressel/baseline/)

### Some Further Resources

- [The Unreasonable Effectiveness of Recurrent Neural Networks, Karpathy, 2015](https://karpathy.github.io/2015/05/21/rnn-effectiveness/)
- Tensorflow tutorial for word2vec: https://www.tensorflow.org/tutorials/representation/word2vec
- Tensorflow docs on feature columns (for images above): https://www.tensorflow.org/guide/feature_columns
- Xin Rong, wrote some amazing software to visualize word embeddings and the training process.  Sadly, Xin is no longer with us -- he was a great researcher and an awesome guy.  We miss him.
  - https://ronxin.github.io/wevi/
  - Accompanying talk from a2-dlearn 2015: https://www.youtube.com/channel/UCVdeq2cIxnujw2kTdzg2N5g
  - https://ronxin.github.io/lamvi/dist/#model=word2vec&backend=browser&query_in=darcy&query_out=G_bennet,B_circumstances
  - Accompanying paper for Lamvi [Visual Tools for Debugging Neural Language Models, Rong & Adar, 2016](http://www.cond.org/ICML16_NeuralVis.pdf)


