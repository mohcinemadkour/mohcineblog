title: Transfer learning in NLP Part II : Contextualized embeddings
Date: 2019-07-07 13:01
Category: Machine Learning, July 2019, Transfer learning, Context vectors
Tags: NLP, July 2019, Transfer learning, Context vectors
Slug: Machine Learning, July 2019, Transfer learning, Context vectors
Author: Mohcine Madkour
Email:mohcine.madkour@gmail.com


In this section, we are going to learn how to train an LSTM-based word-level language model.  Then we will take load a pre-trained langage model checkpoint and use everything below the output layers as the lower layers of our previously defined classification model.  We dont really need to change anything else, we just need to pass this whole network as the `embedding` parameter to the model.

## LSTM Language Models

We are going to quickly build an LSTM language model so that we can see how the training works.  For both our objectives and our metrics, we are interested in the perplexity, which is the exponentiated cross-entropy loss.


```python
!wget https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip
!unzip wikitext-2-v1.zip
```

    --2019-06-30 19:10:48--  https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip
    Resolving s3.amazonaws.com (s3.amazonaws.com)... 52.216.134.253
    Connecting to s3.amazonaws.com (s3.amazonaws.com)|52.216.134.253|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 4475746 (4.3M) [application/zip]
    Saving to: ‘wikitext-2-v1.zip.5’
    
    wikitext-2-v1.zip.5 100%[===================>]   4.27M  18.2MB/s    in 0.2s    
    
    2019-06-30 19:10:49 (18.2 MB/s) - ‘wikitext-2-v1.zip.5’ saved [4475746/4475746]
    
    Archive:  wikitext-2-v1.zip
    replace wikitext-2/wiki.test.tokens? [y]es, [n]o, [A]ll, [N]one, [r]ename: 

Our LSTM model will be a word-based model.  We will have a randomly trained embedding to start and we will put each output timestep through our LSTM blocks and then project to the output vocabulary size. At every step of training, we will detach our hidden states, preventing full backpropagation, but we will initialize the new batch from our old hidden state.  We will also create a function that resets the hidden state, which we will use at the start of each epoch to zero out the hidden states.


```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple
import os
import io
import re
import codecs
import numpy as np
from collections import Counter
import math
import time

class LSTMLanguageModel(nn.Module):

    def __init__(self, vocab_size, embed_dim, hidden_dim, dropout=0.5, layers=2):
        super().__init__()
        self.layers = layers
        self.hidden_dim = hidden_dim
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.rnn = torch.nn.LSTM(embed_dim,
                                 hidden_dim,
                                 layers,
                                 dropout=dropout,
                                 bidirectional=False,
                                 batch_first=True)
        self.proj = nn.Linear(embed_dim, vocab_size)
        self.proj.bias.data.zero_()

        # Tie weights
        self.proj.weight = self.embed.weight

    def forward(self, x, hidden):
        emb = self.embed(x)
        decoded, hidden = self.rnn(emb, hidden)
        return self.proj(decoded), hidden
        
    def init_hidden(self, batchsz):
        weight = next(self.parameters()).data
        return (torch.autograd.Variable(weight.new(self.layers, batchsz, self.hidden_dim).zero_()),
                torch.autograd.Variable(weight.new(self.layers, batchsz, self.hidden_dim).zero_()))

```

Our dataset reader will read in a sequence of words and vectorize them.  We would like this to be a long sequence of text (like maybe a book), and we will read this in contiguously.  Our task is to learn to predict the next word, so we will end up using this sequence for input and output


```python



class WordDatasetReader(object):
    """Provide a base-class to do operations to read words to tensors
    """
    def __init__(self, nctx, vectorizer=None):
        self.nctx = nctx
        self.num_words = {}
        self.vectorizer = vectorizer if vectorizer else self._vectorizer

    def build_vocab(self, files, min_freq=0):
        x = Counter()

        for file in files:
            if file is None:
                continue
            self.num_words[file] = 0
            with codecs.open(file, encoding='utf-8', mode='r') as f:
                sentences = []
                for line in f:
                    split_sentence = line.split() + ['<EOS>']
                    self.num_words[file] += len(split_sentence)
                    sentences += split_sentence
                x.update(Counter(sentences))
        x = dict(filter(lambda cnt: cnt[1] >= min_freq, x.items()))
        alpha = list(x.keys())
        alpha.sort()
        self.vocab = {w: i+1 for i, w in enumerate(alpha)}
        self.vocab['[PAD]'] = 0
    
    def _vectorizer(self, words: List[str]) -> List[int]:
        return [self.vocab.get(w, 0) for w in words]

      
    def load_features(self, filename):

        with codecs.open(filename, encoding='utf-8', mode='r') as f:
            sentences = []
            for line in f:
                sentences += line.strip().split() + ['<EOS>']
            return torch.tensor(self.vectorizer(sentences), dtype=torch.long)

    def load(self, filename, batch_size):
        x_tensor = self.load_features(filename)
        rest = x_tensor.shape[0]//batch_size
        num_steps = rest // self.nctx
        # if num_examples is divisible by batchsz * nctx (equivalent to rest is divisible by nctx), we
        # have a problem. reduce rest in that case.

        if rest % self.nctx == 0:
            rest = rest-1
        trunc = batch_size * rest
        
        x_tensor = x_tensor.narrow(0, 0, trunc)
        x_tensor = x_tensor.view(batch_size, -1).contiguous()
        return x_tensor
     
    
```

This class will keep track of our running average as we go so we dont have to remember to average things in our loops


```python
class Average(object):
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

```

We are going to train on batches of contiguous text. Our batches will have been pre-created by the loader.  Each batch will be `BxT` where `B` is the batch size we specified, and `T` is the number of backprop steps through time.


```python
class SequenceCriterion(nn.Module):

    def __init__(self):
        super().__init__()
        self.crit = nn.CrossEntropyLoss(ignore_index=0, size_average=True)
          
    def forward(self, inputs, targets):
        """Evaluate some loss over a sequence.

        :param inputs: torch.FloatTensor, [B, .., C] The scores from the model. Batch First
        :param targets: torch.LongTensor, The labels.

        :returns: torch.FloatTensor, The loss.
        """
        total_sz = targets.nelement()
        loss = self.crit(inputs.view(total_sz, -1), targets.view(total_sz))
        return loss

class LMTrainer:
  
    def __init__(self, optimizer: torch.optim.Optimizer, nctx):
        self.optimizer = optimizer
        self.nctx = nctx
    
    def run(self, model, train_data, loss_function, batch_size=20, clip=0.25):
        avg_loss = Average('average_train_loss')
        metrics = {}
        self.optimizer.zero_grad()
        start = time.time()
        model.train()
        hidden = model.init_hidden(batch_size)
        num_steps = train_data.shape[1]//self.nctx
        for i in range(num_steps):
            x = train_data[:,i*self.nctx:(i + 1) * self.nctx]
            y = train_data[:, i*self.nctx+1:(i + 1)*self.nctx + 1]
            labels = y.to('cuda:0').transpose(0, 1).contiguous()
            inputs = x.to('cuda:0')
            logits, (h, c) = model(inputs, hidden)
            hidden = (h.detach(), c.detach())
            logits = logits.transpose(0, 1).contiguous()
            loss = loss_function(logits, labels)
            loss.backward()

            avg_loss.update(loss.item())

            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            self.optimizer.step()
            self.optimizer.zero_grad()
            if (i + 1) % 100 == 0:
                print(avg_loss)

        # How much time elapsed in minutes
        elapsed = (time.time() - start)/60
        train_token_loss = avg_loss.avg
        train_token_ppl = math.exp(train_token_loss)
        metrics['train_elapsed_min'] = elapsed
        metrics['average_train_loss'] = train_token_loss
        metrics['train_ppl'] = train_token_ppl
        return metrics

class LMEvaluator:
    def __init__(self, nctx):
        self.nctx = nctx
        
    def run(self, model, valid_data, loss_function, batch_size=20):
        avg_valid_loss = Average('average_valid_loss')
        start = time.time()
        model.eval()
        hidden = model.init_hidden(batch_size)
        metrics = {}
        num_steps = valid_data.shape[1]//self.nctx
        for i in range(num_steps):

            with torch.no_grad():
                x = valid_data[:,i*self.nctx:(i + 1) * self.nctx]
                y = valid_data[:, i*self.nctx+1:(i + 1)*self.nctx + 1]
                labels = y.to('cuda:0').transpose(0, 1).contiguous()
                inputs = x.to('cuda:0')
                
                logits, hidden = model(inputs, hidden)
                logits = logits.transpose(0, 1).contiguous()
                loss = loss_function(logits, labels)
                avg_valid_loss.update(loss.item())

        valid_token_loss = avg_valid_loss.avg
        valid_token_ppl = math.exp(valid_token_loss)

        elapsed = (time.time() - start)/60
        metrics['valid_elapsed_min'] = elapsed

        metrics['average_valid_loss'] = valid_token_loss
        metrics['average_valid_word_ppl'] = valid_token_ppl
        return metrics
      
def fit_lm(model, optimizer, epochs, batch_size, nctx, train_data, valid_data):

    loss = SequenceCriterion()
    trainer = LMTrainer(optimizer, nctx)
    evaluator = LMEvaluator(nctx)
    best_acc = 0.0

    metrics = evaluator.run(model, valid_data, loss, batch_size)

    for epoch in range(epochs):

        print('EPOCH {}'.format(epoch + 1))
        print('=================================')
        print('Training Results')
        metrics = trainer.run(model, train_data, loss, batch_size)
        print(metrics)
        print('Validation Results')
        metrics = evaluator.run(model, valid_data, loss, batch_size)
        print(metrics)
```

Now we will train it on [Wikitext-2, Merity et al. 2016](https://blog.einstein.ai/the-wikitext-long-term-dependency-language-modeling-dataset/).  We will use 35 steps of backprop.


```python
BASE = 'wikitext-2'
TRAIN = os.path.join(BASE, 'wiki.train.tokens')
VALID = os.path.join(BASE, 'wiki.valid.tokens')

batch_size = 20
nctx = 35
reader = WordDatasetReader(nctx)
reader.build_vocab((TRAIN,))

train_set = reader.load(TRAIN, batch_size)
valid_set = reader.load(VALID, batch_size)
```

Lets start with 1 epoch


```python

model = LSTMLanguageModel(len(reader.vocab), 512, 512)
model.to('cuda:0')

num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Model has {num_params} parameters") 


learnable_params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.Adam(learnable_params, lr=0.001)
fit_lm(model, optimizer, 1, batch_size, nctx, train_set, valid_set)

```

    Model has 21274623 parameters


    /usr/local/lib/python3.6/dist-packages/torch/nn/_reduction.py:46: UserWarning: size_average and reduce args will be deprecated, please use reduction='mean' instead.
      warnings.warn(warning.format(ret))


    EPOCH 1
    =================================
    Training Results
    average_train_loss 7.130287 (7.630262)
    average_train_loss 6.948112 (7.174242)
    average_train_loss 6.429612 (6.957250)
    average_train_loss 6.706582 (6.817735)
    average_train_loss 6.480259 (6.716661)
    average_train_loss 6.253604 (6.639940)
    average_train_loss 6.250593 (6.584427)
    average_train_loss 6.086081 (6.535446)
    average_train_loss 6.046218 (6.491021)
    average_train_loss 5.840661 (6.455732)
    average_train_loss 6.127773 (6.425025)
    average_train_loss 5.766460 (6.398616)
    average_train_loss 6.137995 (6.376816)
    average_train_loss 6.115303 (6.351095)
    average_train_loss 6.203366 (6.333509)
    average_train_loss 6.009459 (6.318195)
    average_train_loss 6.126120 (6.297565)
    average_train_loss 5.796104 (6.276726)
    average_train_loss 5.737082 (6.260670)
    average_train_loss 5.954897 (6.243683)
    average_train_loss 5.674878 (6.226430)
    average_train_loss 5.613625 (6.207307)
    average_train_loss 5.878324 (6.189868)
    average_train_loss 5.824322 (6.178013)
    average_train_loss 5.932457 (6.164326)
    average_train_loss 5.771354 (6.153274)
    average_train_loss 5.401644 (6.139012)
    average_train_loss 5.825085 (6.124258)
    average_train_loss 5.493943 (6.110777)
    {'train_elapsed_min': 2.2767986059188843, 'average_train_loss': 6.0969437521813985, 'train_ppl': 444.4971984264256}
    Validation Results
    {'valid_elapsed_min': 0.07908193667729696, 'average_valid_loss': 5.534342344345585, 'average_valid_word_ppl': 253.24118736148796}


We can sample out of our language model using the code below.


```python
def sample(model, index2word, start_word='the', maxlen=20):
  

    model.eval() 
    words = [start_word]
    x = torch.tensor(reader.vocab.get(start_word)).long().reshape(1, 1).to('cuda:0')
    hidden = model.init_hidden(1)

    with torch.no_grad():
        for i in range(20):
            output, hidden = model(x, hidden)
            word_softmax = output.squeeze().exp().cpu()
            selected = torch.multinomial(word_softmax, 1)[0]
            x.fill_(selected)
            word = index2word[selected.item()]
            words.append(word)
    words.append('...')
    return words

index2word = {i: w for w, i in reader.vocab.items()}
words = sample(model, index2word)
print(' '.join(words))

```

    the latter story pass that would be in Park or Ireland . Like Liam Stuart illustrator , NC apologize and livestock ...


Lets train a few more epochs and try again


```python
fit_lm(model, optimizer, 3, batch_size, 35, train_set, valid_set)
```

    /usr/local/lib/python3.6/dist-packages/torch/nn/_reduction.py:46: UserWarning: size_average and reduce args will be deprecated, please use reduction='mean' instead.
      warnings.warn(warning.format(ret))


    EPOCH 1
    =================================
    Training Results
    average_train_loss 5.888901 (5.598958)
    average_train_loss 5.818975 (5.584962)
    average_train_loss 5.503317 (5.580568)
    average_train_loss 5.877174 (5.584086)
    average_train_loss 5.637257 (5.563949)
    average_train_loss 5.447403 (5.540543)
    average_train_loss 5.460862 (5.531339)
    average_train_loss 5.488514 (5.525078)
    average_train_loss 5.359737 (5.517013)
    average_train_loss 5.121772 (5.509718)
    average_train_loss 5.441720 (5.503375)
    average_train_loss 5.280029 (5.499962)
    average_train_loss 5.543726 (5.500008)
    average_train_loss 5.556562 (5.494267)
    average_train_loss 5.593565 (5.495319)
    average_train_loss 5.347257 (5.496266)
    average_train_loss 5.519910 (5.489749)
    average_train_loss 5.264927 (5.483263)
    average_train_loss 5.207999 (5.481013)
    average_train_loss 5.434073 (5.476722)
    average_train_loss 5.112748 (5.471222)
    average_train_loss 5.142471 (5.463090)
    average_train_loss 5.362827 (5.455768)
    average_train_loss 5.287307 (5.454580)
    average_train_loss 5.420770 (5.449693)
    average_train_loss 5.358116 (5.448896)
    average_train_loss 5.019379 (5.443910)
    average_train_loss 5.375151 (5.437743)
    average_train_loss 5.061219 (5.432098)
    {'train_elapsed_min': 2.3279770851135253, 'average_train_loss': 5.425043830046748, 'train_ppl': 227.0212964512552}
    Validation Results
    {'valid_elapsed_min': 0.0792834202448527, 'average_valid_loss': 5.2854782812057, 'average_valid_word_ppl': 197.4485968212379}
    EPOCH 2
    =================================
    Training Results
    average_train_loss 5.594471 (5.241230)
    average_train_loss 5.464106 (5.238882)
    average_train_loss 5.156283 (5.241025)
    average_train_loss 5.523200 (5.251309)
    average_train_loss 5.250049 (5.232446)
    average_train_loss 5.129644 (5.205283)
    average_train_loss 5.083561 (5.195768)
    average_train_loss 5.166030 (5.192733)
    average_train_loss 5.086248 (5.188236)
    average_train_loss 4.777071 (5.182954)
    average_train_loss 5.161200 (5.178061)
    average_train_loss 5.008852 (5.176982)
    average_train_loss 5.142172 (5.180397)
    average_train_loss 5.281511 (5.176891)
    average_train_loss 5.325745 (5.180915)
    average_train_loss 5.139209 (5.184646)
    average_train_loss 5.210761 (5.179825)
    average_train_loss 5.041038 (5.176068)
    average_train_loss 4.949179 (5.175644)
    average_train_loss 5.189332 (5.173076)
    average_train_loss 4.856432 (5.168824)
    average_train_loss 4.836321 (5.162272)
    average_train_loss 5.064670 (5.156098)
    average_train_loss 4.960176 (5.156356)
    average_train_loss 5.104852 (5.152806)
    average_train_loss 5.087171 (5.153634)
    average_train_loss 4.799379 (5.150230)
    average_train_loss 5.135460 (5.145286)
    average_train_loss 4.837797 (5.140926)
    {'train_elapsed_min': 2.3252848744392396, 'average_train_loss': 5.135120418061357, 'train_ppl': 169.88477583838508}
    Validation Results
    {'valid_elapsed_min': 0.0793849547704061, 'average_valid_loss': 5.172104538640668, 'average_valid_word_ppl': 176.2854469011485}
    EPOCH 3
    =================================
    Training Results
    average_train_loss 5.397759 (5.021339)
    average_train_loss 5.223657 (5.019342)
    average_train_loss 4.937864 (5.024831)
    average_train_loss 5.331026 (5.037493)
    average_train_loss 5.013303 (5.017465)
    average_train_loss 4.930802 (4.989496)
    average_train_loss 4.877040 (4.980525)
    average_train_loss 5.008083 (4.979450)
    average_train_loss 4.872459 (4.977045)
    average_train_loss 4.537238 (4.972334)
    average_train_loss 4.895549 (4.967794)
    average_train_loss 4.778692 (4.967996)
    average_train_loss 4.913031 (4.972698)
    average_train_loss 4.998916 (4.970212)
    average_train_loss 5.101923 (4.975337)
    average_train_loss 4.911569 (4.980182)
    average_train_loss 5.020348 (4.975942)
    average_train_loss 4.840934 (4.973322)
    average_train_loss 4.820501 (4.974319)
    average_train_loss 4.994349 (4.972502)
    average_train_loss 4.695219 (4.968739)
    average_train_loss 4.709369 (4.962705)
    average_train_loss 4.882851 (4.957035)
    average_train_loss 4.744756 (4.957699)
    average_train_loss 4.943813 (4.954420)
    average_train_loss 4.885959 (4.955989)
    average_train_loss 4.626930 (4.953493)
    average_train_loss 4.934981 (4.949243)
    average_train_loss 4.673239 (4.945310)
    {'train_elapsed_min': 2.333573551972707, 'average_train_loss': 4.939903614627328, 'train_ppl': 139.75677840163237}
    Validation Results
    {'valid_elapsed_min': 0.07996076345443726, 'average_valid_loss': 5.096776191649899, 'average_valid_word_ppl': 163.49398352530844}



```python
index2word = {i: w for w, i in reader.vocab.items()}
words = sample(model, index2word)
print(' '.join(words))
```

    the Supreme Court did not introduce any contact with its way and Grosser Davies of the chance of the country . ...



## ELMo

For the rest of this section, we will focus on ELMo ([Peters et al 2018](https://export.arxiv.org/pdf/1802.05365)), a language model with an embedding layer and 2 subsequent LSTM layers.  Actually, at training time, ELMo is basically two LMs -- one working in the forward direction and one working in the backward direction.  The losses for the forward and reverse directions are averaged.  At inference time, the forward and backward layers are aggregated into a single bidirectional representation at each layer.

In our example, we created a word-based LM.  You might have been wondering what to do about words that we havent seen yet -- and that is a valid concern!   Instead of using a word embedding layer like our example above, what if we had a model that used a character compositional approach, taking each character in a word and applying a pooling operation to yield a word representation.  This would mean that the model can handle words that its never seen in the input before.

This is exactly what ELMo does -- its based on the research of [Kim et al. 2015](https://arxiv.org/abs/1508.06615).  

There is a nice [slide deck by the authors here](http://www.people.fas.harvard.edu/~yoonkim/data/char-nlm-slides.pdf), but the key high-level points are listed here:

### Kim Language Model

* **Goal**: predict the next word in the sentence (causal LM) but account for unseen words by using a character compositional approach that relies on letters within the pre-segmented words.  This also has the important impact of reducing the number of parameters required in the model drastically over word-level models.

* **Using**: LSTM layers that take in a word representation for each position.  Each word is put in and used to predict the next word over a context

* **The Twist**: use embeddings approach from [dos Santos & Zadrozny 2014](http://proceedings.mlr.press/v32/santos14.pdf) to represent words, but add parallel filters as in [Kim 2014](https://www.aclweb.org/anthology/D14-1181).  Also, add highway layers on top of the base model



### ELMo Language Model

* **Goal**: predict the next word in the sentence (causal LM) on the forward sequence **and** predict the previous word on the sentence conditioned on the following context.  

* **Using**: LSTM layers as before, but bidirectional, sum the forward and backward loss to make one big loss

* **The Twist** Potentially use all layers of the model (except we dont need head with the big softmax at the end over the words). After the fact, we can freeze our biLM embeddings but still provide useful information by learning a linear combination of the layers during downstream training.  During the biLM training, these scalars dont exist



### ELMo with AllenNLP

Even though ELMo is just a network like described above, there are a lot of details to getting it set up and reloading the pre-trained checkpoints that are provided, and these details are not really important for demonstration purposes.  So, we will just install [AllenNLP](https://github.com/allenai/allennlp) and use it as a contextual embedding layer.

If you are interested in learning more about using ELMo with AllenNLP, they have provided a [tutorial here](https://github.com/allenai/allennlp/blob/master/tutorials/how_to/elmo.md)

#### TensorFlow and ELMo

ELMo was originally trained with TensorFlow.  You can find the code to train and use it in the [bilm-tf repository](https://github.com/allenai/bilm-tf/tree/master/bilm)

TF-Hub contains the [pre-trained ELMo model](https://tfhub.dev/google/elmo/2) and is very easy to integrate if you are using TensorFlow already.  The model takes a sequence of words (mixed-case) as inputs and can just be "glued" in to your existing models as a sub-graph of your own.



```python
!pip install allennlp
```

    Collecting allennlp
      Using cached https://files.pythonhosted.org/packages/30/8c/72b14d20c9cbb0306939ea41109fc599302634fd5c59ccba1a659b7d0360/allennlp-0.8.4-py3-none-any.whl
    Collecting jsonnet>=0.10.0; sys_platform != "win32" (from allennlp)
    [?25l  Downloading https://files.pythonhosted.org/packages/a9/a8/adba6cd0f84ee6ab064e7f70cd03a2836cefd2e063fd565180ec13beae93/jsonnet-0.13.0.tar.gz (255kB)
    [K     |████████████████████████████████| 256kB 3.4MB/s 
    [?25hCollecting numpydoc>=0.8.0 (from allennlp)
      Downloading https://files.pythonhosted.org/packages/6a/f3/7cfe4c616e4b9fe05540256cc9c6661c052c8a4cec2915732793b36e1843/numpydoc-0.9.1.tar.gz
    Collecting pytorch-pretrained-bert>=0.6.0 (from allennlp)
    [?25l  Downloading https://files.pythonhosted.org/packages/d7/e0/c08d5553b89973d9a240605b9c12404bcf8227590de62bae27acbcfe076b/pytorch_pretrained_bert-0.6.2-py3-none-any.whl (123kB)
    [K     |████████████████████████████████| 133kB 49.5MB/s 
    [?25hRequirement already satisfied: sqlparse>=0.2.4 in /usr/local/lib/python3.6/dist-packages (from allennlp) (0.3.0)
    Requirement already satisfied: pytest in /usr/local/lib/python3.6/dist-packages (from allennlp) (3.6.4)
    Collecting parsimonious>=0.8.0 (from allennlp)
      Using cached https://files.pythonhosted.org/packages/02/fc/067a3f89869a41009e1a7cdfb14725f8ddd246f30f63c645e8ef8a1c56f4/parsimonious-0.8.1.tar.gz
    Collecting conllu==0.11 (from allennlp)
      Using cached https://files.pythonhosted.org/packages/d4/2c/856344d9b69baf5b374c395b4286626181a80f0c2b2f704914d18a1cea47/conllu-0.11-py2.py3-none-any.whl
    Collecting overrides (from allennlp)
      Downloading https://files.pythonhosted.org/packages/de/55/3100c6d14c1ed177492fcf8f07c4a7d2d6c996c0a7fc6a9a0a41308e7eec/overrides-1.9.tar.gz
    Collecting awscli>=1.11.91 (from allennlp)
    [?25l  Downloading https://files.pythonhosted.org/packages/20/fa/f4b6207d59267da0be60be3df32682d2c7479122c7cb87556bd4412675fe/awscli-1.16.190-py2.py3-none-any.whl (1.7MB)
    [K     |████████████████████████████████| 1.7MB 51.2MB/s 
    [?25hRequirement already satisfied: spacy<2.2,>=2.0.18 in /usr/local/lib/python3.6/dist-packages (from allennlp) (2.1.4)
    Collecting tensorboardX>=1.2 (from allennlp)
    [?25l  Downloading https://files.pythonhosted.org/packages/a2/57/2f0a46538295b8e7f09625da6dd24c23f9d0d7ef119ca1c33528660130d5/tensorboardX-1.7-py2.py3-none-any.whl (238kB)
    [K     |████████████████████████████████| 245kB 52.8MB/s 
    [?25hRequirement already satisfied: boto3 in /usr/local/lib/python3.6/dist-packages (from allennlp) (1.9.175)
    Requirement already satisfied: tqdm>=4.19 in /usr/local/lib/python3.6/dist-packages (from allennlp) (4.28.1)
    Requirement already satisfied: scipy in /usr/local/lib/python3.6/dist-packages (from allennlp) (1.3.0)
    Requirement already satisfied: requests>=2.18 in /usr/local/lib/python3.6/dist-packages (from allennlp) (2.21.0)
    Requirement already satisfied: h5py in /usr/local/lib/python3.6/dist-packages (from allennlp) (2.8.0)
    Requirement already satisfied: gevent>=1.3.6 in /usr/local/lib/python3.6/dist-packages (from allennlp) (1.4.0)
    Collecting jsonpickle (from allennlp)
      Downloading https://files.pythonhosted.org/packages/07/07/c157520a3ebd166c8c24c6ae0ecae7c3968eb4653ff0e5af369bb82f004d/jsonpickle-1.2-py2.py3-none-any.whl
    Collecting flask-cors>=3.0.7 (from allennlp)
      Downloading https://files.pythonhosted.org/packages/78/38/e68b11daa5d613e3a91e4bf3da76c94ac9ee0d9cd515af9c1ab80d36f709/Flask_Cors-3.0.8-py2.py3-none-any.whl
    Requirement already satisfied: numpy in /usr/local/lib/python3.6/dist-packages (from allennlp) (1.16.4)
    Requirement already satisfied: flask>=1.0.2 in /usr/local/lib/python3.6/dist-packages (from allennlp) (1.0.3)
    Requirement already satisfied: pytz>=2017.3 in /usr/local/lib/python3.6/dist-packages (from allennlp) (2018.9)
    Requirement already satisfied: nltk in /usr/local/lib/python3.6/dist-packages (from allennlp) (3.2.5)
    Collecting ftfy (from allennlp)
    [?25l  Downloading https://files.pythonhosted.org/packages/8f/86/df789c5834f15ae1ca53a8d4c1fc4788676c2e32112f6a786f2625d9c6e6/ftfy-5.5.1-py3-none-any.whl (43kB)
    [K     |████████████████████████████████| 51kB 26.0MB/s 
    [?25hRequirement already satisfied: editdistance in /usr/local/lib/python3.6/dist-packages (from allennlp) (0.5.3)
    Requirement already satisfied: torch>=0.4.1 in /usr/local/lib/python3.6/dist-packages (from allennlp) (1.1.0)
    Collecting word2number>=1.1 (from allennlp)
      Downloading https://files.pythonhosted.org/packages/4a/29/a31940c848521f0725f0df6b25dca8917f13a2025b0e8fcbe5d0457e45e6/word2number-1.1.zip
    Collecting responses>=0.7 (from allennlp)
      Using cached https://files.pythonhosted.org/packages/d1/5a/b887e89925f1de7890ef298a74438371ed4ed29b33def9e6d02dc6036fd8/responses-0.10.6-py2.py3-none-any.whl
    Requirement already satisfied: matplotlib>=2.2.3 in /usr/local/lib/python3.6/dist-packages (from allennlp) (3.0.3)
    Collecting flaky (from allennlp)
      Downloading https://files.pythonhosted.org/packages/ae/09/94d623dda1adacd51722f3e3e0f88ba08dd030ac2b2662bfb4383096340d/flaky-3.6.0-py2.py3-none-any.whl
    Requirement already satisfied: scikit-learn in /usr/local/lib/python3.6/dist-packages (from allennlp) (0.21.2)
    Collecting unidecode (from allennlp)
    [?25l  Downloading https://files.pythonhosted.org/packages/d0/42/d9edfed04228bacea2d824904cae367ee9efd05e6cce7ceaaedd0b0ad964/Unidecode-1.1.1-py2.py3-none-any.whl (238kB)
    [K     |████████████████████████████████| 245kB 27.3MB/s 
    [?25hRequirement already satisfied: sphinx>=1.6.5 in /usr/local/lib/python3.6/dist-packages (from numpydoc>=0.8.0->allennlp) (1.8.5)
    Requirement already satisfied: Jinja2>=2.3 in /usr/local/lib/python3.6/dist-packages (from numpydoc>=0.8.0->allennlp) (2.10.1)
    Collecting regex (from pytorch-pretrained-bert>=0.6.0->allennlp)
    [?25l  Downloading https://files.pythonhosted.org/packages/6f/4e/1b178c38c9a1a184288f72065a65ca01f3154df43c6ad898624149b8b4e0/regex-2019.06.08.tar.gz (651kB)
    [K     |████████████████████████████████| 655kB 50.1MB/s 
    [?25hRequirement already satisfied: six>=1.10.0 in /usr/local/lib/python3.6/dist-packages (from pytest->allennlp) (1.12.0)
    Requirement already satisfied: py>=1.5.0 in /usr/local/lib/python3.6/dist-packages (from pytest->allennlp) (1.8.0)
    Requirement already satisfied: more-itertools>=4.0.0 in /usr/local/lib/python3.6/dist-packages (from pytest->allennlp) (7.0.0)
    Requirement already satisfied: setuptools in /usr/local/lib/python3.6/dist-packages (from pytest->allennlp) (41.0.1)
    Requirement already satisfied: atomicwrites>=1.0 in /usr/local/lib/python3.6/dist-packages (from pytest->allennlp) (1.3.0)
    Requirement already satisfied: attrs>=17.4.0 in /usr/local/lib/python3.6/dist-packages (from pytest->allennlp) (19.1.0)
    Requirement already satisfied: pluggy<0.8,>=0.5 in /usr/local/lib/python3.6/dist-packages (from pytest->allennlp) (0.7.1)
    Requirement already satisfied: PyYAML<=5.1,>=3.10; python_version != "2.6" in /usr/local/lib/python3.6/dist-packages (from awscli>=1.11.91->allennlp) (3.13)
    Requirement already satisfied: s3transfer<0.3.0,>=0.2.0 in /usr/local/lib/python3.6/dist-packages (from awscli>=1.11.91->allennlp) (0.2.1)
    Collecting botocore==1.12.180 (from awscli>=1.11.91->allennlp)
    [?25l  Downloading https://files.pythonhosted.org/packages/3b/27/fa7da6feb20d1dfc0ab562226061b20da2d27ea18ca32dc764fe86704a99/botocore-1.12.180-py2.py3-none-any.whl (5.6MB)
    [K     |████████████████████████████████| 5.6MB 35.1MB/s 
    [?25hCollecting rsa<=3.5.0,>=3.1.2 (from awscli>=1.11.91->allennlp)
    [?25l  Downloading https://files.pythonhosted.org/packages/e1/ae/baedc9cb175552e95f3395c43055a6a5e125ae4d48a1d7a924baca83e92e/rsa-3.4.2-py2.py3-none-any.whl (46kB)
    [K     |████████████████████████████████| 51kB 26.7MB/s 
    [?25hRequirement already satisfied: docutils>=0.10 in /usr/local/lib/python3.6/dist-packages (from awscli>=1.11.91->allennlp) (0.14)
    Collecting colorama<=0.3.9,>=0.2.5 (from awscli>=1.11.91->allennlp)
      Downloading https://files.pythonhosted.org/packages/db/c8/7dcf9dbcb22429512708fe3a547f8b6101c0d02137acbd892505aee57adf/colorama-0.3.9-py2.py3-none-any.whl
    Requirement already satisfied: thinc<7.1.0,>=7.0.2 in /usr/local/lib/python3.6/dist-packages (from spacy<2.2,>=2.0.18->allennlp) (7.0.4)
    Requirement already satisfied: blis<0.3.0,>=0.2.2 in /usr/local/lib/python3.6/dist-packages (from spacy<2.2,>=2.0.18->allennlp) (0.2.4)
    Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /usr/local/lib/python3.6/dist-packages (from spacy<2.2,>=2.0.18->allennlp) (2.0.2)
    Requirement already satisfied: jsonschema<3.1.0,>=2.6.0 in /usr/local/lib/python3.6/dist-packages (from spacy<2.2,>=2.0.18->allennlp) (2.6.0)
    Requirement already satisfied: preshed<2.1.0,>=2.0.1 in /usr/local/lib/python3.6/dist-packages (from spacy<2.2,>=2.0.18->allennlp) (2.0.1)
    Requirement already satisfied: srsly<1.1.0,>=0.0.5 in /usr/local/lib/python3.6/dist-packages (from spacy<2.2,>=2.0.18->allennlp) (0.0.7)
    Requirement already satisfied: wasabi<1.1.0,>=0.2.0 in /usr/local/lib/python3.6/dist-packages (from spacy<2.2,>=2.0.18->allennlp) (0.2.2)
    Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /usr/local/lib/python3.6/dist-packages (from spacy<2.2,>=2.0.18->allennlp) (1.0.2)
    Requirement already satisfied: plac<1.0.0,>=0.9.6 in /usr/local/lib/python3.6/dist-packages (from spacy<2.2,>=2.0.18->allennlp) (0.9.6)
    Requirement already satisfied: protobuf>=3.2.0 in /usr/local/lib/python3.6/dist-packages (from tensorboardX>=1.2->allennlp) (3.7.1)
    Requirement already satisfied: jmespath<1.0.0,>=0.7.1 in /usr/local/lib/python3.6/dist-packages (from boto3->allennlp) (0.9.4)
    Requirement already satisfied: urllib3<1.25,>=1.21.1 in /usr/local/lib/python3.6/dist-packages (from requests>=2.18->allennlp) (1.24.3)
    Requirement already satisfied: idna<2.9,>=2.5 in /usr/local/lib/python3.6/dist-packages (from requests>=2.18->allennlp) (2.8)
    Requirement already satisfied: chardet<3.1.0,>=3.0.2 in /usr/local/lib/python3.6/dist-packages (from requests>=2.18->allennlp) (3.0.4)
    Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.6/dist-packages (from requests>=2.18->allennlp) (2019.6.16)
    Requirement already satisfied: greenlet>=0.4.14; platform_python_implementation == "CPython" in /usr/local/lib/python3.6/dist-packages (from gevent>=1.3.6->allennlp) (0.4.15)
    Requirement already satisfied: Werkzeug>=0.14 in /usr/local/lib/python3.6/dist-packages (from flask>=1.0.2->allennlp) (0.15.4)
    Requirement already satisfied: click>=5.1 in /usr/local/lib/python3.6/dist-packages (from flask>=1.0.2->allennlp) (7.0)
    Requirement already satisfied: itsdangerous>=0.24 in /usr/local/lib/python3.6/dist-packages (from flask>=1.0.2->allennlp) (1.1.0)
    Requirement already satisfied: wcwidth in /usr/local/lib/python3.6/dist-packages (from ftfy->allennlp) (0.1.7)
    Requirement already satisfied: kiwisolver>=1.0.1 in /usr/local/lib/python3.6/dist-packages (from matplotlib>=2.2.3->allennlp) (1.1.0)
    Requirement already satisfied: cycler>=0.10 in /usr/local/lib/python3.6/dist-packages (from matplotlib>=2.2.3->allennlp) (0.10.0)
    Requirement already satisfied: python-dateutil>=2.1 in /usr/local/lib/python3.6/dist-packages (from matplotlib>=2.2.3->allennlp) (2.5.3)
    Requirement already satisfied: pyparsing!=2.0.4,!=2.1.2,!=2.1.6,>=2.0.1 in /usr/local/lib/python3.6/dist-packages (from matplotlib>=2.2.3->allennlp) (2.4.0)
    Requirement already satisfied: joblib>=0.11 in /usr/local/lib/python3.6/dist-packages (from scikit-learn->allennlp) (0.13.2)
    Requirement already satisfied: Pygments>=2.0 in /usr/local/lib/python3.6/dist-packages (from sphinx>=1.6.5->numpydoc>=0.8.0->allennlp) (2.1.3)
    Requirement already satisfied: snowballstemmer>=1.1 in /usr/local/lib/python3.6/dist-packages (from sphinx>=1.6.5->numpydoc>=0.8.0->allennlp) (1.2.1)
    Requirement already satisfied: sphinxcontrib-websupport in /usr/local/lib/python3.6/dist-packages (from sphinx>=1.6.5->numpydoc>=0.8.0->allennlp) (1.1.2)
    Requirement already satisfied: packaging in /usr/local/lib/python3.6/dist-packages (from sphinx>=1.6.5->numpydoc>=0.8.0->allennlp) (19.0)
    Requirement already satisfied: imagesize in /usr/local/lib/python3.6/dist-packages (from sphinx>=1.6.5->numpydoc>=0.8.0->allennlp) (1.1.0)
    Requirement already satisfied: babel!=2.0,>=1.3 in /usr/local/lib/python3.6/dist-packages (from sphinx>=1.6.5->numpydoc>=0.8.0->allennlp) (2.7.0)
    Requirement already satisfied: alabaster<0.8,>=0.7 in /usr/local/lib/python3.6/dist-packages (from sphinx>=1.6.5->numpydoc>=0.8.0->allennlp) (0.7.12)
    Requirement already satisfied: MarkupSafe>=0.23 in /usr/local/lib/python3.6/dist-packages (from Jinja2>=2.3->numpydoc>=0.8.0->allennlp) (1.1.1)
    Requirement already satisfied: pyasn1>=0.1.3 in /usr/local/lib/python3.6/dist-packages (from rsa<=3.5.0,>=3.1.2->awscli>=1.11.91->allennlp) (0.4.5)
    Building wheels for collected packages: jsonnet, numpydoc, parsimonious, overrides, word2number, regex
      Building wheel for jsonnet (setup.py) ... [?25l[?25hdone
      Stored in directory: /root/.cache/pip/wheels/1a/30/ab/ae4a57b1df44fa20a531edb9601b27603da8f5336225691f3f
      Building wheel for numpydoc (setup.py) ... [?25l[?25hdone
      Stored in directory: /root/.cache/pip/wheels/51/30/d1/92a39ba40f21cb70e53f8af96eb98f002a781843c065406500
      Building wheel for parsimonious (setup.py) ... [?25l[?25hdone
      Stored in directory: /root/.cache/pip/wheels/b7/8d/e7/a0e74217da5caeb3c1c7689639b6d28ddbf9985b840bc96a9a
      Building wheel for overrides (setup.py) ... [?25l[?25hdone
      Stored in directory: /root/.cache/pip/wheels/8d/52/86/e5a83b1797e7d263b458d2334edd2704c78508b3eea9323718
      Building wheel for word2number (setup.py) ... [?25l[?25hdone
      Stored in directory: /root/.cache/pip/wheels/46/2f/53/5f5c1d275492f2fce1cdab9a9bb12d49286dead829a4078e0e
      Building wheel for regex (setup.py) ... [?25l[?25hdone
      Stored in directory: /root/.cache/pip/wheels/35/e4/80/abf3b33ba89cf65cd262af8a22a5a999cc28fbfabea6b38473
    Successfully built jsonnet numpydoc parsimonious overrides word2number regex
    Installing collected packages: jsonnet, numpydoc, regex, pytorch-pretrained-bert, parsimonious, conllu, overrides, botocore, rsa, colorama, awscli, tensorboardX, jsonpickle, flask-cors, ftfy, word2number, responses, flaky, unidecode, allennlp
      Found existing installation: botocore 1.12.175
        Uninstalling botocore-1.12.175:
          Successfully uninstalled botocore-1.12.175
      Found existing installation: rsa 4.0
        Uninstalling rsa-4.0:
          Successfully uninstalled rsa-4.0
    Successfully installed allennlp-0.8.4 awscli-1.16.190 botocore-1.12.180 colorama-0.3.9 conllu-0.11 flaky-3.6.0 flask-cors-3.0.8 ftfy-5.5.1 jsonnet-0.13.0 jsonpickle-1.2 numpydoc-0.9.1 overrides-1.9 parsimonious-0.8.1 pytorch-pretrained-bert-0.6.2 regex-2019.6.8 responses-0.10.6 rsa-3.4.2 tensorboardX-1.7 unidecode-1.1.1 word2number-1.1




### Approach

We will use mostly the same code as in our previous classification experiments.  For brevity, I have compacted it all here and omitted parts that arent required for this section.  For more information, see the previous section.


```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple
import os
import io
import re
import codecs
import numpy as np
from collections import Counter
from torch.utils.data import DataLoader, TensorDataset

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

### The new thing: set up to use ELMo


```python
from allennlp.modules.elmo import Elmo, batch_to_ids


def elmo_vectorizer(sentence):
    character_ids = batch_to_ids([sentence])
    return character_ids.squeeze(0)

  
class ElmoEmbedding(nn.Module):
    def __init__(self, options_file, weight_file, dropout=0.5):
        super().__init__()
        self.elmo = Elmo(options_file, weight_file, 2, dropout=dropout)
    def forward(self, xch):
        elmo = self.elmo(xch)
        e1, e2 = elmo['elmo_representations']
        mask = elmo['mask']
        embeddings = (e1 + e2) * mask.float().unsqueeze(-1)
        return embeddings

```

As before, we are going to load up our data with a reader.  This time, though, we will provide a vectorizer for ELMo.  In our simple example `Reader`, we only allow a single feature as our input vector to our classifier, so we can stop counting up our vocab.  In real life, you probably want to support both word vector features and context vector features so you might want to modify the code to support both.  This is a very common approach -- just using ELMo to augment an existing setup.  Here, we just look at using ELMo features by themselves.



```python
!wget https://www.dropbox.com/s/08km2ean8bkt7p3/trec.tar.gz?dl=1
!tar -xzf 'trec.tar.gz?dl=1'
```

    --2019-06-30 19:21:55--  https://www.dropbox.com/s/08km2ean8bkt7p3/trec.tar.gz?dl=1
    Resolving www.dropbox.com (www.dropbox.com)... 162.125.8.1, 2620:100:601b:1::a27d:801
    Connecting to www.dropbox.com (www.dropbox.com)|162.125.8.1|:443... connected.
    HTTP request sent, awaiting response... 301 Moved Permanently
    Location: /s/dl/08km2ean8bkt7p3/trec.tar.gz [following]
    --2019-06-30 19:21:56--  https://www.dropbox.com/s/dl/08km2ean8bkt7p3/trec.tar.gz
    Reusing existing connection to www.dropbox.com:443.
    HTTP request sent, awaiting response... 302 Found
    Location: https://uc7fa2ae1930db92d5916f06ba12.dl.dropboxusercontent.com/cd/0/get/Aj3XoF2sz7098a7ulJjBQP5DA6LkkkTQEAgFciDKPLgTZrHSUdejKQ7f8hkI3LiEt0BP_zf3LYg-ul8IZkevEcRCL4oxvYa8Uw-4SCn9GK2Lqw/file?dl=1# [following]
    --2019-06-30 19:21:56--  https://uc7fa2ae1930db92d5916f06ba12.dl.dropboxusercontent.com/cd/0/get/Aj3XoF2sz7098a7ulJjBQP5DA6LkkkTQEAgFciDKPLgTZrHSUdejKQ7f8hkI3LiEt0BP_zf3LYg-ul8IZkevEcRCL4oxvYa8Uw-4SCn9GK2Lqw/file?dl=1
    Resolving uc7fa2ae1930db92d5916f06ba12.dl.dropboxusercontent.com (uc7fa2ae1930db92d5916f06ba12.dl.dropboxusercontent.com)... 162.125.8.6, 2620:100:601b:6::a27d:806
    Connecting to uc7fa2ae1930db92d5916f06ba12.dl.dropboxusercontent.com (uc7fa2ae1930db92d5916f06ba12.dl.dropboxusercontent.com)|162.125.8.6|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 117253 (115K) [application/binary]
    Saving to: ‘trec.tar.gz?dl=1’
    
    trec.tar.gz?dl=1    100%[===================>] 114.50K  --.-KB/s    in 0.07s   
    
    2019-06-30 19:21:56 (1.71 MB/s) - ‘trec.tar.gz?dl=1’ saved [117253/117253]
    


We will set up our reader slightly differently than in the last experiment.  Here we will use an `elmo_vectorizer`


```python
BASE = 'trec'
TRAIN = os.path.join(BASE, 'trec.nodev.utf8')
VALID = os.path.join(BASE, 'trec.dev.utf8')
TEST = os.path.join(BASE, 'trec.test.utf8')



reader = Reader((TRAIN, VALID, TEST,), lowercase=False, vectorizer=elmo_vectorizer)
train = reader.load(TRAIN)
valid = reader.load(VALID)
test = reader.load(TEST)
```

    /usr/local/lib/python3.6/dist-packages/ipykernel_launcher.py:392: UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor).


Building the network is basically the same as before, but we are using ELMo instead of word vectors.  The command below will take a few minutes -- this is a much larger (forward) network than before, even though the learnable parameters havent really changed


```python
options_file = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
weight_file = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
embeddings = ElmoEmbedding(options_file, weight_file)
model = LSTMClassifier(embeddings, len(reader.labels), embed_dims=1024, rnn_units=100, hidden_units=[100])

num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Model has {num_params} parameters") 


model.to('cuda:0')
loss = torch.nn.NLLLoss()
loss = loss.to('cuda:0')

learnable_params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.Adadelta(learnable_params, lr=1.0)

fit(model, reader.labels, optimizer, loss, 12, 50, train, valid, test)
```

    100%|██████████| 336/336 [00:00<00:00, 192499.13B/s]
    100%|██████████| 374434792/374434792 [00:07<00:00, 47927932.74B/s]
    /usr/local/lib/python3.6/dist-packages/torch/nn/modules/rnn.py:54: UserWarning: dropout option adds dropout after all but last recurrent layer, so non-zero dropout expects num_layers greater than 1, but got dropout=0.5 and num_layers=1
      "num_layers={}".format(dropout, num_layers))


    Model has 461114 parameters
    EPOCH 1
    =================================
    Training Results
    {'acc': 0.5608, 'mean_precision': 0.6483439079531595, 'mean_recall': 0.48504498768062404, 'macro_f1': 0.49849106627634976, 'weighted_precision': 0.5726962123308302, 'weighted_recall': 0.5608, 'weighted_f1': 0.5554788741148295}
    Validation Results
    {'acc': 0.7942477876106194, 'mean_precision': 0.8454765420711672, 'mean_recall': 0.7033276693176708, 'macro_f1': 0.7144610587048233, 'weighted_precision': 0.8102895316760798, 'weighted_recall': 0.7942477876106194, 'weighted_f1': 0.7887777126414355}
    New best model 0.79
    EPOCH 2
    =================================
    Training Results
    {'acc': 0.806, 'mean_precision': 0.799350329535837, 'mean_recall': 0.7675872074813431, 'macro_f1': 0.780728542640896, 'weighted_precision': 0.8062829252605372, 'weighted_recall': 0.806, 'weighted_f1': 0.8058397968891035}
    Validation Results
    {'acc': 0.8628318584070797, 'mean_precision': 0.8566120843164245, 'mean_recall': 0.7974693543452065, 'macro_f1': 0.8182667932069821, 'weighted_precision': 0.8675313347987196, 'weighted_recall': 0.8628318584070797, 'weighted_f1': 0.8625189847178025}
    New best model 0.86
    EPOCH 3
    =================================
    Training Results
    {'acc': 0.8678, 'mean_precision': 0.8675015318855253, 'mean_recall': 0.8346532456259291, 'macro_f1': 0.8484927361816553, 'weighted_precision': 0.8682517001586247, 'weighted_recall': 0.8678, 'weighted_f1': 0.8677362764323896}
    Validation Results
    {'acc': 0.8451327433628318, 'mean_precision': 0.8284211573091326, 'mean_recall': 0.8093879960516328, 'macro_f1': 0.8110225138172149, 'weighted_precision': 0.8691115810447773, 'weighted_recall': 0.8451327433628318, 'weighted_f1': 0.8465397357783465}
    EPOCH 4
    =================================
    Training Results
    {'acc': 0.8872, 'mean_precision': 0.8764661421280517, 'mean_recall': 0.8546009991636673, 'macro_f1': 0.8643704500888516, 'weighted_precision': 0.887561002584866, 'weighted_recall': 0.8872, 'weighted_f1': 0.8872276932804481}
    Validation Results
    {'acc': 0.911504424778761, 'mean_precision': 0.8515226408802844, 'mean_recall': 0.8617077224611437, 'macro_f1': 0.8561887828467433, 'weighted_precision': 0.9122749445064818, 'weighted_recall': 0.911504424778761, 'weighted_f1': 0.9118158975632823}
    New best model 0.91
    EPOCH 5
    =================================
    Training Results
    {'acc': 0.9034, 'mean_precision': 0.9068352283292169, 'mean_recall': 0.8843802250756597, 'macro_f1': 0.8946296708241798, 'weighted_precision': 0.9040643245149811, 'weighted_recall': 0.9034, 'weighted_f1': 0.9035884797279896}
    Validation Results
    {'acc': 0.8871681415929203, 'mean_precision': 0.8310659320074388, 'mean_recall': 0.841863153832931, 'macro_f1': 0.8355145420604436, 'weighted_precision': 0.8885588116644558, 'weighted_recall': 0.8871681415929203, 'weighted_f1': 0.8871217953267708}
    EPOCH 6
    =================================
    Training Results
    {'acc': 0.9136, 'mean_precision': 0.9192746333288291, 'mean_recall': 0.8914669258673943, 'macro_f1': 0.903828395837297, 'weighted_precision': 0.9139512391702285, 'weighted_recall': 0.9136, 'weighted_f1': 0.913614469191629}
    Validation Results
    {'acc': 0.9048672566371682, 'mean_precision': 0.8453363940567148, 'mean_recall': 0.8564313119872883, 'macro_f1': 0.8503405229048734, 'weighted_precision': 0.905338925288292, 'weighted_recall': 0.9048672566371682, 'weighted_f1': 0.9048873521303485}
    EPOCH 7
    =================================
    Training Results
    {'acc': 0.9184, 'mean_precision': 0.9217954417236368, 'mean_recall': 0.9035341951741954, 'macro_f1': 0.9119837710405331, 'weighted_precision': 0.9188094085046438, 'weighted_recall': 0.9184, 'weighted_f1': 0.9185065760698944}
    Validation Results
    {'acc': 0.9004424778761062, 'mean_precision': 0.8343728710441182, 'mean_recall': 0.8542340405568197, 'macro_f1': 0.8426629413676556, 'weighted_precision': 0.9019737446759757, 'weighted_recall': 0.9004424778761062, 'weighted_f1': 0.9006848775343249}
    EPOCH 8
    =================================
    Training Results
    {'acc': 0.9252, 'mean_precision': 0.9227662229391251, 'mean_recall': 0.9085845822017588, 'macro_f1': 0.9152560555320276, 'weighted_precision': 0.9254505663098069, 'weighted_recall': 0.9252, 'weighted_f1': 0.9252609572329403}
    Validation Results
    {'acc': 0.8960176991150443, 'mean_precision': 0.8848359324236518, 'mean_recall': 0.8515305594157283, 'macro_f1': 0.8641410893717477, 'weighted_precision': 0.897474904298379, 'weighted_recall': 0.8960176991150443, 'weighted_f1': 0.8954448264468791}
    EPOCH 9
    =================================
    Training Results
    {'acc': 0.9366, 'mean_precision': 0.9421415595699045, 'mean_recall': 0.9253828413493465, 'macro_f1': 0.9332020129586184, 'weighted_precision': 0.9367741614764589, 'weighted_recall': 0.9366, 'weighted_f1': 0.9366203849323997}
    Validation Results
    {'acc': 0.9004424778761062, 'mean_precision': 0.8408851907016573, 'mean_recall': 0.8542708251432938, 'macro_f1': 0.8466559111080202, 'weighted_precision': 0.9022774132643538, 'weighted_recall': 0.9004424778761062, 'weighted_f1': 0.9006261595204735}
    EPOCH 10
    =================================
    Training Results
    {'acc': 0.9422, 'mean_precision': 0.9415872377873563, 'mean_recall': 0.9301100255239593, 'macro_f1': 0.9356066415360083, 'weighted_precision': 0.9423787344008276, 'weighted_recall': 0.9422, 'weighted_f1': 0.9422531801175381}
    Validation Results
    {'acc': 0.9026548672566371, 'mean_precision': 0.8534388800712419, 'mean_recall': 0.855449985872144, 'macro_f1': 0.8538969412521858, 'weighted_precision': 0.9037659180936365, 'weighted_recall': 0.9026548672566371, 'weighted_f1': 0.90246529999771}
    EPOCH 11
    =================================
    Training Results
    {'acc': 0.9432, 'mean_precision': 0.9422754608090832, 'mean_recall': 0.938376139581592, 'macro_f1': 0.9402970803722553, 'weighted_precision': 0.9432734858574917, 'weighted_recall': 0.9432, 'weighted_f1': 0.943229377825017}
    Validation Results
    {'acc': 0.9137168141592921, 'mean_precision': 0.8628400105220431, 'mean_recall': 0.8646482805732667, 'macro_f1': 0.8633776502808389, 'weighted_precision': 0.9132203845237589, 'weighted_recall': 0.9137168141592921, 'weighted_f1': 0.9130050592497775}
    New best model 0.91
    EPOCH 12
    =================================
    Training Results
    {'acc': 0.9544, 'mean_precision': 0.9557163129978826, 'mean_recall': 0.9458500359607124, 'macro_f1': 0.9506063779628039, 'weighted_precision': 0.9545506681185594, 'weighted_recall': 0.9544, 'weighted_f1': 0.9544423559597639}
    Validation Results
    {'acc': 0.9092920353982301, 'mean_precision': 0.8510768742634296, 'mean_recall': 0.8608961905116529, 'macro_f1': 0.8550990513587272, 'weighted_precision': 0.9106944939486582, 'weighted_recall': 0.9092920353982301, 'weighted_f1': 0.9093039549088799}
    Final result
    {'acc': 0.944, 'mean_precision': 0.9333687372820768, 'mean_recall': 0.9161547629123813, 'macro_f1': 0.9230157805001022, 'weighted_precision': 0.9449538854974426, 'weighted_recall': 0.944, 'weighted_f1': 0.9429751846143404}





    0.944



Let's see how this number compares against a randomly initialized baseline model that is otherwise identical.  We dont really need to use such a huge embedding size in this case -- we are using word vectors instead of character compositional vectors and we dont really have enough information to train a huge word embedding from scratch.  Also, since we dont have much information, we will use lowercased features.  Note that using these word embeddings features, our model has **6x more parameters than before**.  Also, we might want to train it longer.


```python

r = Reader((TRAIN, VALID, TEST,), lowercase=True)
train = r.load(TRAIN)
valid = r.load(VALID)
test = r.load(TEST)

embeddings = nn.Embedding(len(r.vocab), 300)
model = LSTMClassifier(embeddings, len(r.labels), embeddings.weight.shape[1], rnn_units=100, hidden_units=[100])

num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Model has {num_params} parameters") 


model.to('cuda:0')
loss = torch.nn.NLLLoss()
loss = loss.to('cuda:0')

learnable_params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.Adadelta(learnable_params, lr=1.0)

fit(model, r.labels, optimizer, loss, 48, 50, train, valid, test)
```

    /usr/local/lib/python3.6/dist-packages/torch/nn/modules/rnn.py:54: UserWarning: dropout option adds dropout after all but last recurrent layer, so non-zero dropout expects num_layers greater than 1, but got dropout=0.5 and num_layers=1
      "num_layers={}".format(dropout, num_layers))


    Model has 2801306 parameters
    EPOCH 1
    =================================
    Training Results
    {'acc': 0.2542, 'mean_precision': 0.31261376969825, 'mean_recall': 0.2142077519956256, 'macro_f1': 0.21100023435767312, 'weighted_precision': 0.24503296037652494, 'weighted_recall': 0.2542, 'weighted_f1': 0.22759490197394544}
    Validation Results
    {'acc': 0.31194690265486724, 'mean_precision': 0.30219607432329726, 'mean_recall': 0.2697651533307747, 'macro_f1': 0.23033350512330675, 'weighted_precision': 0.3308620779538906, 'weighted_recall': 0.31194690265486724, 'weighted_f1': 0.25984647084205925}
    New best model 0.31
    EPOCH 2
    =================================
    Training Results
    {'acc': 0.3776, 'mean_precision': 0.4211727454667504, 'mean_recall': 0.3614489295644783, 'macro_f1': 0.3731848509532312, 'weighted_precision': 0.37235350929169814, 'weighted_recall': 0.3776, 'weighted_f1': 0.3631077775707718}
    Validation Results
    {'acc': 0.4557522123893805, 'mean_precision': 0.486368442230124, 'mean_recall': 0.44059437145396335, 'macro_f1': 0.4116212305386946, 'weighted_precision': 0.5028626993515615, 'weighted_recall': 0.4557522123893805, 'weighted_f1': 0.42325171491881114}
    New best model 0.46
    EPOCH 3
    =================================
    Training Results
    {'acc': 0.5432, 'mean_precision': 0.5874298893317987, 'mean_recall': 0.5213749406751867, 'macro_f1': 0.5432030532887705, 'weighted_precision': 0.5414128746603742, 'weighted_recall': 0.5432, 'weighted_f1': 0.539731134218227}
    Validation Results
    {'acc': 0.6592920353982301, 'mean_precision': 0.6419746068058818, 'mean_recall': 0.6268845865056035, 'macro_f1': 0.6307977552497951, 'weighted_precision': 0.662392825769239, 'weighted_recall': 0.6592920353982301, 'weighted_f1': 0.6571270878109032}
    New best model 0.66
    EPOCH 4
    =================================
    Training Results
    {'acc': 0.6652, 'mean_precision': 0.7019592671173288, 'mean_recall': 0.6360073323821094, 'macro_f1': 0.6596012872763003, 'weighted_precision': 0.6699451752521243, 'weighted_recall': 0.6652, 'weighted_f1': 0.6663461696747198}
    Validation Results
    {'acc': 0.7256637168141593, 'mean_precision': 0.6947344026048728, 'mean_recall': 0.684312146723145, 'macro_f1': 0.6882240153205471, 'weighted_precision': 0.7329104731535425, 'weighted_recall': 0.7256637168141593, 'weighted_f1': 0.7276818270316268}
    New best model 0.73
    EPOCH 5
    =================================
    Training Results
    {'acc': 0.7262, 'mean_precision': 0.7559759521433286, 'mean_recall': 0.6964030168119725, 'macro_f1': 0.718961419314074, 'weighted_precision': 0.7314068173369025, 'weighted_recall': 0.7262, 'weighted_f1': 0.7276772804705455}
    Validation Results
    {'acc': 0.7699115044247787, 'mean_precision': 0.767162729738685, 'mean_recall': 0.7171683804327849, 'macro_f1': 0.7358018640759697, 'weighted_precision': 0.7869035814233268, 'weighted_recall': 0.7699115044247787, 'weighted_f1': 0.7730466432148493}
    New best model 0.77
    EPOCH 6
    =================================
    Training Results
    {'acc': 0.7774, 'mean_precision': 0.7991829240246574, 'mean_recall': 0.7460179937770648, 'macro_f1': 0.7670813561197591, 'weighted_precision': 0.7829082373273452, 'weighted_recall': 0.7774, 'weighted_f1': 0.7790066442175066}
    Validation Results
    {'acc': 0.7942477876106194, 'mean_precision': 0.7722010233604815, 'mean_recall': 0.7413137565797516, 'macro_f1': 0.7527105440234411, 'weighted_precision': 0.7965885015875331, 'weighted_recall': 0.7942477876106194, 'weighted_f1': 0.7932362973950707}
    New best model 0.79
    EPOCH 7
    =================================
    Training Results
    {'acc': 0.8094, 'mean_precision': 0.8190739759778141, 'mean_recall': 0.7893403911783444, 'macro_f1': 0.802535501291079, 'weighted_precision': 0.8128192118492741, 'weighted_recall': 0.8094, 'weighted_f1': 0.8105827362652248}
    Validation Results
    {'acc': 0.7876106194690266, 'mean_precision': 0.7485831299937852, 'mean_recall': 0.737125918471548, 'macro_f1': 0.739828985707053, 'weighted_precision': 0.7889226923611903, 'weighted_recall': 0.7876106194690266, 'weighted_f1': 0.7851833379237152}
    EPOCH 8
    =================================
    Training Results
    {'acc': 0.8334, 'mean_precision': 0.8425566386344959, 'mean_recall': 0.8020487930554469, 'macro_f1': 0.8188927928986053, 'weighted_precision': 0.8361427439552401, 'weighted_recall': 0.8334, 'weighted_f1': 0.8341487335405394}
    Validation Results
    {'acc': 0.8053097345132744, 'mean_precision': 0.7836886850840924, 'mean_recall': 0.7717037075656982, 'macro_f1': 0.7766933663665941, 'weighted_precision': 0.8124743795936054, 'weighted_recall': 0.8053097345132744, 'weighted_f1': 0.8076258267664438}
    New best model 0.81
    EPOCH 9
    =================================
    Training Results
    {'acc': 0.8468, 'mean_precision': 0.8504311729666333, 'mean_recall': 0.8200134573572487, 'macro_f1': 0.8331837029506359, 'weighted_precision': 0.8494250195065668, 'weighted_recall': 0.8468, 'weighted_f1': 0.8475914482379496}
    Validation Results
    {'acc': 0.8163716814159292, 'mean_precision': 0.8122787276154813, 'mean_recall': 0.7728318623385028, 'macro_f1': 0.7845697172233551, 'weighted_precision': 0.841008158316879, 'weighted_recall': 0.8163716814159292, 'weighted_f1': 0.8194346806526578}
    New best model 0.82
    EPOCH 10
    =================================
    Training Results
    {'acc': 0.856, 'mean_precision': 0.8583456863001512, 'mean_recall': 0.8344367321617767, 'macro_f1': 0.8452794073478773, 'weighted_precision': 0.858502410730061, 'weighted_recall': 0.856, 'weighted_f1': 0.8568463379552975}
    Validation Results
    {'acc': 0.827433628318584, 'mean_precision': 0.8065578820468894, 'mean_recall': 0.7875269000666668, 'macro_f1': 0.7948885464239536, 'weighted_precision': 0.8376851538502355, 'weighted_recall': 0.827433628318584, 'weighted_f1': 0.8300346063518604}
    New best model 0.83
    EPOCH 11
    =================================
    Training Results
    {'acc': 0.8752, 'mean_precision': 0.8867543443057883, 'mean_recall': 0.8682828032176982, 'macro_f1': 0.876877938560822, 'weighted_precision': 0.8765970442190595, 'weighted_recall': 0.8752, 'weighted_f1': 0.8756987373269298}
    Validation Results
    {'acc': 0.8429203539823009, 'mean_precision': 0.810890350645565, 'mean_recall': 0.8200588537595558, 'macro_f1': 0.8091225485091558, 'weighted_precision': 0.856571882711008, 'weighted_recall': 0.8429203539823009, 'weighted_f1': 0.8457428127083414}
    New best model 0.84
    EPOCH 12
    =================================
    Training Results
    {'acc': 0.8792, 'mean_precision': 0.8792557729505756, 'mean_recall': 0.8559171118516193, 'macro_f1': 0.8664273096431591, 'weighted_precision': 0.8802943928708451, 'weighted_recall': 0.8792, 'weighted_f1': 0.8794929463166824}
    Validation Results
    {'acc': 0.831858407079646, 'mean_precision': 0.7962677456184699, 'mean_recall': 0.8169955255150682, 'macro_f1': 0.8012212821990726, 'weighted_precision': 0.8439342848824284, 'weighted_recall': 0.831858407079646, 'weighted_f1': 0.8345906017116217}
    EPOCH 13
    =================================
    Training Results
    {'acc': 0.8982, 'mean_precision': 0.8926979300525084, 'mean_recall': 0.8720075500179675, 'macro_f1': 0.8813865977662623, 'weighted_precision': 0.8990274368829654, 'weighted_recall': 0.8982, 'weighted_f1': 0.898400074877493}
    Validation Results
    {'acc': 0.8407079646017699, 'mean_precision': 0.8044426437483548, 'mean_recall': 0.8230926174040855, 'macro_f1': 0.8115058510891844, 'weighted_precision': 0.8446897114706726, 'weighted_recall': 0.8407079646017699, 'weighted_f1': 0.8418088284238727}
    EPOCH 14
    =================================
    Training Results
    {'acc': 0.8994, 'mean_precision': 0.90397473118525, 'mean_recall': 0.8869886283764901, 'macro_f1': 0.8949155550364044, 'weighted_precision': 0.9000862304408612, 'weighted_recall': 0.8994, 'weighted_f1': 0.8996270341701729}
    Validation Results
    {'acc': 0.834070796460177, 'mean_precision': 0.7941534082526428, 'mean_recall': 0.8181802560779231, 'macro_f1': 0.802448595927142, 'weighted_precision': 0.839962405943381, 'weighted_recall': 0.834070796460177, 'weighted_f1': 0.8360121837053236}
    EPOCH 15
    =================================
    Training Results
    {'acc': 0.9078, 'mean_precision': 0.9025906984450397, 'mean_recall': 0.8840592072689094, 'macro_f1': 0.8925740985975588, 'weighted_precision': 0.9079516195367331, 'weighted_recall': 0.9078, 'weighted_f1': 0.9077917625402113}
    Validation Results
    {'acc': 0.8429203539823009, 'mean_precision': 0.8178604107482675, 'mean_recall': 0.8027279926613361, 'macro_f1': 0.807847400976306, 'weighted_precision': 0.8541454685154969, 'weighted_recall': 0.8429203539823009, 'weighted_f1': 0.8452297642709833}
    EPOCH 16
    =================================
    Training Results
    {'acc': 0.9114, 'mean_precision': 0.9091156125381755, 'mean_recall': 0.8954153246439646, 'macro_f1': 0.901884018226399, 'weighted_precision': 0.9120534355149411, 'weighted_recall': 0.9114, 'weighted_f1': 0.9115938328769627}
    Validation Results
    {'acc': 0.8584070796460177, 'mean_precision': 0.8452925879669096, 'mean_recall': 0.8140923455585375, 'macro_f1': 0.8271059195675882, 'weighted_precision': 0.866784367505974, 'weighted_recall': 0.8584070796460177, 'weighted_f1': 0.8601283537102449}
    New best model 0.86
    EPOCH 17
    =================================
    Training Results
    {'acc': 0.922, 'mean_precision': 0.92318827916946, 'mean_recall': 0.9167872228173993, 'macro_f1': 0.919907344495105, 'weighted_precision': 0.9222631334663767, 'weighted_recall': 0.922, 'weighted_f1': 0.9220983595773123}
    Validation Results
    {'acc': 0.8495575221238938, 'mean_precision': 0.8226381745596281, 'mean_recall': 0.8064587339697171, 'macro_f1': 0.8118331344786779, 'weighted_precision': 0.8588090104120005, 'weighted_recall': 0.8495575221238938, 'weighted_f1': 0.8509990153606489}
    EPOCH 18
    =================================
    Training Results
    {'acc': 0.925, 'mean_precision': 0.9272713764228396, 'mean_recall': 0.9097196644145997, 'macro_f1': 0.9178903031944213, 'weighted_precision': 0.9256665310570704, 'weighted_recall': 0.925, 'weighted_f1': 0.9251760923127654}
    Validation Results
    {'acc': 0.8362831858407079, 'mean_precision': 0.8130991700945112, 'mean_recall': 0.8186032795814143, 'macro_f1': 0.8123228744939271, 'weighted_precision': 0.8468684662820305, 'weighted_recall': 0.8362831858407079, 'weighted_f1': 0.8392648607279487}
    EPOCH 19
    =================================
    Training Results
    {'acc': 0.9226, 'mean_precision': 0.9265697322783559, 'mean_recall': 0.907365970956147, 'macro_f1': 0.9162751616718158, 'weighted_precision': 0.9229974594112873, 'weighted_recall': 0.9226, 'weighted_f1': 0.922695479769021}
    Validation Results
    {'acc': 0.8407079646017699, 'mean_precision': 0.7961368682179472, 'mean_recall': 0.8225977371214205, 'macro_f1': 0.8033500248964449, 'weighted_precision': 0.8499048273885906, 'weighted_recall': 0.8407079646017699, 'weighted_f1': 0.8433057108007154}
    EPOCH 20
    =================================
    Training Results
    {'acc': 0.9306, 'mean_precision': 0.9325672078372992, 'mean_recall': 0.9292912885334105, 'macro_f1': 0.9308909230234336, 'weighted_precision': 0.9309373491584835, 'weighted_recall': 0.9306, 'weighted_f1': 0.9307301028725288}
    Validation Results
    {'acc': 0.8451327433628318, 'mean_precision': 0.8169164169164169, 'mean_recall': 0.8240567649306224, 'macro_f1': 0.81237323931053, 'weighted_precision': 0.8649006598121644, 'weighted_recall': 0.8451327433628318, 'weighted_f1': 0.8481262461147536}
    EPOCH 21
    =================================
    Training Results
    {'acc': 0.9372, 'mean_precision': 0.94778588624423, 'mean_recall': 0.931739382135247, 'macro_f1': 0.9392845360347503, 'weighted_precision': 0.9376509979046739, 'weighted_recall': 0.9372, 'weighted_f1': 0.9373365710922599}
    Validation Results
    {'acc': 0.8429203539823009, 'mean_precision': 0.8122060187568131, 'mean_recall': 0.8039074327168304, 'macro_f1': 0.8074842893609094, 'weighted_precision': 0.846085726902366, 'weighted_recall': 0.8429203539823009, 'weighted_f1': 0.843907236330442}
    EPOCH 22
    =================================
    Training Results
    {'acc': 0.9396, 'mean_precision': 0.9374128845763084, 'mean_recall': 0.919864102313256, 'macro_f1': 0.9279841844104714, 'weighted_precision': 0.9395460389520726, 'weighted_recall': 0.9396, 'weighted_f1': 0.9395077256432057}
    Validation Results
    {'acc': 0.838495575221239, 'mean_precision': 0.8166287688346512, 'mean_recall': 0.7965319356459588, 'macro_f1': 0.8036210071046136, 'weighted_precision': 0.8514493271781142, 'weighted_recall': 0.838495575221239, 'weighted_f1': 0.8411121771904451}
    EPOCH 23
    =================================
    Training Results
    {'acc': 0.9394, 'mean_precision': 0.9343864652716637, 'mean_recall': 0.9292029606780483, 'macro_f1': 0.9317217171943124, 'weighted_precision': 0.9395732746438947, 'weighted_recall': 0.9394, 'weighted_f1': 0.9394587072607552}
    Validation Results
    {'acc': 0.8495575221238938, 'mean_precision': 0.8152645128671007, 'mean_recall': 0.8101445689658334, 'macro_f1': 0.8116836837706645, 'weighted_precision': 0.853045900832462, 'weighted_recall': 0.8495575221238938, 'weighted_f1': 0.8499444374703017}
    EPOCH 24
    =================================
    Training Results
    {'acc': 0.9366, 'mean_precision': 0.9385826391144599, 'mean_recall': 0.9310259128633419, 'macro_f1': 0.9346948428938683, 'weighted_precision': 0.9367440103332517, 'weighted_recall': 0.9366, 'weighted_f1': 0.9366526615949582}
    Validation Results
    {'acc': 0.8473451327433629, 'mean_precision': 0.8025432410455261, 'mean_recall': 0.8283322432914666, 'macro_f1': 0.8061102573975748, 'weighted_precision': 0.861458116214064, 'weighted_recall': 0.8473451327433629, 'weighted_f1': 0.8502841373610759}
    EPOCH 25
    =================================
    Training Results
    {'acc': 0.9474, 'mean_precision': 0.945203027270548, 'mean_recall': 0.9397940805142109, 'macro_f1': 0.9424193517790628, 'weighted_precision': 0.9476039831561757, 'weighted_recall': 0.9474, 'weighted_f1': 0.9474674977170525}
    Validation Results
    {'acc': 0.8517699115044248, 'mean_precision': 0.8122309943824346, 'mean_recall': 0.8316675962171595, 'macro_f1': 0.817457525740406, 'weighted_precision': 0.8589009917207507, 'weighted_recall': 0.8517699115044248, 'weighted_f1': 0.8536305273856584}
    EPOCH 26
    =================================
    Training Results
    {'acc': 0.9412, 'mean_precision': 0.9333724069841898, 'mean_recall': 0.9327119878566131, 'macro_f1': 0.9330083105008247, 'weighted_precision': 0.9415099002878724, 'weighted_recall': 0.9412, 'weighted_f1': 0.9413101950625798}
    Validation Results
    {'acc': 0.8672566371681416, 'mean_precision': 0.8605083530628996, 'mean_recall': 0.8246033707390573, 'macro_f1': 0.8391902596303691, 'weighted_precision': 0.8669141290171042, 'weighted_recall': 0.8672566371681416, 'weighted_f1': 0.8665873995194234}
    New best model 0.87
    EPOCH 27
    =================================
    Training Results
    {'acc': 0.9452, 'mean_precision': 0.9393570838696085, 'mean_recall': 0.9381922191949007, 'macro_f1': 0.938758641015847, 'weighted_precision': 0.9451643947972791, 'weighted_recall': 0.9452, 'weighted_f1': 0.9451693027949712}
    Validation Results
    {'acc': 0.8539823008849557, 'mean_precision': 0.8230172208866048, 'mean_recall': 0.8123173810979951, 'macro_f1': 0.8156989547159269, 'weighted_precision': 0.86038962724662, 'weighted_recall': 0.8539823008849557, 'weighted_f1': 0.8546422455744229}
    EPOCH 28
    =================================
    Training Results
    {'acc': 0.9496, 'mean_precision': 0.9460303491056411, 'mean_recall': 0.9315008817259457, 'macro_f1': 0.938347207012888, 'weighted_precision': 0.9496092730822427, 'weighted_recall': 0.9496, 'weighted_f1': 0.9495572903768849}
    Validation Results
    {'acc': 0.8495575221238938, 'mean_precision': 0.8264021292481143, 'mean_recall': 0.8075150926865313, 'macro_f1': 0.8142132100384322, 'weighted_precision': 0.8614722274611524, 'weighted_recall': 0.8495575221238938, 'weighted_f1': 0.8519763491364317}
    EPOCH 29
    =================================
    Training Results
    {'acc': 0.9492, 'mean_precision': 0.9523110976436452, 'mean_recall': 0.9431909498306, 'macro_f1': 0.9475782265270856, 'weighted_precision': 0.9492766331969764, 'weighted_recall': 0.9492, 'weighted_f1': 0.9492146339395731}
    Validation Results
    {'acc': 0.8517699115044248, 'mean_precision': 0.8258343506751618, 'mean_recall': 0.8103395389633586, 'macro_f1': 0.8158066635203088, 'weighted_precision': 0.860982906249134, 'weighted_recall': 0.8517699115044248, 'weighted_f1': 0.8534222923294074}
    EPOCH 30
    =================================
    Training Results
    {'acc': 0.9486, 'mean_precision': 0.9560017443521739, 'mean_recall': 0.9400692468871957, 'macro_f1': 0.9475746328501561, 'weighted_precision': 0.9488545438754485, 'weighted_recall': 0.9486, 'weighted_f1': 0.9486620015461529}
    Validation Results
    {'acc': 0.8517699115044248, 'mean_precision': 0.8177916897275509, 'mean_recall': 0.8317336038356844, 'macro_f1': 0.8189379757266612, 'weighted_precision': 0.8641615919629556, 'weighted_recall': 0.8517699115044248, 'weighted_f1': 0.8543175095648456}
    EPOCH 31
    =================================
    Training Results
    {'acc': 0.958, 'mean_precision': 0.9538284266182177, 'mean_recall': 0.9446486493014933, 'macro_f1': 0.9490660805682173, 'weighted_precision': 0.9581057645690427, 'weighted_recall': 0.958, 'weighted_f1': 0.958018653218701}
    Validation Results
    {'acc': 0.8517699115044248, 'mean_precision': 0.8013986378685444, 'mean_recall': 0.8119935701678367, 'macro_f1': 0.804411328159357, 'weighted_precision': 0.8579675339835053, 'weighted_recall': 0.8517699115044248, 'weighted_f1': 0.8534679803361881}
    EPOCH 32
    =================================
    Training Results
    {'acc': 0.954, 'mean_precision': 0.9512729471583152, 'mean_recall': 0.9522326430845925, 'macro_f1': 0.9517306869942752, 'weighted_precision': 0.9541486554643743, 'weighted_recall': 0.954, 'weighted_f1': 0.9540543038088639}
    Validation Results
    {'acc': 0.8584070796460177, 'mean_precision': 0.8247875724959964, 'mean_recall': 0.813382140867244, 'macro_f1': 0.8148679951657951, 'weighted_precision': 0.8728418185739959, 'weighted_recall': 0.8584070796460177, 'weighted_f1': 0.8607256592741758}
    EPOCH 33
    =================================
    Training Results
    {'acc': 0.957, 'mean_precision': 0.9565681480304719, 'mean_recall': 0.9454327046857133, 'macro_f1': 0.9507466808370677, 'weighted_precision': 0.9571322707135252, 'weighted_recall': 0.957, 'weighted_f1': 0.9570200735546852}
    Validation Results
    {'acc': 0.8407079646017699, 'mean_precision': 0.8012295177369267, 'mean_recall': 0.7988123932835594, 'macro_f1': 0.7956541104094256, 'weighted_precision': 0.854940819905837, 'weighted_recall': 0.8407079646017699, 'weighted_f1': 0.843834784432942}
    EPOCH 34
    =================================
    Training Results
    {'acc': 0.9562, 'mean_precision': 0.9492352170638396, 'mean_recall': 0.9501709445363726, 'macro_f1': 0.9496843838684074, 'weighted_precision': 0.9563024102937812, 'weighted_recall': 0.9562, 'weighted_f1': 0.9562374926155631}
    Validation Results
    {'acc': 0.8539823008849557, 'mean_precision': 0.8074247463353991, 'mean_recall': 0.8132105098252905, 'macro_f1': 0.8095076634215594, 'weighted_precision': 0.8571530973672192, 'weighted_recall': 0.8539823008849557, 'weighted_f1': 0.8549969022822371}
    EPOCH 35
    =================================
    Training Results
    {'acc': 0.958, 'mean_precision': 0.9581863087122168, 'mean_recall': 0.9556184908351323, 'macro_f1': 0.9568856509156166, 'weighted_precision': 0.9581083670874939, 'weighted_recall': 0.958, 'weighted_f1': 0.9580411865238977}
    Validation Results
    {'acc': 0.8495575221238938, 'mean_precision': 0.8178061479593234, 'mean_recall': 0.805632013633958, 'macro_f1': 0.807789691127139, 'weighted_precision': 0.8649796275543147, 'weighted_recall': 0.8495575221238938, 'weighted_f1': 0.8526672381640437}
    EPOCH 36
    =================================
    Training Results
    {'acc': 0.9574, 'mean_precision': 0.9531272685674982, 'mean_recall': 0.9515769491207995, 'macro_f1': 0.9523405767311283, 'weighted_precision': 0.9573866537776834, 'weighted_recall': 0.9574, 'weighted_f1': 0.9573873211612607}
    Validation Results
    {'acc': 0.8561946902654868, 'mean_precision': 0.8131151427942895, 'mean_recall': 0.8110658945033787, 'macro_f1': 0.8077795032214256, 'weighted_precision': 0.8693120626269972, 'weighted_recall': 0.8561946902654868, 'weighted_f1': 0.8589532280938094}
    EPOCH 37
    =================================
    Training Results
    {'acc': 0.9566, 'mean_precision': 0.9589041860848261, 'mean_recall': 0.9547448481946009, 'macro_f1': 0.9567909158741493, 'weighted_precision': 0.9566010555107037, 'weighted_recall': 0.9566, 'weighted_f1': 0.9565890848568762}
    Validation Results
    {'acc': 0.8495575221238938, 'mean_precision': 0.8030750892649611, 'mean_recall': 0.8075017730562345, 'macro_f1': 0.8022990697875542, 'weighted_precision': 0.858293669895924, 'weighted_recall': 0.8495575221238938, 'weighted_f1': 0.851768488915397}
    EPOCH 38
    =================================
    Training Results
    {'acc': 0.957, 'mean_precision': 0.9544796723873579, 'mean_recall': 0.9522358611983669, 'macro_f1': 0.9533352482886838, 'weighted_precision': 0.9571066919837921, 'weighted_recall': 0.957, 'weighted_f1': 0.9570317579959188}
    Validation Results
    {'acc': 0.8539823008849557, 'mean_precision': 0.8106297488632482, 'mean_recall': 0.8123418403963019, 'macro_f1': 0.8098548414705528, 'weighted_precision': 0.8585169985936104, 'weighted_recall': 0.8539823008849557, 'weighted_f1': 0.8548092957757126}
    EPOCH 39
    =================================
    Training Results
    {'acc': 0.9564, 'mean_precision': 0.9532129259082603, 'mean_recall': 0.9506704508751782, 'macro_f1': 0.951926709753589, 'weighted_precision': 0.9564913201918928, 'weighted_recall': 0.9564, 'weighted_f1': 0.9564356136646421}
    Validation Results
    {'acc': 0.8539823008849557, 'mean_precision': 0.8017848860212725, 'mean_recall': 0.8119553221526408, 'macro_f1': 0.804904791937732, 'weighted_precision': 0.8588867681742558, 'weighted_recall': 0.8539823008849557, 'weighted_f1': 0.855560943534421}
    EPOCH 40
    =================================
    Training Results
    {'acc': 0.9644, 'mean_precision': 0.9670062774145092, 'mean_recall': 0.9573277463956331, 'macro_f1': 0.9619863456379548, 'weighted_precision': 0.9645118227152805, 'weighted_recall': 0.9644, 'weighted_f1': 0.9644225052829172}
    Validation Results
    {'acc': 0.8495575221238938, 'mean_precision': 0.8119564587793305, 'mean_recall': 0.8080099729691964, 'macro_f1': 0.8081568266024983, 'weighted_precision': 0.8580128676213757, 'weighted_recall': 0.8495575221238938, 'weighted_f1': 0.8519747328540073}
    EPOCH 41
    =================================
    Training Results
    {'acc': 0.9628, 'mean_precision': 0.9582207266041486, 'mean_recall': 0.9557915435105366, 'macro_f1': 0.956989832500887, 'weighted_precision': 0.9628821971783511, 'weighted_recall': 0.9628, 'weighted_f1': 0.9628296736582989}
    Validation Results
    {'acc': 0.8451327433628318, 'mean_precision': 0.7965093061348186, 'mean_recall': 0.8045728236987139, 'macro_f1': 0.7982598158685201, 'weighted_precision': 0.8513942451058316, 'weighted_recall': 0.8451327433628318, 'weighted_f1': 0.8470011392375405}
    EPOCH 42
    =================================
    Training Results
    {'acc': 0.964, 'mean_precision': 0.9645740109204436, 'mean_recall': 0.9606825654964162, 'macro_f1': 0.9625915166364768, 'weighted_precision': 0.9640606955680181, 'weighted_recall': 0.964, 'weighted_f1': 0.9640155053392376}
    Validation Results
    {'acc': 0.8495575221238938, 'mean_precision': 0.8037359987261851, 'mean_recall': 0.8058441622200901, 'macro_f1': 0.8012095203517711, 'weighted_precision': 0.8602910880270568, 'weighted_recall': 0.8495575221238938, 'weighted_f1': 0.8518937207419582}
    EPOCH 43
    =================================
    Training Results
    {'acc': 0.967, 'mean_precision': 0.9673205256002168, 'mean_recall': 0.9608809490156882, 'macro_f1': 0.9640313629040639, 'weighted_precision': 0.967090742201184, 'weighted_recall': 0.967, 'weighted_f1': 0.9670273824588701}
    Validation Results
    {'acc': 0.8584070796460177, 'mean_precision': 0.8071223611395778, 'mean_recall': 0.815180322836991, 'macro_f1': 0.8086659140417566, 'weighted_precision': 0.8654518199826157, 'weighted_recall': 0.8584070796460177, 'weighted_f1': 0.8603401318409323}
    EPOCH 44
    =================================
    Training Results
    {'acc': 0.9658, 'mean_precision': 0.9609493483829002, 'mean_recall': 0.9637935616364777, 'macro_f1': 0.9623305485433838, 'weighted_precision': 0.9659282069261604, 'weighted_recall': 0.9658, 'weighted_f1': 0.9658413169266654}
    Validation Results
    {'acc': 0.8451327433628318, 'mean_precision': 0.811964839602692, 'mean_recall': 0.804398454041185, 'macro_f1': 0.8075118081867685, 'weighted_precision': 0.8495195126057683, 'weighted_recall': 0.8451327433628318, 'weighted_f1': 0.846453823025803}
    EPOCH 45
    =================================
    Training Results
    {'acc': 0.9658, 'mean_precision': 0.9598439053443757, 'mean_recall': 0.9526976401858064, 'macro_f1': 0.9561667547996912, 'weighted_precision': 0.9657984947117778, 'weighted_recall': 0.9658, 'weighted_f1': 0.9657855836823659}
    Validation Results
    {'acc': 0.8539823008849557, 'mean_precision': 0.8027867894992161, 'mean_recall': 0.8119553221526408, 'macro_f1': 0.8053321278501304, 'weighted_precision': 0.8592071325485433, 'weighted_recall': 0.8539823008849557, 'weighted_f1': 0.8556693507865624}
    EPOCH 46
    =================================
    Training Results
    {'acc': 0.9662, 'mean_precision': 0.9604700540065974, 'mean_recall': 0.9625075307979163, 'macro_f1': 0.9614812705685866, 'weighted_precision': 0.9662056816206401, 'weighted_recall': 0.9662, 'weighted_f1': 0.966200656893312}
    Validation Results
    {'acc': 0.8606194690265486, 'mean_precision': 0.8296025254745819, 'mean_recall': 0.8174572008429252, 'macro_f1': 0.8224370751844012, 'weighted_precision': 0.8663240618203951, 'weighted_recall': 0.8606194690265486, 'weighted_f1': 0.8621524978508673}
    EPOCH 47
    =================================
    Training Results
    {'acc': 0.9658, 'mean_precision': 0.9623643280997959, 'mean_recall': 0.9601608706065794, 'macro_f1': 0.9612521277633985, 'weighted_precision': 0.9657974850075743, 'weighted_recall': 0.9658, 'weighted_f1': 0.9657933462320052}
    Validation Results
    {'acc': 0.8606194690265486, 'mean_precision': 0.8289130643629324, 'mean_recall': 0.8162020131702755, 'macro_f1': 0.8212697994296662, 'weighted_precision': 0.8663358703067995, 'weighted_recall': 0.8606194690265486, 'weighted_f1': 0.8619390343788721}
    EPOCH 48
    =================================
    Training Results
    {'acc': 0.9694, 'mean_precision': 0.965498989334344, 'mean_recall': 0.9691871765140303, 'macro_f1': 0.9673126581580327, 'weighted_precision': 0.969441121137196, 'weighted_recall': 0.9694, 'weighted_f1': 0.969411138333605}
    Validation Results
    {'acc': 0.8473451327433629, 'mean_precision': 0.8108888178689653, 'mean_recall': 0.8042876327132501, 'macro_f1': 0.8053138141785771, 'weighted_precision': 0.8569467358619901, 'weighted_recall': 0.8473451327433629, 'weighted_f1': 0.8498193213999172}
    Final result
    {'acc': 0.882, 'mean_precision': 0.8966656639557661, 'mean_recall': 0.8599837937999005, 'macro_f1': 0.8737226125229776, 'weighted_precision': 0.8813809951547577, 'weighted_recall': 0.882, 'weighted_f1': 0.878768700658464}





    0.882



## Conclusion

Without even concatenating word features, our ELMo model, with far fewer parameters, surpasses the performance of the randomly initialized baseline, which we would expect.  It also significantly out-performs our CNN pre-trained, fine-tuned word embeddings baseline from the last section -- that model's max performance is around 93.  Note that this dataset is tiny, and the variance is large between datasets, but this model consistently outperforms both CNN and LSTM baselines.

Contextual embeddings consistently outperform non-contextual embeddings on almost every task in NLP, not just in text classification.  This method is becoming so commonly used that some papers have even started reporting this approach as a baseline.

### Some more references

- The PyTorch examples actually contain a [nice word-language model](https://github.com/pytorch/examples/tree/master/word_language_model)

- There is a [Tensorflow tutorial](https://www.tensorflow.org/tutorials/sequences/recurrent) as well

- The original source code for training [ELMo's bilm is here](https://github.com/allenai/bilm-tf/tree/master/bilm)

- [A succinct implementation](https://github.com/dpressel/baseline/blob/master/python/baseline/pytorch/embeddings.py#L63) of character-compositional embeddings in Baseline for PyTorch





