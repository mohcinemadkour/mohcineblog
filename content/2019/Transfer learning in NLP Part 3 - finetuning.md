title: Transfer learning in NLP Part III: Fine-tuning a pre-trained model
Date: 2019-07-07 13:01
Category: Machine Learning, July 2019, Transfer learning, filtering
Tags: NLP, July 2019, Transfer learning, filtering
Slug: Machine Learning, July 2019, Transfer learning, filtering
Author: Mohcine Madkour
Email:mohcine.madkour@gmail.com

In the last section, we looked at using a biLM networks layers as embeddings for our classification model.  In that approach, we maintain the exact same model architecture as before, but just switching our word embeddings out for context embeddings (or, more commonly, using them in concert).

The paper [Improving Language Understanding
by Generative Pre-Training](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf) (Radford et al 2018) explored a different approach, much more similar to what is typically done in computer vision.  In fine-tuning, we reuse the network architecture and simply replace the head.  We dont use any model specific architecture anymore, just a final layer.  There is an accompanying blog post [here](https://openai.com/blog/language-unsupervised/).  The image below is borrowed from that blog post

![alt text](https://openai.com/content/images/2018/06/zero-shot-transfer@2x.png)

As we can see from the images, these models can rapidly improve our downstream performance with very limited fine-tuning supervision.



## The Transformer

The original Transformer is an all-attention encoder-decoder model first introduced in [Attention Is All You Need, Vaswani et al., 2017](https://arxiv.org/abs/1706.03762).  It is described at a high-level in [this Google AI post](https://ai.googleblog.com/2017/08/transformer-novel-neural-network.html).
Here is an image of the model architecture for a Transformer:

![Transformer Architecture](http://nlp.seas.harvard.edu/images/the-annotated-transformer_14_0.png)

The reference implementation from Google is the [tensor2tensor repository](https://github.com/tensorflow/tensor2tensor/tree/master/tensor2tensor).  There is a lot going on in that codebase, which some people may find hard to follow.

We are going to go through each component in a hands-on manner, which will hopefully give you a visual feel of what is happening.

If you want to understand Transformers better, there is a terrific blog post called [The Annotated Transformer, Rush, 2018](http://nlp.seas.harvard.edu/2018/04/03/attention.html) where you can see how to code up a Transformer from scratch to do Neural Machine Translation (NMT) while following along with the paper.

In versions used in practice, there are slight differences from the actual image, most notably, that layer norm is performed first.  Also, in a causal LM pre-training setting, as in the case of GPT, we have no need for the decoder, which simplifies our architecture substantially, leaving only a masked self-attention in the encoder (this prevents us from seeing the future as we predict).




### A Transformer Encoder Layer

Here is code adapted from [Baseline](https://github.com/dpressel/baseline) that implements a Transformer block used in a GPT-like architecture (pictured above).  We are going to take a closer look at these blocks, so lets think of this as the high-level overview.  The input to this class is a `torch.Tensor` of shape `BxT`.  The first sub-component in a Transformer block is the Multi-Headed Attention.  The second is the "FFN" shown in the image -- an MLP layer followed by a linear projection back to the original size.  We encapsulate these transformations in an `nn.Sequential`.  Notice that each sub-layer is also a residual connection.


```python

class TransformerEncoder(nn.Module):
    def __init__(self, num_heads, d_model, pdrop, scale=True, activation_type='relu', d_ff=None):
        """
        :param num_heads (`int`): the number of heads for self-attention
        :param d_model (`int`): The model dimension size
        :param pdrop (`float`): The dropout probability
        :param scale (`bool`): Whether we are doing scaled dot-product attention
        :param activation_type: What activation type to use
        :param d_ff: The feed forward layer size
        """
        super(TransformerEncoder, self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff if d_ff is not None else num_heads * d_model
        self.self_attn = MultiHeadedAttention(num_heads, d_model, pdrop, scale=scale)
        self.ffn = nn.Sequential(nn.Linear(self.d_model, self.d_ff),
                                 pytorch_activation(activation_type),
                                 nn.Linear(self.d_ff, self.d_model))
        self.ln1 = nn.LayerNorm(self.d_model, eps=1e-12)
        self.ln2 = nn.LayerNorm(self.d_model, eps=1e-12)
        self.dropout = nn.Dropout(pdrop)

    def forward(self, x, mask=None):
        """
        :param x: the inputs
        :param mask: a mask for the inputs
        :return: the encoder output
        """
        # Builtin Attention mask
        x = self.ln1(x)
        h = self.self_attn(x, x, x, mask)
        x = x + self.dropout(h)

        x = self.ln2(x)
        x = x + self.dropout(self.ffn(x))
        return x

```



### Multi-headed Attention

Multi-headed attention is one of the key innovations of the Transformer.  The idea was to allow each attention head to learn different relations.

![MHA](https://1.bp.blogspot.com/-AVGK0ApREtk/WaiAuzddKVI/AAAAAAAAB_A/WPV5ropBU-cxrcMpqJBFHg73K9NX4vywwCLcBGAs/s1600/image2.png)

#### Scaled dot product attention

Here is a picture of the operations involved in scaled dot product attention.

![MHA Architecture](http://nlp.seas.harvard.edu/images/the-annotated-transformer_33_0.png)

`Q`, `K` and `V` are low-order projections of the input.  For Encoder-Decoders, the `Q` is a query vector in the decoder, and `K` and `V` are representations of the Encoder.  A dot product of the encoder keys and the query vector determines a set of weights that are applied against the `V` (again, also a representation of the encoder values).  In the case of the encoder, these are all drawn from the same input.  Basic dot product attention was actually introduced in [Effective Approaches to Attention-based Neural Machine Translation, Luong et al., 2014](https://arxiv.org/abs/1508.04025), but in the the Transformer paper, the authors made a strong case that the basic dot product attention benefits from scaling.

This is implemented (again adapted from Baseline), as follows:

```python
def scaled_dot_product_attention(query, key, value, mask=None, dropout=None):
    """Scaled dot product attention, as defined in https://arxiv.org/abs/1706.03762

    We apply the query to the keys to recieve our weights via softmax, which are then applied
    for each value, but in a series of efficient matrix operations.  In the case of self-attention,
    the key, query and values are all low order projections of the same input.

    :param query: a query for alignment. Can come from self in case of self-attn or decoder in case of E/D
    :param key: a set of keys from encoder or self
    :param value: a set of values from encoder or self
    :param mask: masking (for destination) to prevent seeing what we shouldnt
    :param dropout: apply dropout operator post-attention (this is not a float)
    :return: A tensor that is (BxHxTxT)

    """
    # (., H, T, T) = (., H, T, D) x (., H, D, T)
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    weights = F.softmax(scores, dim=-1)
    if dropout is not None:
        weights = dropout(weights)
    return torch.matmul(weights, value), weights
```


#### The Multi-head part

Each of the attention operations above that we apply is going to learn some weighted representation of our input -- what are we paying attention to?  There are lots of things that might be useful!   We might want to attend to the next word for language modeling.  To remember what we said, we might want to learn something like which pronouns refer to which nouns that we saw in previous tokens (this is called anaphora resolution and is a subset of coreference resolution).  We might hope that it picks up parse dependencies, that could help us with tasks that benefit from syntax.  Remember that each of our `Q`, `K` and `V` are low-order projections of our input.  What if we had many low-order projections and used each to learn different weightings?  This  is exactly what multi-head attention is.  Each "head" does the operation above and learns something meaningful (or at least, we hope it does!).

Here is some code that implements multi-headed attention using our function above:

```python
class MultiHeadedAttention(nn.Module):
    """
    Multi-headed attention from https://arxiv.org/abs/1706.03762 via http://nlp.seas.harvard.edu/2018/04/03/attention.html

    Multi-headed attention provides multiple looks of low-order projections K, Q and V using an attention function
    (specifically `scaled_dot_product_attention` in the paper.  This allows multiple relationships to be illuminated
    via attention on different positional and representational information from each head.

    The number of heads `h` times the low-order projection dim `d_k` is equal to `d_model` (which is asserted upfront).
    This means that each weight matrix can be simply represented as a linear transformation from `d_model` to `d_model`,
    and partitioned into heads after the fact.

    Finally, an output projection is applied which brings the output space back to `d_model`, in preparation for the
    sub-sequent `FFN` sub-layer.

    There are 3 uses of multi-head attention in the Transformer.
    For encoder-decoder layers, the queries come from the previous decoder layer, and the memory keys come from
    the encoder.  For encoder layers, the K, Q and V all come from the output of the previous layer of the encoder.
    And for self-attention in the decoder, K, Q and V all come from the decoder, but here it is masked to prevent using
    future values
    """
    def __init__(self, h, d_model, dropout=0.1, scale=False):
        """Constructor for multi-headed attention

        :param h: The number of heads
        :param d_model: The model hidden size
        :param dropout (``float``): The amount of dropout to use
        :param attn_fn: A function to apply attention, defaults to SDP
        """
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.w_Q = nn.Linear(d_model, d_model)
        self.w_K = nn.Linear(d_model, d_model)
        self.w_V = nn.Linear(d_model, d_model)
        self.w_O = nn.Linear(d_model, d_model)
        self.attn_fn = scaled_dot_product_attention if scale else dot_product_attention
        self.attn = None
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        """Low-order projections of query, key and value into multiple heads, then attention application and dropout

        :param query: a query for alignment. Can come from self in case of self-attn or decoder in case of E/D
        :param key: a set of keys from encoder or self
        :param value: a set of values from encoder or self
        :param mask: masking (for destination) to prevent seeing what we shouldnt
        :return: Multi-head attention output, result of attention application to sequence (B, T, d_model)
        """
        batchsz = query.size(0)

        # (B, H, T, D)
        query = self.w_Q(query).view(batchsz, -1, self.h, self.d_k).transpose(1, 2)
        key = self.w_K(key).view(batchsz, -1, self.h, self.d_k).transpose(1, 2)
        value = self.w_V(value).view(batchsz, -1, self.h, self.d_k).transpose(1, 2)

        x, self.attn = self.attn_fn(query, key, value, mask=mask, dropout=self.dropout)

        x = x.transpose(1, 2).contiguous() \
            .view(batchsz, -1, self.h * self.d_k)
        return self.w_O(x)
```



We are going to take a look at how multi-headed attention works visually. To do this, we are going to use the [viz-bert codebase](https://github.com/jessevig/bertviz) from Jesse Vig.  The accompanying paper is [A Multiscale Visualization of Attention in the Transformer Model, Vig, 2019](https://arxiv.org/pdf/1906.05714.pdf).




```python
import sys

!test -d bertviz_repo && echo "FYI: bertviz_repo directory already exists, to pull latest version uncomment this line: !rm -r bertviz_repo"
# !rm -r bertviz_repo # Uncomment if you need a clean pull from repo
!test -d bertviz_repo || git clone https://github.com/jessevig/bertviz bertviz_repo
if not 'bertviz_repo' in sys.path:
  sys.path += ['bertviz_repo']
!pip install regex

```

    Cloning into 'bertviz_repo'...
    remote: Enumerating objects: 3, done.[K
    remote: Counting objects: 100% (3/3), done.[K
    remote: Compressing objects: 100% (3/3), done.[K
    remote: Total 488 (delta 0), reused 1 (delta 0), pack-reused 485[K
    Receiving objects: 100% (488/488), 37.01 MiB | 22.80 MiB/s, done.
    Resolving deltas: 100% (294/294), done.
    Collecting regex
    [?25l  Downloading https://files.pythonhosted.org/packages/6f/4e/1b178c38c9a1a184288f72065a65ca01f3154df43c6ad898624149b8b4e0/regex-2019.06.08.tar.gz (651kB)
    [K     |████████████████████████████████| 655kB 9.8MB/s 
    [?25hBuilding wheels for collected packages: regex
      Building wheel for regex (setup.py) ... [?25l[?25hdone
      Stored in directory: /root/.cache/pip/wheels/35/e4/80/abf3b33ba89cf65cd262af8a22a5a999cc28fbfabea6b38473
    Successfully built regex
    Installing collected packages: regex
    Successfully installed regex-2019.6.8



```python
from bertviz import attention, visualization
from bertviz.pytorch_pretrained_bert import BertModel as VizBertModel
from bertviz.pytorch_pretrained_bert import BertTokenizer as VizBertTokenizer
```

    Better speed can be achieved with apex installed from https://www.github.com/nvidia/apex.



```javascript
%%javascript
require.config({
  paths: {
      d3: '//cdnjs.cloudflare.com/ajax/libs/d3/3.4.8/d3.min'
  }
});
```


    <IPython.core.display.Javascript object>



```python
def call_html():
  import IPython
  display(IPython.core.display.HTML('''
        <script src="/static/components/requirejs/require.js"></script>
        <script>
          requirejs.config({
            paths: {
              base: '/static/base',
              "d3": "https://cdnjs.cloudflare.com/ajax/libs/d3/3.5.8/d3.min",
              jquery: '//ajax.googleapis.com/ajax/libs/jquery/2.0.0/jquery.min',
            },
          });
        </script>
        '''))
```


```python
model = VizBertModel.from_pretrained('bert-base-uncased')
tokenizer = VizBertTokenizer.from_pretrained('bert-base-uncased')
sentence_a = "The dog crossed the road ."
sentence_b = "The owner came out and put him on a leash ."
attention_visualizer = visualization.AttentionVisualizer(model, tokenizer)
tokens_a, tokens_b, attn = attention_visualizer.get_viz_data(sentence_a, sentence_b)
call_html()
attention.show(tokens_a, tokens_b, attn)
```

    100%|██████████| 407873900/407873900 [00:14<00:00, 27887659.37B/s]
    100%|██████████| 231508/231508 [00:00<00:00, 913649.61B/s]




        <script src="/static/components/requirejs/require.js"></script>
        <script>
          requirejs.config({
            paths: {
              base: '/static/base',
              "d3": "https://cdnjs.cloudflare.com/ajax/libs/d3/3.5.8/d3.min",
              jquery: '//ajax.googleapis.com/ajax/libs/jquery/2.0.0/jquery.min',
            },
          });
        </script>
        




  <span style="user-select:none">
    Layer: <select id="layer"></select>
    Attention: <select id="att_type">
      <option value="all">All</option>
      <option value="a">Sentence A self-attention</option>
      <option value="b">Sentence B self-attention</option>
      <option value="ab">Sentence A -> Sentence B</option>
      <option value="ba">Sentence B -> Sentence A</option>
    </select>
  </span>
  <div id='vis'></div>




    <IPython.core.display.Javascript object>



    <IPython.core.display.Javascript object>


Try playing around with `sentence_a` and `sentence_b`.  You can select and unselect different attention heads, as well as the layer that you are visualizing.  There is a lot going on here.  [This blog post](https://towardsdatascience.com/deconstructing-bert-distilling-6-patterns-from-100-million-parameters-b49113672f77)  by Jesse Vig, the author of the software we are using to render the attention heads above, discusses how BERT attention heads learn various types of attention.  [Clark et al 2019 have a paper](https://arxiv.org/abs/1906.04341) that also delves into what learns, particular in the context of our linguistic notions of syntax

It turns out BERT learns a lot of stuff:


- **next/previous/identical word tracking**

- **stuff that correlates closely to linguistic notions of syntax**:  

  - BERT attention heads learn something like coreference
  - BERT attention heads learn some approximation of dependency parsing.  Different attention heads learn different dependency/governor relationships




#### Multi-Headed Attention is easy now in PyTorch!!

This operation is now built into PyTorch.  There is a caveat that only scaled-dot product attention is supported.  The code above does not use that module since it supports both scaled and unscaled attention.





### Positional embeddings

To eliminate auto-regressive (RNN) models from the transformer, positional embeddings need to be created and added to the word embeddings.  Otherwise, during attention there would be no way to account for word position. There are several ways to support positional embeddings.

The first way is very simple -- you just need to create a `nn.Embedding` that you give your offsets for each token.  Embedding representations will be learned for each position, but you can only learn up to the number of positions you have seen.

Another way, used in the original Transformer is to embed a bunch of sinusoids with different frequencies that are a function of the position:

$$PE_{(pos,2i)}=sin(pos/10000^{2i}/dmodel)$$
$$PE_{(pos,2i+1)}=cos(pos/10000^{2i}/dmodel)$$ 

where $pos$ is the position and $i$ is the dimension corresponding to a sinusoid. The wavelengths form a geometric progression from $2\pi$ to $10000\times2\pi$.



## BERT

For this section of the tutorial, we are going to fine-tune BERT [Devlin et al 2018](https://arxiv.org/abs/1810.04805), a transformer architecture that replaces the causal LM objective with 2 new objectives:

1. **Masking out words** with some probability, predict the missing words (MLM objective)

![MLM](https://2.bp.blogspot.com/-pNxcHHXNZg0/W9iv3evVyOI/AAAAAAAADfA/KTSvKXNzzL0W8ry28PPl7nYI1CG_5WuvwCLcBGAs/s1600/f1.png)

2. Given 2 adjacent sentences, **predict if the second sentence follows the first** (NSP objective)

![NSP](https://4.bp.blogspot.com/-K_7yu3kjF18/W9iv-R-MnyI/AAAAAAAADfE/xUwR_G1iTY0vq9X-Z3LnW5t4NLS9BQzdgCLcBGAs/s1600/f2.png)

From an architecture diagram, [this blog post announcing BERT](https://ai.googleblog.com/2018/11/open-sourcing-bert-state-of-art-pre.html) notes the differences:

![BERT vs GPT and ELMo](https://1.bp.blogspot.com/-RLAbr6kPNUo/W9is5FwUXmI/AAAAAAAADeU/5y9466Zoyoc96vqLjbruLK8i_t8qEdHnQCLcBGAs/s1600/image3.png)

Our model will simply build on the existing model architecture with a single transformation layer to the output number of classes.  BERT is [open source](https://github.com/google-research/bert) but the code is in TensorFlow, and since this tutorial is written in PyTorch, we need a different solution.  We will use the [Hugging Face Transformer codebase](https://github.com/huggingface/pytorch-pretrained-BERT) as our API -- it can read in the original Google-trained weights.


```python
!pip install pytorch-pretrained-bert

```

    Collecting pytorch-pretrained-bert
    [?25l  Downloading https://files.pythonhosted.org/packages/d7/e0/c08d5553b89973d9a240605b9c12404bcf8227590de62bae27acbcfe076b/pytorch_pretrained_bert-0.6.2-py3-none-any.whl (123kB)
    [K     |████████████████████████████████| 133kB 9.5MB/s 
    [?25hRequirement already satisfied: regex in /usr/local/lib/python3.6/dist-packages (from pytorch-pretrained-bert) (2019.6.8)
    Requirement already satisfied: torch>=0.4.1 in /usr/local/lib/python3.6/dist-packages (from pytorch-pretrained-bert) (1.1.0)
    Requirement already satisfied: numpy in /usr/local/lib/python3.6/dist-packages (from pytorch-pretrained-bert) (1.16.4)
    Requirement already satisfied: boto3 in /usr/local/lib/python3.6/dist-packages (from pytorch-pretrained-bert) (1.9.175)
    Requirement already satisfied: requests in /usr/local/lib/python3.6/dist-packages (from pytorch-pretrained-bert) (2.21.0)
    Requirement already satisfied: tqdm in /usr/local/lib/python3.6/dist-packages (from pytorch-pretrained-bert) (4.28.1)
    Requirement already satisfied: botocore<1.13.0,>=1.12.175 in /usr/local/lib/python3.6/dist-packages (from boto3->pytorch-pretrained-bert) (1.12.175)
    Requirement already satisfied: jmespath<1.0.0,>=0.7.1 in /usr/local/lib/python3.6/dist-packages (from boto3->pytorch-pretrained-bert) (0.9.4)
    Requirement already satisfied: s3transfer<0.3.0,>=0.2.0 in /usr/local/lib/python3.6/dist-packages (from boto3->pytorch-pretrained-bert) (0.2.1)
    Requirement already satisfied: idna<2.9,>=2.5 in /usr/local/lib/python3.6/dist-packages (from requests->pytorch-pretrained-bert) (2.8)
    Requirement already satisfied: chardet<3.1.0,>=3.0.2 in /usr/local/lib/python3.6/dist-packages (from requests->pytorch-pretrained-bert) (3.0.4)
    Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.6/dist-packages (from requests->pytorch-pretrained-bert) (2019.6.16)
    Requirement already satisfied: urllib3<1.25,>=1.21.1 in /usr/local/lib/python3.6/dist-packages (from requests->pytorch-pretrained-bert) (1.24.3)
    Requirement already satisfied: docutils>=0.10 in /usr/local/lib/python3.6/dist-packages (from botocore<1.13.0,>=1.12.175->boto3->pytorch-pretrained-bert) (0.14)
    Requirement already satisfied: python-dateutil<3.0.0,>=2.1; python_version >= "2.7" in /usr/local/lib/python3.6/dist-packages (from botocore<1.13.0,>=1.12.175->boto3->pytorch-pretrained-bert) (2.5.3)
    Requirement already satisfied: six>=1.5 in /usr/local/lib/python3.6/dist-packages (from python-dateutil<3.0.0,>=2.1; python_version >= "2.7"->botocore<1.13.0,>=1.12.175->boto3->pytorch-pretrained-bert) (1.12.0)
    Installing collected packages: pytorch-pretrained-bert
    Successfully installed pytorch-pretrained-bert-0.6.2



```python
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import io
import os
import re
import codecs
from collections import Counter
from torch.utils.data import DataLoader, TensorDataset
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertModel
```


### Tokenization in BERT

In the last sequence, we talked about how ELMo biLMs can limit their parameters while accounting for unseen words using character-compositional word embeddings.  This technique is very powerful, but its also slow.  It is common in NMT to use some sort of sub-word encoding that limits the vocabulary size, but allows us to not have unattested words.  The `tensor2tensor` codebase, for example, creates an invertible encoding for words into sub-tokens with a limited vocabulary.  The tokenizer is built from a corpus upfront and stored in a file, and then can be used to encode text.

There are 4 phases in this algorithm described in the tensor2tensor codebase:


    1. Tokenize into a list of tokens.  Each token is a unicode string of either
      all alphanumeric characters or all non-alphanumeric characters.  We drop
      tokens consisting of a single space that are between two alphanumeric
      tokens.
    2. Escape each token.  This escapes away special and out-of-vocabulary
      characters, and makes sure that each token ends with an underscore, and
      has no other underscores.
    3. Represent each escaped token as a the concatenation of a list of subtokens
      from the limited vocabulary.  Subtoken selection is done greedily from
      beginning to end.  That is, we construct the list in order, always picking
      the longest subtoken in our vocabulary that matches a prefix of the
      remaining portion of the encoded token.
    4. Concatenate these lists.  This concatenation is invertible due to the
      fact that the trailing underscores indicate when one list is finished.



We can access Google's trained BERT Tokenizer via the Hugging Face API


```python
def whitespace_tokenizer(words):
    return words.split() 

def sst2_tokenizer(words):
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

BERT_TOKENIZER = BertTokenizer.from_pretrained('bert-base-uncased')
BERT_MODEL = BertModel.from_pretrained('bert-base-uncased')
def bert_tokenizer(words, pretokenizer=whitespace_tokenizer):
    subwords = ['[CLS]']
    for word in pretokenizer(words):
        if word == '<unk>':
            subword = '[UNK]'
        else:
            subword = BERT_TOKENIZER.tokenize(word)
        subwords += subword
    return subwords + ['[SEP]']

def bert_vectorizer(sentence):
    return BERT_TOKENIZER.convert_tokens_to_ids(sentence)
    #return [BERT_TOKENIZER.vocab.get(subword, BERT_TOKENIZER.vocab['[PAD]']) for subword in sentence]


```

Our model this time around is very simple.  It has an output linear layer that comes from pooled output from BERT


```python

class FineTuneClassifier(nn.Module):

    def __init__(self, base_model, num_classes, embed_dim, hidden_units=[]):
        super().__init__()
        self.base_model = base_model
        input_units = embed_dim
        output_units = embed_dim
        sequence = []
        for h in hidden_units:
            sequence.append(nn.Linear(input_units, h))
            input_units = h
            output_units = h
            
        sequence.append(nn.Linear(output_units, num_classes))
        self.outputs = nn.Sequential(*sequence)

    def forward(self, inputs):
        x, lengths = inputs
        
        input_mask = torch.zeros(x.shape, device=x.device, dtype=torch.long).masked_fill(x != 0, 1)
        input_type_ids = torch.zeros(x.shape, device=x.device, dtype=torch.long)
        _, pooled = self.base_model(x, token_type_ids=input_type_ids, attention_mask=input_mask)
        
        stacked = self.outputs(pooled)
        return F.log_softmax(stacked, dim=-1)
```

All the rest of our code comes from the previous sections


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

Lets use the trec dataset again


```python
!wget https://www.dropbox.com/s/08km2ean8bkt7p3/trec.tar.gz?dl=1
!tar -xzf 'trec.tar.gz?dl=1'
```

    --2019-06-30 00:05:36--  https://www.dropbox.com/s/08km2ean8bkt7p3/trec.tar.gz?dl=1
    Resolving www.dropbox.com (www.dropbox.com)... 162.125.65.1, 2620:100:6021:1::a27d:4101
    Connecting to www.dropbox.com (www.dropbox.com)|162.125.65.1|:443... connected.
    HTTP request sent, awaiting response... 301 Moved Permanently
    Location: /s/dl/08km2ean8bkt7p3/trec.tar.gz [following]
    --2019-06-30 00:05:36--  https://www.dropbox.com/s/dl/08km2ean8bkt7p3/trec.tar.gz
    Reusing existing connection to www.dropbox.com:443.
    HTTP request sent, awaiting response... 302 Found
    Location: https://ucc71251671e8209ff817842136f.dl.dropboxusercontent.com/cd/0/get/Ajy7yx6BcLy_D2C847YK2MZsIYRIe4WHUQShODQUsHevIJVdUp_Gu5qxvUTiNpCJ_u89irvfKQJ8E71KrGpT_m0HhXBh79ywpr8iSXN5QO5OpQ/file?dl=1# [following]
    --2019-06-30 00:05:37--  https://ucc71251671e8209ff817842136f.dl.dropboxusercontent.com/cd/0/get/Ajy7yx6BcLy_D2C847YK2MZsIYRIe4WHUQShODQUsHevIJVdUp_Gu5qxvUTiNpCJ_u89irvfKQJ8E71KrGpT_m0HhXBh79ywpr8iSXN5QO5OpQ/file?dl=1
    Resolving ucc71251671e8209ff817842136f.dl.dropboxusercontent.com (ucc71251671e8209ff817842136f.dl.dropboxusercontent.com)... 162.125.65.6, 2620:100:6021:6::a27d:4106
    Connecting to ucc71251671e8209ff817842136f.dl.dropboxusercontent.com (ucc71251671e8209ff817842136f.dl.dropboxusercontent.com)|162.125.65.6|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 117253 (115K) [application/binary]
    Saving to: ‘trec.tar.gz?dl=1’
    
    trec.tar.gz?dl=1    100%[===================>] 114.50K  --.-KB/s    in 0.01s   
    
    2019-06-30 00:05:37 (11.3 MB/s) - ‘trec.tar.gz?dl=1’ saved [117253/117253]
    



```python
BASE = 'trec'
TRAIN = os.path.join(BASE, 'trec.nodev.utf8')
VALID = os.path.join(BASE, 'trec.dev.utf8')
TEST = os.path.join(BASE, 'trec.test.utf8')

# lowercase=False so we can defer to BERT's tokenizer to handle
r = Reader((TRAIN, VALID, TEST,), lowercase=False, vectorizer=bert_vectorizer, tokenizer=bert_tokenizer)
train = r.load(TRAIN)
valid = r.load(VALID)
test = r.load(TEST)
```


```python
bert_small_dims = 768
batch_size = 50
epochs = 12
model = FineTuneClassifier(BERT_MODEL, len(r.labels), bert_small_dims)
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Model has {num_params} parameters") 


model.to('cuda:0')
loss = torch.nn.NLLLoss()
loss = loss.to('cuda:0')

learnable_params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.Adam(learnable_params, lr=1.0e-4)

fit(model, r.labels, optimizer, loss, epochs, batch_size, train, valid, test)
```

    Model has 109486854 parameters
    EPOCH 1
    =================================
    Training Results
    {'acc': 0.8072, 'mean_precision': 0.8136183754170369, 'mean_recall': 0.7346122783828277, 'macro_f1': 0.7542932115362501, 'weighted_precision': 0.8086486190442354, 'weighted_recall': 0.8072, 'weighted_f1': 0.8039545756095884}
    Validation Results
    {'acc': 0.8960176991150443, 'mean_precision': 0.836374911050043, 'mean_recall': 0.8982751203248097, 'macro_f1': 0.8490935982370543, 'weighted_precision': 0.9102435016370314, 'weighted_recall': 0.8960176991150443, 'weighted_f1': 0.8940889603495116}
    New best model 0.90
    EPOCH 2
    =================================
    Training Results
    {'acc': 0.9592, 'mean_precision': 0.9564446996913963, 'mean_recall': 0.9441293710502765, 'macro_f1': 0.9499487073005133, 'weighted_precision': 0.9591434277886723, 'weighted_recall': 0.9592, 'weighted_f1': 0.9591174670360441}
    Validation Results
    {'acc': 0.9446902654867256, 'mean_precision': 0.8878199940761919, 'mean_recall': 0.8917873400275057, 'macro_f1': 0.8895121143056146, 'weighted_precision': 0.9449440203747359, 'weighted_recall': 0.9446902654867256, 'weighted_f1': 0.9445031298581608}
    New best model 0.94
    EPOCH 3
    =================================
    Training Results
    {'acc': 0.9784, 'mean_precision': 0.9682000141119786, 'mean_recall': 0.961629040591354, 'macro_f1': 0.9648113009615465, 'weighted_precision': 0.9783229179206641, 'weighted_recall': 0.9784, 'weighted_f1': 0.9783479120110454}
    Validation Results
    {'acc': 0.9513274336283186, 'mean_precision': 0.904094720396214, 'mean_recall': 0.9185124935642534, 'macro_f1': 0.9107662900322363, 'weighted_precision': 0.9517688289888032, 'weighted_recall': 0.9513274336283186, 'weighted_f1': 0.9514516397598308}
    New best model 0.95
    EPOCH 4
    =================================
    Training Results
    {'acc': 0.9878, 'mean_precision': 0.9879920250105357, 'mean_recall': 0.9803402721232928, 'macro_f1': 0.9840564690593671, 'weighted_precision': 0.9877906525358927, 'weighted_recall': 0.9878, 'weighted_f1': 0.987783074278954}
    Validation Results
    {'acc': 0.9469026548672567, 'mean_precision': 0.8920679986535167, 'mean_recall': 0.9123880140014059, 'macro_f1': 0.9004386136056346, 'weighted_precision': 0.9489356221151654, 'weighted_recall': 0.9469026548672567, 'weighted_f1': 0.9476173076771826}
    EPOCH 5
    =================================
    Training Results
    {'acc': 0.9904, 'mean_precision': 0.9904490381789716, 'mean_recall': 0.9862979863492165, 'macro_f1': 0.9883441530923012, 'weighted_precision': 0.9904177347691059, 'weighted_recall': 0.9904, 'weighted_f1': 0.9904029148588175}
    Validation Results
    {'acc': 0.922566371681416, 'mean_precision': 0.8451025054934176, 'mean_recall': 0.9188274343553847, 'macro_f1': 0.8614088235624355, 'weighted_precision': 0.9374259585988132, 'weighted_recall': 0.922566371681416, 'weighted_f1': 0.9265075358706969}
    EPOCH 6
    =================================
    Training Results
    {'acc': 0.9928, 'mean_precision': 0.9849192613386126, 'mean_recall': 0.9903185624313342, 'macro_f1': 0.9875625640550693, 'weighted_precision': 0.9928298071879748, 'weighted_recall': 0.9928, 'weighted_f1': 0.9928084053080909}
    Validation Results
    {'acc': 0.9513274336283186, 'mean_precision': 0.893967686997322, 'mean_recall': 0.9183550231686878, 'macro_f1': 0.9044542863397047, 'weighted_precision': 0.952784511352824, 'weighted_recall': 0.9513274336283186, 'weighted_f1': 0.951835857546077}
    EPOCH 7
    =================================
    Training Results
    {'acc': 0.9876, 'mean_precision': 0.9804630124660259, 'mean_recall': 0.9841335871628117, 'macro_f1': 0.9822722690431217, 'weighted_precision': 0.9876262764090674, 'weighted_recall': 0.9876, 'weighted_f1': 0.9876091385818745}
    Validation Results
    {'acc': 0.9358407079646017, 'mean_precision': 0.869223467111762, 'mean_recall': 0.8845714614348155, 'macro_f1': 0.8760358467213519, 'weighted_precision': 0.9368478438194213, 'weighted_recall': 0.9358407079646017, 'weighted_f1': 0.935724397414086}
    EPOCH 8
    =================================
    Training Results
    {'acc': 0.9882, 'mean_precision': 0.9884659041101195, 'mean_recall': 0.9867900044940896, 'macro_f1': 0.9876206331590174, 'weighted_precision': 0.9881936208229168, 'weighted_recall': 0.9882, 'weighted_f1': 0.9881954635511713}
    Validation Results
    {'acc': 0.9358407079646017, 'mean_precision': 0.8653465960956793, 'mean_recall': 0.881882140940112, 'macro_f1': 0.8719854535151271, 'weighted_precision': 0.9377869080790039, 'weighted_recall': 0.9358407079646017, 'weighted_f1': 0.9363262436397539}
    EPOCH 9
    =================================
    Training Results
    {'acc': 0.9948, 'mean_precision': 0.9938950329200654, 'mean_recall': 0.995900483358696, 'macro_f1': 0.9948909466726953, 'weighted_precision': 0.9948017988410376, 'weighted_recall': 0.9948, 'weighted_f1': 0.9947999511094636}
    Validation Results
    {'acc': 0.9535398230088495, 'mean_precision': 0.9109100734862207, 'mean_recall': 0.8984192710900785, 'macro_f1': 0.9035579458427662, 'weighted_precision': 0.9539677301282238, 'weighted_recall': 0.9535398230088495, 'weighted_f1': 0.9530178469339025}
    New best model 0.95
    EPOCH 10
    =================================
    Training Results
    {'acc': 0.9954, 'mean_precision': 0.995972032266384, 'mean_recall': 0.99609205194803, 'macro_f1': 0.9960309853717155, 'weighted_precision': 0.9954031919811314, 'weighted_recall': 0.9954, 'weighted_f1': 0.9954004886293786}
    Validation Results
    {'acc': 0.9380530973451328, 'mean_precision': 0.8994923889440322, 'mean_recall': 0.8819619949625506, 'macro_f1': 0.8895000197629823, 'weighted_precision': 0.9389177796726109, 'weighted_recall': 0.9380530973451328, 'weighted_f1': 0.9377173424408575}
    EPOCH 11
    =================================
    Training Results
    {'acc': 0.9928, 'mean_precision': 0.9940289451739184, 'mean_recall': 0.9938861281254447, 'macro_f1': 0.9939570832728007, 'weighted_precision': 0.992803497450866, 'weighted_recall': 0.9928, 'weighted_f1': 0.992801187319044}
    Validation Results
    {'acc': 0.9358407079646017, 'mean_precision': 0.8802438200074015, 'mean_recall': 0.8827911745872409, 'macro_f1': 0.8809691508186295, 'weighted_precision': 0.9370109437460447, 'weighted_recall': 0.9358407079646017, 'weighted_f1': 0.9356818495736176}
    EPOCH 12
    =================================
    Training Results
    {'acc': 0.9964, 'mean_precision': 0.9968305108450592, 'mean_recall': 0.9968291018285336, 'macro_f1': 0.996829742211563, 'weighted_precision': 0.9964006985237956, 'weighted_recall': 0.9964, 'weighted_f1': 0.9964002616562497}
    Validation Results
    {'acc': 0.9402654867256637, 'mean_precision': 0.8849039337406327, 'mean_recall': 0.8855626535491959, 'macro_f1': 0.8850671576152163, 'weighted_precision': 0.9401396828339421, 'weighted_recall': 0.9402654867256637, 'weighted_f1': 0.9400061909300451}
    Final result
    {'acc': 0.968, 'mean_precision': 0.9599401688044865, 'mean_recall': 0.9541044368804091, 'macro_f1': 0.9566602649716377, 'weighted_precision': 0.9685049225387307, 'weighted_recall': 0.968, 'weighted_f1': 0.967793480357041}





    0.968



We can see that this is a *massive* gain over our CNN baseline and also improves over our ELMo contextual embeddings for this dataset.  BERT has been shown high-performance results across many datasets, and integrating it into unstructured prediction problems is quite simple, as we saw in this section.

## Conclusion

In this section we investigated the Transformer model architecture, particularly in the context of pretraining LMs.  We discussed some of the model details and we looked at how BERT extends the GPT approach from OpenAI.  We then built our own fine-tuned classifier using the Hugging Face PyTorch library to create and re-load the BERT model and add our own layers on top.

### Some further resources

We have only scratched the surface of the exciting way that transfer learning is transforming NLP. 


- **Transformer Architecture**
  - [The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html): mentioned previously, but so good it deserves mentioning again
  - [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/):  good tutorial on how the Transformer works
-  A really nice blogpost on transfer learning from Sebastian Ruder (http://ruder.io/nlp-imagenet/)

- **Transfer Learning**
  - A [fantastic tutorial at NAACL this year](https://docs.google.com/presentation/d/1fIhGikFPnb7G5kr58OvYC3GN4io7MznnM0aAgadvJfc/edit) which is both thorough and introductory.  It covers a lot of material including how to probe pretrain models to try and figure out what they are up to
  - A nice colab from the Google BERT devs showing using BERT from TF-Hub (https://colab.research.google.com/github/google-research/bert/blob/master/predicting_movie_reviews_with_bert_on_tf_hub.ipynb)

- **Model Intepretation and Probing**
  - Jesse Vig's Blog post analyzing the different heads of BERT based
    - Part I: https://towardsdatascience.com/deconstructing-bert-distilling-6-patterns-from-100-million-parameters-b49113672f77
    -  Part II: https://towardsdatascience.com/deconstructing-bert-part-2-visualizing-the-inner-workings-of-attention-60a16d86b5c1
    - And a colab that drills into the [Q and K vectors during multi-head attention here](https://colab.research.google.com/drive/1Nlhh2vwlQdKleNMqpmLDBsAwrv_7NnrB): 
  - [Kevin Clark's Jupyter Notebooks](https://github.com/clarkkev/attention-analysis) for [What Does BERT Look At? An Analysis of BERT's Attention, Clark et al., 2019](https://arxiv.org/abs/1906.04341)
  - [Tal Linzen's code](https://github.com/TalLinzen/rnn_agreement) for [Assessing the ability of LSTMs to learn syntax-sensitive dependencies, Linzen et al., 2016](https://arxiv.org/abs/1611.01368)
  - [Yoav Goldberg's code](https://github.com/yoavg/bert-syntax) assessing syntactic abilities of BERT
  - [Nelson Liu's code](https://github.com/nelson-liu/contextual-repr-analysis) for [Linguistic Knowledge and Transferability of Contextual Representations, Liu et al., 2019](https://homes.cs.washington.edu/~nfliu/papers/liu+gardner+belinkov+peters+smith.naacl2019.pdf)

- **More about Neural NLP**
  -  Get right into the source material.  Some papers that are helpful to understand deep learning in NLP (https://github.com/dpressel/lit)

- **Get Hacking**
  - Implementations of most of what we talked about today in TensorFlow and PyTorch (https://github.com/dpressel/baseline)

There is also an end-to-end example using the Baseline API above to train a GPT-like LM using the code above in PyTorch:

https://github.com/dpressel/baseline/blob/master/api-examples/pretrain-transformer-lm.py





