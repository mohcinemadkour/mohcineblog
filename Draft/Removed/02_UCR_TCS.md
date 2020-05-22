title: Practical Deep Learning for Time Series using fastai/ Pytorch: Part 2
Date: 2019-10-12 13:01
Category: Time Series Classification
Tags: Machine Learning, timeseriesAI, Time Series Classification, fastai_timeseries, TSC bechmark
Slug: Machine Learning, timeseriesAI, Time Series Classification, fastai_timeseries, TSC bechmark
Author: Mohcine Madkour
Email:mohcine.madkour@gmail.com

The UCR datasets are broadly used in TSC problems as s bechmark to measure performance. This notebook will allow you to test any of the available datasets, with the model of your choice and any training scheme. You can easily tweak any of them to try to beat a SOTA.














```python
import fastai, os
from fastai_timeseries import *
from torchtimeseries.models import *
path = Path(os.getcwd())
print(path)
print('fastai :', fastai.__version__)
print('torch  :', torch.__version__)
print('device :', device)
```


<style>.container { width:100% !important; }</style>


    /home/oguizadl/tradingLab⚗️
    fastai : 1.0.57
    torch  : 1.2.0
    device : cpu
    


```python
def run_UCR_test(iters, epochs, datasets, arch, 
                 bs=64, max_lr=3e-3, pct_start=.7, warmup=False, wd=1e-2, metrics=[accuracy], 
                 scale_type ='standardize', scale_subtype='per_channel', scale_range=(-1, 1), 
                 opt_func=functools.partial(torch.optim.Adam, betas=(0.9, 0.99)), 
                 loss_func=None, **arch_kwargs):
    ds_, acc_, acces_, accmax_, iter_, time_, epochs_, loss_, val_loss_   = [], [], [], [], [], [], [], [], []
    datasets = listify(datasets)
    for ds in datasets: 
        db = create_UCR_databunch(ds)
        for i in range(iters):
            print('\n', ds, i)
            ds_.append(ds)
            iter_.append(i)
            epochs_.append(epochs)
            model = arch(db.features, db.c, **arch_kwargs).to(defaults.device)
            learn = Learner(db, model, opt_func=opt_func, loss_func=loss_func)
            learn.metrics = metrics
            start_time = time.time()
            learn.fit_one_cycle(epochs, max_lr=max_lr, pct_start=pct_start, moms=(.95, .85) if warmup else (.95, .95),
                                div_factor=25.0 if warmup else 1., wd=wd)
            duration = time.time() - start_time
            time_.append('{:.0f}'.format(duration))
            early_stop = math.ceil(np.argmin(learn.recorder.losses) / len(learn.data.train_dl))
            acc_.append(learn.recorder.metrics[-1][0].item())
            acces_.append(learn.recorder.metrics[early_stop - 1][0].item())
            accmax_.append(np.max(learn.recorder.metrics))
            loss_.append(learn.recorder.losses[-1].item())
            val_loss_.append(learn.recorder.val_losses[-1].item())
            if len(datasets) * iters >1: clear_output()
            df = (pd.DataFrame(np.stack((ds_, iter_, epochs_, loss_, val_loss_ ,acc_, acces_, accmax_, time_)).T,
                               columns=['dataset', 'iter', 'epochs', 'loss', 'val_loss', 
                                        'accuracy', 'accuracy_ts', 
                                        'max_accuracy', 'time (s)'])
                  )
            df = df.astype({'loss': float, 'val_loss': float, 'accuracy': float, 
                            'accuracy_ts': float, 'max_accuracy': float})
            display(df)
    return learn, df
```


```python
# This is an unofficial PyTorch implementation by Ignacio Oguiza - oguiza@gmail.com based on:

# Fawaz, H. I., Lucas, B., Forestier, G., Pelletier, C., Schmidt, D. F., Weber, J., ... & Petitjean, F. (2019). InceptionTime: Finding AlexNet for Time Series Classification. arXiv preprint arXiv:1909.04939.
# Official InceptionTime tensorflow implementation: https://github.com/hfawaz/InceptionTime

import torch
import torch.nn as nn

def noop(x):
    return x

def shortcut(c_in, c_out):
    return nn.Sequential(*[nn.Conv1d(c_in, c_out, kernel_size=1), 
                           nn.BatchNorm1d(c_out)])
    
class Inception(nn.Module):
    def __init__(self, c_in, bottleneck=32, ks=40, nb_filters=32):

        super().__init__()
        self.bottleneck = nn.Conv1d(c_in, bottleneck, 1) if bottleneck and c_in > 1 else noop
        mts_feat = bottleneck or c_in
        conv_layers = []
        kss = [ks // (2**i) for i in range(3)]
        # ensure odd kss until nn.Conv1d with padding='same' is available in pytorch 1.3
        kss = [ksi if ksi % 2 != 0 else ksi - 1 for ksi in kss]  
        for i in range(len(kss)):
            conv_layers.append(
                nn.Conv1d(mts_feat, nb_filters, kernel_size=kss[i], padding=kss[i] // 2))
        self.conv_layers = nn.ModuleList(conv_layers)
        self.maxpool = nn.MaxPool1d(3, stride=1, padding=1)
        self.conv = nn.Conv1d(c_in, nb_filters, kernel_size=1)
        self.bn = nn.BatchNorm1d(nb_filters * 4)
        self.act = nn.ReLU()

    def forward(self, x):
        input_tensor = x
        x = self.bottleneck(input_tensor)
        for i in range(3):
            out_ = self.conv_layers[i](x)
            if i == 0: out = out_
            else: out = torch.cat((out, out_), 1)
        mp = self.conv(self.maxpool(input_tensor))
        inc_out = torch.cat((out, mp), 1)
        return self.act(self.bn(inc_out))


class InceptionBlock(nn.Module):
    def __init__(self,c_in,bottleneck=32,ks=40,nb_filters=32,residual=True,depth=6):

        super().__init__()

        self.residual = residual
        self.depth = depth

        #inception & residual layers
        inc_mods = []
        res_layers = []
        res = 0
        for d in range(depth):
            inc_mods.append(
                Inception(c_in if d == 0 else nb_filters * 4, bottleneck=bottleneck if d > 0 else 0,ks=ks,
                          nb_filters=nb_filters))
            if self.residual and d % 3 == 2:
                res_layers.append(shortcut(c_in if res == 0 else nb_filters * 4, nb_filters * 4))
                res += 1
            else: res_layer = res_layers.append(None)
        self.inc_mods = nn.ModuleList(inc_mods)
        self.res_layers = nn.ModuleList(res_layers)
        self.act = nn.ReLU()
        
    def forward(self, x):
        res = x
        for d, l in enumerate(range(self.depth)):
            x = self.inc_mods[d](x)
            if self.residual and d % 3 == 2:
                res = self.res_layers[d](res)
                x += res
                res = x
                x = self.act(x)
        return x
    
class InceptionTime(nn.Module):
    def __init__(self,c_in,c_out,bottleneck=32,ks=40,nb_filters=32,residual=True,depth=6):
        super().__init__()
        self.block = InceptionBlock(c_in,bottleneck=bottleneck,ks=ks,nb_filters=nb_filters,
                                    residual=residual,depth=depth)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(nb_filters * 4, c_out)

    def forward(self, x):
        x = self.block(x)
        x = self.gap(x).squeeze(-1)
        x = self.fc(x)
        return x
```


```python
# Data
bottom10 = [ 'Wine', 'BeetleFly',  #'CinCECGtorso', not available for download 
            'InlineSkate', 'MiddlePhalanxTW', 'OliveOil', 'SmallKitchenAppliances', 'WordSynonyms', 
            'MiddlePhalanxOutlineAgeGroup', 'MoteStrain', 'Phoneme']
top3 = ['Herring', 'ScreenType', 'ChlorineConcentration']
datasets = bottom10 + top3
bs = 64
scale_type = 'standardize'
scale_subtype = 'per_channel'
scale_range = (-1, 1)

# Arch
arch = InceptionTime
arch_kwargs = dict()

# Training
iters = 1
epochs = 500
max_lr = 3e-3
warmup = False
pct_start = .7
metrics = [accuracy]
wd = 1e-2
opt_func = Ranger
loss_func = LabelSmoothingCrossEntropy()
```


```python
output = run_UCR_test(iters,
                      epochs,
                      datasets,
                      arch,
                      bs=bs,
                      max_lr=max_lr,
                      pct_start=pct_start,
                      warmup=warmup,
                      wd=wd,
                      metrics=metrics,
                      scale_type=scale_type,
                      scale_subtype=scale_subtype,
                      scale_range=scale_range,
                      opt_func=opt_func,
                      loss_func=loss_func,
                      **arch_kwargs)
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
      <th>dataset</th>
      <th>iter</th>
      <th>epochs</th>
      <th>loss</th>
      <th>val_loss</th>
      <th>accuracy</th>
      <th>accuracy_ts</th>
      <th>max_accuracy</th>
      <th>time (s)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>Wine</td>
      <td>0</td>
      <td>1</td>
      <td>0.692424</td>
      <td>0.693793</td>
      <td>0.500000</td>
      <td>0.500000</td>
      <td>0.500000</td>
      <td>2</td>
    </tr>
    <tr>
      <td>1</td>
      <td>BeetleFly</td>
      <td>0</td>
      <td>1</td>
      <td>0.737083</td>
      <td>0.697668</td>
      <td>0.500000</td>
      <td>0.500000</td>
      <td>0.500000</td>
      <td>2</td>
    </tr>
    <tr>
      <td>2</td>
      <td>InlineSkate</td>
      <td>0</td>
      <td>1</td>
      <td>1.961044</td>
      <td>1.951494</td>
      <td>0.152727</td>
      <td>0.152727</td>
      <td>0.152727</td>
      <td>42</td>
    </tr>
    <tr>
      <td>3</td>
      <td>MiddlePhalanxTW</td>
      <td>0</td>
      <td>1</td>
      <td>1.589655</td>
      <td>1.749170</td>
      <td>0.272727</td>
      <td>0.272727</td>
      <td>0.272727</td>
      <td>5</td>
    </tr>
    <tr>
      <td>4</td>
      <td>OliveOil</td>
      <td>0</td>
      <td>1</td>
      <td>1.667427</td>
      <td>1.428959</td>
      <td>0.166667</td>
      <td>0.166667</td>
      <td>0.166667</td>
      <td>3</td>
    </tr>
    <tr>
      <td>5</td>
      <td>SmallKitchenAppliances</td>
      <td>0</td>
      <td>1</td>
      <td>1.135667</td>
      <td>1.109895</td>
      <td>0.333333</td>
      <td>0.333333</td>
      <td>0.333333</td>
      <td>26</td>
    </tr>
    <tr>
      <td>6</td>
      <td>WordSynonyms</td>
      <td>0</td>
      <td>1</td>
      <td>3.382129</td>
      <td>3.268570</td>
      <td>0.021944</td>
      <td>0.021944</td>
      <td>0.021944</td>
      <td>12</td>
    </tr>
    <tr>
      <td>7</td>
      <td>MiddlePhalanxOutlineAgeGroup</td>
      <td>0</td>
      <td>1</td>
      <td>1.000210</td>
      <td>1.080597</td>
      <td>0.571429</td>
      <td>0.571429</td>
      <td>0.571429</td>
      <td>5</td>
    </tr>
    <tr>
      <td>8</td>
      <td>MoteStrain</td>
      <td>0</td>
      <td>1</td>
      <td>0.709574</td>
      <td>0.690208</td>
      <td>0.539137</td>
      <td>0.539137</td>
      <td>0.539137</td>
      <td>4</td>
    </tr>
    <tr>
      <td>9</td>
      <td>Phoneme</td>
      <td>0</td>
      <td>1</td>
      <td>3.737337</td>
      <td>3.657671</td>
      <td>0.004747</td>
      <td>0.004747</td>
      <td>0.004747</td>
      <td>72</td>
    </tr>
    <tr>
      <td>10</td>
      <td>Herring</td>
      <td>0</td>
      <td>1</td>
      <td>0.862592</td>
      <td>0.729897</td>
      <td>0.406250</td>
      <td>0.406250</td>
      <td>0.406250</td>
      <td>6</td>
    </tr>
    <tr>
      <td>11</td>
      <td>ScreenType</td>
      <td>0</td>
      <td>1</td>
      <td>1.117127</td>
      <td>1.100862</td>
      <td>0.333333</td>
      <td>0.333333</td>
      <td>0.333333</td>
      <td>27</td>
    </tr>
    <tr>
      <td>12</td>
      <td>ChlorineConcentration</td>
      <td>0</td>
      <td>1</td>
      <td>1.032640</td>
      <td>1.065673</td>
      <td>0.532552</td>
      <td>0.532552</td>
      <td>0.532552</td>
      <td>25</td>
    </tr>
  </tbody>
</table>
</div>



```python

```
