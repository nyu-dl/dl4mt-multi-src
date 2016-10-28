# Multi-Source Neural Machine Translation

In this repository we implement Multi-Source Neural Machine Translation
(msNMT), in particular Firat et al.,2016 (EMNLP'16) (partial) and Zoph and
Knight, 2016 (NAACL'16) (partial).

We present three scenarios that can be encountered in msNMT, but in general
any Sequence-to-Sequence problem that involves many-to-one mapping.  

1. Multi-text available only at test

2. Multi-text available only at training

3. Multi-text available both at training and test

But of course, first we need multi-text (n-way parallel sentences) for our
training development and test sets. We will be using
`English`-`French`-`Spanish` corpora from
[Europarl-v7](http://www.statmt.org/europarl/) for training and
[newstest2011-2012](http://www.statmt.org/wmt13/dev.tgz) for development and
test.<br>

Simply follow the steps below to download and preprocess all the data.

```bash
$ cd dl4mt-multi-src/data
$ ./prepare_data.sh
```

which will first retrieve the necessary data (in forms of bi-text), compose
multi-text, tokenize, encode using Byte Pair Encoding
([BPE](https://arxiv.org/abs/1508.07909)) and extract vocabularies. 

Please see `data/prepare_data.sh` for details.<br>


Some Requirements: 
------------------

This repo is part of `dl4mt` habitat and extends `dl4mt-multi`, so same
[dependencies](https://github.com/nyu-dl/dl4mt-multi#dependencies) apply. 

You may consider taking a look at
[`setup.sh`](https://github.com/nyu-dl/dl4mt-multi/blob/master/setup.sh) script
for setting up your environment.<br>


Running the models:
-------------------

As mentioned above, we present three different scenarios for msNMT.<br>
For demonstration purposes, we restrict the number of source sequences to 2 but 
of course, you can increase it as long as you have multi-text and enough GPU
memory :wink: <br>

### 1. Multi-text available only at test

In the first scenario, we will first train a multi-encoder NMT model by using
`Spanish`-`English` and `French`-`English` **bi-texts only**. Note that, a
multi-encoder NMT has multiple encoders (for `English` and `French` in our
case) and a single decoder (`English`) that is shared across both translation
directions.<br>

After the model is trained with **bi-texts**, we will test (decode) the model
using **multi-text**, practically by giving both sources at the same time.<br>

Note: This scenario necessitates a non-parametric merger-operator (eg. mean).
<br>

Training
```bash
$ export THEANO_FLAGS=device=gpu,floatX=float32
$ python dl4mt-multi-src/train_mlnmt.py --proto=get_config_EsFr2En_single
```
Decoding (in a bash script)
```{r, engine='bash', count_lines}
#!/bin/bash 

src_es2en=dl4mt-multi-src/data/dev/newstest2012.es.tok.bpe20k
src_fr2en=dl4mt-multi-src/data/dev/newstest2012.fr.tok.bpe20k

ref_file=dl4mt-multi-src/data/dev/newstest2012.en.tok
out_file=translation.esfr2en.out.early

# translate from spanish and french
export THEANO_FLAGS=device=cpu,floatX=float32
python dl4mt-multi-src/translate.py \
    --num-process 8 \
    --beam-size 7 \
    --cgs-to-translate "model0:es.fr_en" \
    --source-files "es.fr_en:fr=${src_fr2en}@es=${src_es2en}" \
    --output-file ${out_file} \
    --gold-file ${ref_file} \
    --bleu-script dl4mt-multi-src/data/multi-bleu.perl \
    --changes "init_merger_op:mean,attend_merge_op:mean" \
    --protos get_config_EsFr2En_single \
    --models esfr2en_single/params.npz
```

### 2. Multi-text available only at training

In the second scenario, we will again train a multi-encoder NMT model, but
this time, only by using **multi-text** (you should observe faster convergence
compared to using bi-texts only.)<br>

Although nothing prevents us to use **multi-text** at test time, we will use
only **bi-texts** at test time.<br>

Note: If you use a parametric gated merger-operator and also use multi-text at
test time, this scenario is very much similar to Zoph and Knight, 2016.<br>

Training
```bash
$ export THEANO_FLAGS=device=gpu,floatX=float32
$ python dl4mt-multi-src/train_mlnmt.py --proto=get_config_EsFr2En_mSrc
```

Decoding (in a bash script)
```{r, engine='bash', count_lines}
#!/bin/bash

src_es2en=dl4mt-multi-src/data/dev/newstest2012.es.tok.bpe20k
src_fr2en=dl4mt-multi-src/data/dev/newstest2012.fr.tok.bpe20k

ref_file=dl4mt-multi-src/data/dev/newstest2012.en.tok
out_file_es2en=translation.es2en.out.single
out_file_fr2en=translation.fr2en.out.single

# translate from spanish
export THEANO_FLAGS=device=cpu,floatX=float32 
python dl4mt-multi-src/translate.py \
    --num-process 8 \
    --beam-size 7 \
    --cgs-to-translate "model0:es_en" \
    --source-files "es_en:${src_es2en}" \
    --output-file ${out_file_es2en} \
    --gold-file ${ref_file} \
    --bleu-script dl4mt-multi-src/data/multi-bleu.perl \
    --protos get_config_EsFr2En_mSrc \
    --models esfr2en_mSrc/params.npz

# translate from french
export THEANO_FLAGS=device=cpu,floatX=float32
python dl4mt-multi-src/translate.py \
    --num-process 8 \
    --beam-size 7 \
    --cgs-to-translate "model0:fr_en" \
    --source-files "fr_en:${src_fr2en}" \
    --output-file ${out_file_fr2en} \
    --gold-file ${ref_file} \
    --bleu-script dl4mt-multi-src/data/multi-bleu.perl \
    --protos get_config_EsFr2En_mSrc \
    --models esfr2en_mSrc/params.npz
```

### 3. Multi-text available both at training and test

Finally, the last scenario combines first and second scenarios and we will be using 
both **bi-texts** and **multi-text** during training. <br>

At test time, we will feed the model again with both **bi-texts** and **multi-text**
which in turn enables us to compute an ensemble of all three outputs
(`Spanish`-`English` + `French`-`English` + `Spanish+French`-`English`)<br>

Training
```bash
$ export THEANO_FLAGS=device=gpu,floatX=float32
$ python dl4mt-multi-src/train_mlnmt.py --proto=get_config_EsFr2En_single_and_mSrc
```
Decoding (in a bash script)
```{r, engine='bash', count_lines}
#!/bin/bash 

src_es2en=dl4mt-multi-src/data/dev/newstest2012.es.tok.bpe20k
src_fr2en=dl4mt-multi-src/data/dev/newstest2012.fr.tok.bpe20k

ref_file=dl4mt-multi-src/data/dev/newstest2012.en.tok
out_file=translation.esfr2en.out.late

# translate from spanish and french
export THEANO_FLAGS=device=cpu,floatX=float32
python dl4mt-multi-src/translate.py \
    --num-process 8 \
    --beam-size 7 \
    --cgs-to-translate "model0:es.fr_en,es_en,fr_en" \
    --source-files "es.fr_en:fr=${src_fr2en}@es=${src_es2en},es_en:${src_es2en},fr_en:${src_fr2en}" \
    --output-file ${out_file} \
    --gold-file ${ref_file} \
    --bleu-script dl4mt-multi-src/data/multi-bleu.perl \
    --changes "init_merger_op:mean,attend_merge_op:mean" \
    --protos get_config_EsFr2En_single_and_mSrc \
    --models esfr2en_single/params.npz
```

Merger-Operators:
-----------------

The choice of merger-operator is crucial in msNMT and should be decided
according to the task requirements. Depending on your task, you may need to
blend (combine) multiple sources or you may consider using a mechanism that
chooses `n` sources among `m` multiple sources where `n` < `m`. <br>

[Here](https://github.com/nyu-dl/dl4mt-multi-src/blob/master/mcg/layers.py#L429-L491)
we implement two sets of merger-operators.

1. Non-parametric (arithmetic): mean, sum, max

2. Parametric (by a neural network): softplus gated feed-forward net, attentive 
merger (a second level of attention).<br>

Merger-operators are needed in two places:

1. Merging the information for decoder initializer network.

2. Merging the context vectors coming from multiple sources.<br>

You can change the merger-operators along with an additional non-linearity
from the model configurations:<br>
```python
# decoder initializer merger
config['init_merge_op'] = 'mean'
config['init_merge_act'] = 'tanh'

# post-attention merger
config['attend_merge_op'] = 'mean'
config['attend_merge_act'] = 'tanh'
```
<br>

## References

"Zero-Resource Translation with Multi-Lingual Neural Machine Translation" <br>
Orhan Firat, Baskaran Sankaran, Yaser Al-Onaizan, Fatos Vural and Kyunghyun Cho<br>
EMNLP 2016.<br>

"Multi-Source Neural Machine Translation"<br>
Barret Zoph and Kevin Knight<br>
NAACL-HLT 2016.
