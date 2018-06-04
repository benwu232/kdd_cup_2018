# Seq2Seq (with attention) solution for KDD CUP 2018

## KDD CUP 2018 Introduction

https://biendata.com/competition/kdd_2018/

## Progress
I formally participated this competition on about 23 April, which is too late. I used PyTorch to implemented some models base on Seq2Seq framework.

### Seq2Seq （no_attn branch)
This is the basic Seq2Seq framework, also the baseline. The encoder and decoder are GRUs, which has layers and hidden units as hyper parameters. The model use former N(e.g. 120, 240, 480, 720 ...) hours' air quality sequence to predict the next 48 hours data. For simplicity, the model only uses the air quality data of Beijing and London. I also use some timing features, such as month, the week of year, the day of week, the hour in the day, etc. I used this model as my major model till the firt half of the competion and found some fatal bugs, which caused some bad results.

## Seq2Seq with attention (attn_pos branch)
In the middle of the competition, I tried to add attention mechanism. First, I added time attention, which is the common attention on time dimention. This for sure improve the socre. After that, I tried another so called space attention, which actually pays attention to all native air quality stations. This one seems not as good as time attention. Later, I kept it as a hyper parameter in hyperopt. The most severe problem of attentions is they are really really slow. For average, they are 50x slower than the baseline model. Hence it is much slower to use that in hyperopt than use the baseline model.

## Something tried and failed
I also tried to use grid meterology data. I tried to calculate the meterology data of the air quality stations through interpolating grid data (seq2seq_grid branch). However, the result is bad. I'd like to know how other guys to use these data.

## Some lessons
1. Participating the competition as early as you can. For me, a novice, to make a stable model takes at least a month. 
2. A good coross validation framework is a must. I did bad this time. Seems my model overfitted when I use hyperopt.
3. Seems the data in this competion is a little too few for the models like RNN.

## Some philosophy thinking
I think a good model should be succint and beautiful. A model which piles all kinds of models are neither beautiful nor practical. I tried to build a unified model to solve various problems. I tried to use as less feature engineering as possible or make it automatic. Although it is a long way, it is a way to generic AI. 



