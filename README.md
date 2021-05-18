# Attention_analysis
Analysis of Bert's attention heads using huggingface transformers

Unofficial Pytorch implementation of paper [What Does BERT Look At? An Analysis of BERT's Attention](https://arxiv.org/abs/1906.04341) using [HuggingFace Transformers](https://huggingface.co/transformers/).

## Overwiew

BERT-base has 12 layers and each layer has 12 attention heads. Each of the attention heads is of shape `(sequence_length, sequence_length)`. The attentions of whole network can by captured in a four dimensional tensor `attentions_map` of shape `(12,12,sequence_length, sequence_length)`. `attentions_map[i][j][k][l]` tells us the amount of attention k'th word of sequence pays to l'th word in i'th layer and j'th head. This paper sets out to find interesting patterns of attention each of 144 heads exhibit. 

## General Analysis:
