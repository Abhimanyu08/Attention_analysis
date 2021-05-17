import json
from transformers import BertTokenizer ,BertModel
import argparse
from torch.utils.data import DataLoader
from functools import partial
import torch
import pickle
import numpy as np

def words_to_tokens(words, tokenizer):
    tokens_list = []
    for word in words:
        tokens_list.append(tokenizer.tokenize(word))
        
    indexes = []
    index = 0
    
    for tokens in tokens_list:
        ls = []
        for _ in tokens:
            ls.append(index)
            index += 1
        indexes.append(ls)
            
    return tokens_list, indexes

def token_to_word_attention(amap, token_lists, indexes):
    to_delete_indexes = []
    for i,token_list in enumerate(token_lists):
        if len(token_list) > 1:
            for index in indexes[i][1:]:
                amap[:,:,:,indexes[i][0]] += amap[:,:,:,index]
    
                amap[:,:,indexes[i][0],:] += amap[:,:,:,index]
        
                to_delete_indexes.append(index)
            
    amap = amap.numpy()
    amap = np.delete(amap, to_delete_indexes, axis = -1)
    amap = np.delete(amap, to_delete_indexes, axis = -2)
    
    amap /= amap.sum(-1, keepdims = True)
    
    return amap

class Dataset():
    def __init__(self, list_of_dicts):
        self.ls = list_of_dicts
        
    def __len__(self): return len(self.ls)
    
    def __getitem__(self, i): return self.ls[i]


def collate_fn(samples, tokenizer):
    batch = []
    for sample in samples:
        words = " ".join(word for word in sample['word'])
        batch.append(words)
        
    tok_out = tokenizer(batch, padding = True, truncation = True, return_tensors = 'pt')
    return tok_out, samples

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_file', type= str, help = "json file containing parsing data")
    parser.add_argument('--batch_size', type = int)
    parser.add_argument('--pickle_file', type= str, help= "pickle file in which to save the attention maps")

    args = parser.parse_args()

    list_of_examples = None
    with open(args.json_file, 'r') as file:
        list_of_examples = json.load(file)

    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    model = BertModel.from_pretrained('bert-base-cased')

    ds = Dataset(list_of_examples)
    dl = DataLoader(ds, batch_size= args.batch_size, collate_fn= partial(collate_fn, tokenizer = tokenizer))

    model.eval()
    model = model.to('cuda')
    feature_dicts = []
    for examples in dl:
        inputs = {i:j.to('cuda') for i,j in examples[0].items()}
        output = model(**inputs, output_attentions = True)
        attns = torch.stack(output.attentions, dim = 0).transpose(0,1).detach().to('cpu') #attns are of shape (batch_size, num_layers, num_heads, seq_len, seq_len)

        num_elems = attns.size(0)

        for j in range(num_elems):
            # dic = {}
            ids = inputs['input_ids'][j].to('cpu')
            try:
                seq_len = next(i for i in range(len(ids)) if ids[i] == tokenizer.pad_token_id)
            except:
                seq_len = attns.size(-1)

            words = [tokenizer.cls_token] + examples[1][j]['word'] + [tokenizer.sep_token]
            token_lists, indexes = words_to_tokens(words, tokenizer)

            attn = token_to_word_attention(attns[j, :,:,:seq_len,:seq_len], token_lists, indexes)


            dic = {"attention_map": attn, 'words': words, 'heads': examples[1][j]['head'], 'labels': examples[1][j]['label']}

            feature_dicts.append(dic)

    with open(args.pickle_file, 'wb') as file:
        pickle.dump(feature_dicts, file)


if __name__ == "__main__":
    main()

            
            
            

