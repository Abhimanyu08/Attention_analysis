from transformers import BertModel, BertTokenizer
import argparse
import json
from torch.utils.data import DataLoader
import torch
import pickle

class Dataset:
    def __init__(self, first_ls, second_ls):
        self.first_ls = first_ls
        self.second_ls = second_ls

    def __getitem__(self, i): return (self.first_ls[i], self.second_ls[i])

    def __len__(self): return len(self.first_ls)

def collate_fn(samples): 
    first_sents, second_sents = zip(*samples)
    return list(first_sents), list(second_sents)

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--json_file", type = str, help = "path to json files where list of sentences are stored")
    parser.add_argument("--batch_size", type = int, default= 4, help = "size of batch")
    parser.add_argument("--max_segment_length", type = int, default= 128, help = "maximum length of the segment")
    parser.add_argument("--pickle_name", type = str, help = "name of the pickle file to save attention maps in")


    args = parser.parse_args()

    dic = None
    with open(args.json_file, 'r') as file:
        dic = json.load(file)

    list_of_first_sentences = dic['list_of_first_sentences']
    list_of_second_sentences = dic['list_of_second_sentences']

    ds = Dataset(list_of_first_sentences,  list_of_second_sentences)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')

    dl = DataLoader(dataset= ds, batch_size= args.batch_size, collate_fn= collate_fn)

    model.eval()
    feature_dicts = []
    for sentences in dl:
        tok_out = tokenizer(*sentences, is_split_into_words= True, padding= 'max_length', max_length= args.max_segment_length, truncation = True, return_tensors= 'pt')
        out = model(**tok_out, output_attentions = True)
        attns = torch.stack(out.attentions, dim = 0).transpose(0,1) #attns are of shape (batch_size, num_layers, num_heads, seq_len, seq_len)
        

        num_elem = tok_out['input_ids'].size(0)
        for j in range(num_elem):
            ids= tok_out['input_ids'][j]
            try:
                seq_len = next(i for i in range(len(ids)) if ids[i] == tokenizer.pad_token_id)
            except:
                seq_len = args.max_segment_length
            ids = ids[:seq_len]
            tokens = tokenizer.convert_ids_to_tokens(ids)
            attn = attns[j, :, :, :seq_len, :seq_len].detach()
            feature_dicts.append({'ids':ids, 'tokens':tokens, 'attention_map': attn})

    with open(args.pickle_name, 'wb') as file:
        pickle.dump(feature_dicts, file, -1)


    
if __name__ == "__main__":
    main()





