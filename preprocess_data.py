from importlib.machinery import BYTECODE_SUFFIXES
from typing import List, Tuple
from transformers import BertTokenizer
import random
import argparse
import os
import json
# import transformers

def _read_wiki(file_name: str) -> List[List[str]]:
    file_name = os.path.join('wikitext-2', file_name)
    with open(file_name, 'r', encoding = 'utf-8') as f:
        lines = f.readlines()
    # Uppercase letters are converted to lowercase ones
    documents = [
        line.strip().lower().split(' . ') for line in lines
        if len(line.split(' . ')) >= 2]
    random.shuffle(documents)
    return documents

def document_to_segments(document: List[List[str]], max_segment_len: int = 128) -> List[Tuple[List[str], List[str]]]:
    max_seq_len = max_segment_len - 3 #accounting for the [CLS] and 2 [SEP] token


    current_len = 0
    current_chunk = [] #this is a list of list of words
    instances = [] # each item of this list will be a list of tokens of the form [CLS]..<tokens_a>..[SEP]..<tokens_b>..[SEP]
    i = 0
    while i < len(document):
        sent = document[i] #each sent is list of tokens(int)
        current_len += len(sent)
        current_chunk.append(sent)

        if current_len > max_seq_len or i == len(document) -1:
            if current_chunk:
                if len(current_chunk) == 1:
                    a_end = 1
                else:
                    a_end = random.randint(1, len(current_chunk)-1)

                tokens_a = []
                for j in range(a_end):
                    tokens_a.extend(current_chunk[j])

                tokens_b = []
                for j in range(a_end, len(current_chunk)):
                    tokens_b.extend(current_chunk[j])

                truncate_seq_pair(tokens_a, tokens_b, max_seq_len)


                instances.append((tokens_a, tokens_b))

                
                current_chunk = []
                current_len = 0
        i+= 1

    return instances

        

def truncate_seq_pair(tokens_a: List[int], tokens_b: List[int], max_num_tokens):
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_num_tokens:
            break

        larger_seg = tokens_a if len(tokens_a) > len(tokens_b) else tokens_b

        if random.random() < 0.5:
            larger_seg.pop(0)

        else:
            larger_seg.pop()

            

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--tokens_file', help = "Location of tokens file which contains raw text data")
    parser.add_argument('--json_file', help = "path to json file to which the output needs to be written")
    parser.add_argument('--max_num_segments', type = int, default= 1000, help = "maximum no of segments to write to json file")
    parser.add_argument('--max_segment_length', type = int, default=  128, help = "maximum length of an input segment")

    args = parser.parse_args()

    documents = _read_wiki(args.tokens_file)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    a_segments = []
    b_segments = []
    for document in documents:
        document = [tokenizer.tokenize(sent) for sent in document] #tokenize every line of a given document
        instances = document_to_segments(document, max_segment_len= args.max_segment_length)

        for instance in instances:
            if len(a_segments) < args.max_num_segments:  
                a_segments.append(instance[0])
                if instance[1]:
                    b_segments.append(instance[1])
                else: b_segments.append([''])
            else: break

        if len(a_segments) >= args.max_num_segments:
            break



    with open(args.json_file, 'w') as file:
        json.dump({'list_of_first_sentences': a_segments, 'list_of_second_sentences': b_segments}, file)

if __name__ == '__main__':
    main()
    
        




