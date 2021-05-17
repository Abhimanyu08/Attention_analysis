import argparse
import json

def read_conll(in_file, file_name, lowercase=False, max_example=None):
    examples = []
    with open(in_file) as f:
        word, pos, head, label = [], [], [], []
        for line in f.readlines():
            sp = line.strip().split('\t')
            if len(sp) == 10:
                if '-' not in sp[0]:
                    word.append(sp[1].lower() if lowercase else sp[1])
                    pos.append(sp[4])
                    head.append(int(sp[6]))
                    label.append(sp[7])
            elif len(word) > 0:
                examples.append({'word': word, 'pos': pos, 'head': head, 'label': label})
                word, pos, head, label = [], [], [], []
                if (max_example is not None) and (len(examples) == max_example):
                    break
        if len(word) > 0:
            examples.append({'word': word, 'pos': pos, 'head': head, 'label': label})

    with open(file_name, 'w') as file:
        json.dump(examples, file)
    



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data_path", type = str, help = "path to the training parser data")
    parser.add_argument("--dev_data_path", type = str, help = "path to the valid parser data")
    parser.add_argument("--train_json_name", type = str, help = "name of the file in which to save the training data")
    parser.add_argument("--dev_json_name", type = str, help = "name of the file in which to save the dev data")
    args = parser.parse_args()

    read_conll(args.train_data_path, args.train_json_name)
    read_conll(args.dev_data_path, args.dev_json_name)


if __name__ == "__main__":
    main()






