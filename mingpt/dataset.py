from torch.utils.data import Dataset, IterableDataset, DataLoader, get_worker_info
import dask
dask.config.set({"dataframe.query-planning": True})
import dask.dataframe as dd
from torch.distributed import get_rank, get_world_size
from transformers import AutoTokenizer
import torch
import pandas as pd

class PileDataset(Dataset):
    def __init__(self, path_to_data, sequence_length):
        # Load json file with pandas
        self.data = pd.read_json(path_to_data, lines=True)

        self.length = len(self.data)

        self.sequence_length = sequence_length

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        # Add padding token to tokenizer
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.pad_tok = self.tokenizer.pad_token_id
        self.vocab_size = self.tokenizer.vocab_size

    def get_vocab_size(self):
        return self.vocab_size
    
    def get_block_size(self):
        # the length of the sequence that will feed into transformer, 
        # containing concatenated input and the output, but -1 because
        # the transformer starts making predictions at the last input element
        return self.sequence_length * 2 - 1

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        item = self.data.iloc[idx].text
        # Tokenize item
        item = self.tokenizer(item)['input_ids']
        
        if len(item) <= self.sequence_length:
            length = len(item)
            #item = np.append(item, self.eos_tok)
            x = item[:length-1]
            y = item[1:length]  
        else:
            x = item[:self.sequence_length]
            y = item[1:self.sequence_length+1]
        
        return x, y
    
    def pad_to_longest(self, batch):
        x, y = zip(*batch)

        x_lens = [len(s) for s in x]
        pad_len = max(x_lens)

        pad_x = [s + [self.pad_tok] * (pad_len - len(s)) for s in x]

        y_lens = [len(s) for s in y]
        pad_len = max(y_lens)
        pad_y = [s + [self.pad_tok] * (pad_len - len(s)) for s in y]

        pad_x = torch.tensor(pad_x)
        pad_y = torch.tensor(pad_y)

        return pad_x, pad_y
