from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import torch
import pandas as pd
import random
import math

class PileDataset(Dataset):
    def __init__(self, 
                 path_to_data, 
                 sequence_length, 
                 use_UL2: bool = False, 
                 ul2_percentage: float | None = None):
        # Load json file with pandas
        self.data = pd.read_json(path_to_data, lines=True)
        
        if use_UL2 and ul2_percentage is not None:
            self.data = self.data.sample(frac=ul2_percentage, random_state=1)

        self.length = len(self.data)

        if use_UL2:
            # Create a list the length of the data, where each element is a string
            # representing the type of the data (S2S, NLU, NLG), divided so that
            # S2S is randomly 50% of the data, and NLU and NLG are 25% each
            # round up to the nearest integer
            self.data_types = ['[S2S]'] * math.ceil(self.length / 2) + ['[NLU]'] * math.ceil(self.length / 4) + ['[NLG]'] * math.ceil(self.length / 4)
            # Shuffle the data_types list
            random.shuffle(self.data_types)

        self.sequence_length = sequence_length

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")

        print('Adding new tokens to tokenizer...')

        new_tokens = ['[S2S]', '[NLU]', '[NLG]', '<PAD>']
        
        for i in range(100):
            new_tokens.append(f'<extra_id_{i}>')

        # Add new tokens to tokenizer
        num_added_tokens = self.tokenizer.add_tokens(new_tokens)

        print(f'Added {num_added_tokens} new tokens to tokenizer')

        # Specify padding token
        self.tokenizer.pad_token = '<PAD>'

        # Add padding token to tokenizer
        self.pad_tok = self.tokenizer.pad_token_id

        self.vocab_size = len(self.tokenizer)

        self.use_UL2 = use_UL2

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
        
        if len(item) > self.get_block_size():
            # trim item to 3/4 of the block size
            item = item[:int(self.get_block_size() * 3/4)]

        prefix_index = None

        if self.use_UL2:
            type = self.data_types[idx]
            item, prefix_index = self.SpanCorrupt(item, type)

            if prefix_index >= self.get_block_size():
                print('prefix_index: ', prefix_index)
                print('block_size: ', self.get_block_size())
                # throw error
                raise ValueError('Prefix index is greater than or equal to block size')
            
        new_length = len(item)

        if new_length < self.get_block_size():
            x = item[:new_length-1]
            y = item[1:new_length]
        else:
            x = item[:self.get_block_size()]
            y = item[1:self.get_block_size()+1]
        
        return x, y, prefix_index
    
    def pad_to_longest(self, batch):
    
        x, y, prefix_index = zip(*batch)

        x_lens = [len(s) for s in x]
        pad_len = max(x_lens)

        pad_x = [s + [self.pad_tok] * (pad_len - len(s)) for s in x]

        y_lens = [len(s) for s in y]
        pad_len = max(y_lens)
        pad_y = [s + [self.pad_tok] * (pad_len - len(s)) for s in y]

        pad_x = torch.tensor(pad_x)
        pad_y = torch.tensor(pad_y)

        return pad_x, pad_y, prefix_index
    
    def SpanCorrupt(self, input_tokens, type):
        # Type is either:
        # '[S2S]' - PrefixLM - S-Denoiser
        # '[NLU]' - R-Denoiser
        # '[NLG]' - X-Denoiser

        if type == '[NLU]':
            """
            R-Denoiser
            Uniformly sample spans with a mean of 3 and a corruption rate of 15%
            mu = 3
            r = 0.15
            n = L * r / mu
            """
            mu = 3
            r = 0.15
            # Calculate the number of spans to corrupt
            # round up to the nearest integer
            n = int(math.ceil(len(input_tokens) * r / mu))
        elif type == '[NLG]':
            """
            X-Denoiser
            Randomly sample spans with a mean of 32 OR a corruption rate of up to 50%
            mu = 32 | 3
            r = 0.50 | 0.15
            n = L / mu
            """
            # randomly select a mean of 32 or 3
            if random.random() < 0.5:
                mu = 3
            else:
                mu = 32
            r = 0.50
            # Calculate the number of spans to corrupt
            n = int(math.ceil(len(input_tokens) * r / mu))
        elif type == '[S2S]':
            """
            S-Denoiser
            Randomly select a point in the text
            """
            # Randomly pick an index in input_tokens
            p = random.randint(1, len(input_tokens))
            mu = len(input_tokens) - p
            r = 1.0 - p / len(input_tokens)
            n = 1
        else:
            raise ValueError('Invalid type')

        selected_spans = []
        # Randomly select n spans
        for _ in range(int(n)):
            # sample length from a normal distribution with mean mu and standard deviation 1
            length = int(random.gauss(mu, 1))
            # Randomly select a span
            start = random.randint(1, len(input_tokens))
            if type == '[S2S]':
                end = len(input_tokens)
            else:
                end = start + length
            # Ensure the span does not overlap with any other spans
            while any(start < s < end for s, e in selected_spans) or any(start < e < end for s, e in selected_spans):
                start = random.randint(1, len(input_tokens))
                end = start + length
            selected_spans.append((start, end))
        
        # order spans by start index
        selected_spans = sorted(selected_spans, key=lambda x: x[0])

        target_tokens = []

        new_input_tokens = input_tokens.copy()

        # Create new input, moving backwards through list to not mess up indices of spans
        for i in range(len(selected_spans)-1, -1, -1):
            start, end = selected_spans[i]
            id_token = self.tokenizer.convert_tokens_to_ids([f'<extra_id_{i}>'])[0]
            new_input_tokens = new_input_tokens[:start] + [id_token] + new_input_tokens[end:]

        # Create target
        for i, span in enumerate(selected_spans):
            start, end = span
            id_token = self.tokenizer.convert_tokens_to_ids([f'<extra_id_{i}>'])[0]
            target_tokens.append(id_token)
            for token in input_tokens[start:end]:
                target_tokens.append(token)

        # Add type token
        type_token = self.tokenizer.convert_tokens_to_ids([type])[0]

        new_src_tokens = [type_token] + new_input_tokens 
        new_target_tokens = [self.tokenizer.pad_token_id] + target_tokens
        prefix_index = len(new_src_tokens)
        final_tokens = new_src_tokens + new_target_tokens
        # print("prefix_index: ", prefix_index)

        # print('input_tokens: ', self.tokenizer.decode(input_tokens))
        # decode new_input_tokens to check if it looks correct
        # print('new_input_tokens:', self.tokenizer.decode(final_tokens))

        return final_tokens, prefix_index


