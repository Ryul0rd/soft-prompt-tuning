import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, GPTNeoForCausalLM, GPTJForCausalLM, GPT2LMHeadModel
from torchmetrics import Accuracy, MeanMetric
from tqdm import tqdm
import random
import os
import numpy as np

def main():
    # hyperparameters
    soft_prompt_len = 50
    n_iters = 200
    validate_every = n_iters
    batch_size = 32
    sub_batch_size = 16
    lr=0.2
    weight_decay = 0.0
    use_vocab = True
    model_name = [
        'distilgpt2',
        'EleutherAI/gpt-neo-125M',
        'EleutherAI/gpt-neo-1.3B',
        'EleutherAI/gpt-neo-2.7B',
        'EleutherAI/gpt-j-6B',
        # 'gpt2',
        # 'gpt2-medium', # gpt2 works differently for some reason
    ][0]
    data_class = [
        ToySeqData,
        ToyCategoryData,
        ToyOverfitSentenceData,
    ][1]

    assert batch_size % sub_batch_size == 0
    grad_accumulation_steps = batch_size // sub_batch_size

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # model init
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    if 'gpt-neo' in model_name:
        lm = GPTNeoForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True).to(device)
    elif 'gpt-j' in model_name:
        lm = GPTJForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True).to(device)
    else:
        lm = GPT2LMHeadModel.from_pretrained(model_name, low_cpu_mem_usage=True).to(device)

    # soft prompt and optimizer init
    soft_prompt = initialize_soft_prompt(lm, soft_prompt_len, use_vocab=use_vocab).to(device)
    optimizer = torch.optim.Adam((soft_prompt,), lr=lr, weight_decay=weight_decay)

    # data init
    train_dl = DataLoader(
        data_class(tokenizer, 10000),
        batch_size=sub_batch_size,
        num_workers=os.cpu_count(),
        shuffle=True,
        drop_last=True
    )
    val_dl = DataLoader(
        data_class(tokenizer, 1000),
        batch_size=sub_batch_size,
        num_workers=os.cpu_count()
    )

    # init metrics
    train_loss = MeanMetric().to(device)
    val_loss = MeanMetric().to(device)
    train_accuracy = Accuracy(ignore_index=-100).to(device)
    val_accuracy = Accuracy(ignore_index=-100).to(device)

    # training loop
    train_iter = DataIterator(train_dl, device)
    for current_iter in tqdm(range(n_iters)):
        # train
        lm.train()
        for _ in range(grad_accumulation_steps):
            batch = train_iter.next()
            output = soft_prompted_lm(batch, soft_prompt, lm)
            loss = output.loss / grad_accumulation_steps
            loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # validate
        if current_iter % validate_every == validate_every - 1:
            lm.eval()
            val_loss.reset()
            val_accuracy.reset()
            for batch in tqdm(val_dl):
                output = soft_prompted_lm(batch, soft_prompt, lm)
                loss = output.loss
                val_loss(loss)
                val_accuracy(output.logits[:, :-1].transpose(1, 2), batch['labels'][:, 1:])

    # show results
    n_examples = 3
    print(output.logits.shape)
    for i in range(n_examples):
        print(f'Example {i+1}:')
        print(f'Input Text: {batch["input_text"][i]}')
        print(f'Label Text: {batch["label_text"][i]}')
        print(f'Prediction: {tokenizer.decode(torch.argmax(output.logits[i], dim=1)[batch["n_input_tokens"][i]+soft_prompt_len-1])}')
    #print(train_loss.compute())
    print(f'Validation Loss: {val_loss.compute()}')
    #print(train_accuracy.compute())
    print(f'Validation Accuracy: {val_accuracy.compute()}')

def initialize_soft_prompt(lm, seq_len: int=20, use_vocab=False):
    if use_vocab:
        indices = torch.randperm(lm.get_input_embeddings().weight.shape[0])[:seq_len]
        soft_prompt = lm.get_input_embeddings().weight[indices].clone().detach()
    else:
        embedding_dim = lm.config.hidden_size
        soft_prompt = torch.distributions.uniform.Uniform(-1., 1.).sample((seq_len, embedding_dim))
    soft_prompt.requires_grad = True
    return soft_prompt

def prep_batch(batch, soft_prompt, lm):
    word_to_embedding_layer = lm.get_input_embeddings()
    sub_batch_size = batch['input_ids'].shape[0]

    # embed each input and attach the soft prompt
    word_embeddings = word_to_embedding_layer(batch['input_ids'])
    soft_prompts = torch.broadcast_to(soft_prompt, size=(sub_batch_size,)+soft_prompt.shape)
    batch['input_embeddings'] = torch.concat((soft_prompts, word_embeddings), dim=1)
    null_label = -100
    soft_prompt_padding = torch.full(size=soft_prompts.shape[:2], fill_value=null_label, device=batch['labels'].device)
    batch['labels'] = torch.concat((soft_prompt_padding, batch['labels']), dim=1)

    return batch

def to_device(batch, device):
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(device)
    return batch

def soft_prompted_lm(batch, soft_prompt, lm):
    batch = to_device(batch, soft_prompt.device)
    batch = prep_batch(batch, soft_prompt, lm)
    output = lm(inputs_embeds=batch['input_embeddings'], labels=batch['labels'])
    return output

class DataIterator:
    def __init__(self, dataloader: DataLoader, device):
        self.dataloader = dataloader
        self.iterloader = iter(self.dataloader)
        self.device = device

    def __next__(self):
        return self.next()

    def next(self):
        try:
            batch = next(self.iterloader)
        except:
            self.iterloader = iter(self.dataloader)
            batch = next(self.iterloader)
        batch = to_device(batch, self.device)
        return batch


class ToyData(Dataset):
    def __init__(self, tokenizer, size: int):
        self.tokenizer = tokenizer
        self.data = [self.tokenize_sample(**self.create_sample()) for _ in range(size)]

    def create_sample(self):
        sample = {
            'input_text': 'a',
            'label_text': 'b',
        }
        return sample

    def tokenize_sample(self, input_text: str, label_text: str):
        input_ids = self.tokenizer(
            input_text,
            return_tensors='np',
        ).input_ids[0]
        label_ids = self.tokenizer(
            label_text,
            return_tensors='np',
        ).input_ids[0]

        # if we have n input tokens and l max length, input tokens should be: n input tokens then l - n pad tokens
        pad_token = self.tokenizer.pad_token_id
        null_label = -100
        max_length = 32 + 1 # 1 gets snipped off later
        n_input_tokens = len(input_ids)
        n_label_tokens = len(label_ids)
        input_ids = np.concatenate((
            input_ids,
            label_ids,
            np.full(shape=(max_length - n_input_tokens - n_label_tokens,), fill_value=pad_token),
        ))
        # and if we have m label tokens, our labels should be: n nulls then m label tokens then l - m - n nulls
        label_ids = np.concatenate((
            np.full(shape=(n_input_tokens,), fill_value=null_label),
            label_ids,
            np.full(shape=(max_length - n_input_tokens - n_label_tokens,), fill_value=null_label),
        ))

        sample = {
            'input_text': input_text,
            'label_text': label_text,
            'input_ids': input_ids,
            'labels': label_ids,
            'n_input_tokens': n_input_tokens,
        }
        return sample

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class ToySeqData(ToyData):
    def __init__(self, tokenizer, size: int):
        super().__init__(tokenizer, size)

    def create_sample(self):
        '''Find the symbol with the most instances'''
        symbols = ['X', 'O']
        min_symbols = 3
        max_symbols = 25

        valid_seq = False
        while not valid_seq:
            symbol_seq = random.choices(symbols, k=random.randint(min_symbols, max_symbols))
            symbol_counts = {symbol: 0 for symbol in symbols}
            for symbol in symbol_seq:
                symbol_counts[symbol] += 1
            count_vals = sorted(list(symbol_counts.values()), reverse=True)
            if count_vals[0] != count_vals[1]:
                valid_seq = True

        most_frequent_symbol = None
        most_frequent_symbol_count = -1
        for symbol, count in symbol_counts.items():
            if count > most_frequent_symbol_count:
                most_frequent_symbol_count = count
                most_frequent_symbol = symbol
        label_text = ' ' + most_frequent_symbol

        input_text = ''.join([' ' + symbol for symbol in symbol_seq])

        # testing the addition of part descriptors
        #input_text = 'Symbols:' + input_text + '\nMost Common Symbol:'
        
        sample = {
            'input_text': input_text,
            'label_text': label_text,
        }
        return sample


class ToyCategoryData(ToyData):
    def __init__(self, tokenizer, size: int):
        super().__init__(tokenizer, size)

    def create_sample(self):
        '''Find the odd one out'''
        categories = {
            'animals': ['cat', 'dog', 'bird', 'lizard', 'dinosaur', 'fish', 'cow', 'chicken', 'bug'],
            'plants': ['tree', 'grass', 'vines', 'shrub', 'bush'],
            'elements': ['hydrogen', 'helium', 'carbon', 'nitrogen', 'oxygen', 'sulfur', 'phosphorous', 'sulfur', 'uranium'],
            'countries': ['United States', 'Canada', 'Mexico', 'France', 'England', 'Germany', 'Japan', 'China', 'India'],
            'names': ['Ryan', 'Devon', 'Lynn', 'Brandon', 'Hayden', 'Bill', 'Mary', 'Bob', 'Alice', 'Charlie', 'Nathan'],
            'subjects': ['biology', 'chemistry', 'physics', 'computer science', 'math', 'philosophy', 'history', 'sociology'],
        }
        categories = [category_items for category_name, category_items in categories.items()]
        min_items = 3
        max_items = 10

        n_items = random.randint(min_items, max_items)

        odd_category_out = random.choice(categories)
        categories.remove(odd_category_out)
        majority_category = random.choice(categories)

        odd_item_out = random.choice(odd_category_out)
        majority_items = random.choices(majority_category, k=n_items-1)

        all_items = [odd_item_out] + majority_items
        random.shuffle(all_items)

        label_text = ' ' + odd_item_out
        input_text = ''.join([f' {item},' for item in all_items])[:-1]

        # testing the addition of part descriptors
        input_text = input_text + '\nOdd One Out:'
        
        sample = {
            'input_text': input_text,
            'label_text': label_text,
        }
        return sample


class ToyOverfitSentenceData(ToyData):
    def __init__(self, tokenizer, size: int):
        super().__init__(tokenizer, size)

    def create_sample(self):
        '''Just produce the sentence'''
        sentences = [
            ' This is the first test sentence!',
        ]

        sample = {
            'input_text': '!',
            'label_text': random.choice(sentences),
        }
        return sample

if __name__ == '__main__':
    main()
