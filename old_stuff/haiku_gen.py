import torch
from torch.nn.parameter import Parameter
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPTNeoForCausalLM
from datetime import datetime
from torchmetrics import MeanMetric
from tqdm import tqdm
from fixed_summary_writer import FixedSummaryWriter
import random
from copy import deepcopy

def main():
    rng_seed = 42
    log_every = 1
    validate_every = 50
    n_steps = 500
    train_size = 0.99

    batch_size = 256
    sub_batch_size = 4
    accumulate_grad_steps = batch_size // sub_batch_size
    n_iterations = n_steps * accumulate_grad_steps
    validate_every *= accumulate_grad_steps

    lr = 0.3
    weight_decay = 0.0
    soft_prompt_length = 20

    model_name = 'EleutherAI/gpt-neo-1.3B' # 'distilgpt2', 'gpt2', 'gpt2-medium' 'EleutherAI/gpt-neo-1.3B'

    log_gen_words = [
        'sunshine',
        'machine',
        'heavenly',
        'lust',
        'pierce',
        'gloom',
        'havoc',
        'melody',
        'twitter',
    ]

    hparam_dict = {
        'batch_size': batch_size,
        'lr': lr,
        'weight_decay': weight_decay,
        'soft_prompt_length': soft_prompt_length,
    }

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = GPTNeoForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True)
    #d_model = model.config.n_embd
    d_model = model.config.hidden_size
    # extend embedding matrix to be capable of containing soft prompt tokens
    model._modules['transformer']._modules['wte'].weight = Parameter(torch.cat((
        model._modules['transformer']._modules['wte'].weight,
        torch.zeros((soft_prompt_length, d_model))
    )))
    model.to(device)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    data = HaikuDataModule(tokenizer, sub_batch_size, train_size=train_size)

    soft_prompt = init_uniform_soft_prompt(soft_prompt_length, d_model, device)
    optimizer = torch.optim.Adam((soft_prompt,), lr=lr, weight_decay=weight_decay)

    train_loss = MeanMetric().to(device)
    val_loss = MeanMetric().to(device)

    start_time = datetime.now().strftime('%y-%m-%d_%H-%M-%S')
    logger = FixedSummaryWriter(log_dir=f'logs/haiku/{start_time}')

    train_iterloader = iter(data.train_loader())
    for current_iteration in tqdm(range(n_iterations), desc='Training', ncols=160):
        # Train
        try:
            batch= next(train_iterloader)
        except:
            train_iterloader = iter(data.train_loader())
            batch = next(train_iterloader)
        batch['input_ids'] = batch['input_ids'].to(device)
        batch['labels'] = batch['labels'].to(device)
        batch = prep_batch(model, soft_prompt, batch, device)
        # print()
        # print(batch)
        # exit()
        output = model(inputs_embeds=batch['input_embeddings'], labels=batch['labels'])
        loss = output.loss
        train_loss(loss)

        if (current_iteration + 1) % (log_every * accumulate_grad_steps) == 0:
            logger.add_scalar('train/loss', train_loss.compute(), current_iteration)
            train_loss.reset()
        loss /= accumulate_grad_steps
        loss.backward()
        if (current_iteration + 1) % accumulate_grad_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        # Validate
        if (current_iteration % validate_every) == (validate_every - 1):
            val_loss.reset()
            for batch in tqdm(data.val_loader(), desc=f'Validation', leave=False, ncols=160):
                batch['input_ids'] = batch['input_ids'].to(device)
                batch['labels'] = batch['labels'].to(device)
                batch = prep_batch(model, soft_prompt, batch, device)
                with torch.no_grad():
                    output = model(inputs_embeds=batch['input_embeddings'])
                    logits_t = output.logits.transpose(1, 2)
                    loss = torch.nn.functional.cross_entropy(logits_t, batch['labels'], ignore_index=-100)
                    val_loss(loss)
            logger.add_hparams(
                hparam_dict=hparam_dict,
                metric_dict={'val/loss': val_loss.compute()},
                global_step=current_iteration,
            )
            write_haiku(log_gen_words, tokenizer, soft_prompt, model)
            examples = {'prompt': [], 'haiku': []}
            for i in range(len(batch['prompt'])):
                examples['prompt'].append(batch['prompt'][i])
                examples['haiku'].append(batch['haiku'][i])
            logger.add_text('examples', pd.DataFrame(examples).to_markdown(maxcolwidths=[None, 150, None]), current_iteration)
    logger.flush()
    logger.close()

def init_uniform_soft_prompt(soft_prompt_length, d_model, device):
    soft_prompt = torch.rand(size=(soft_prompt_length, d_model), device=device) * 2. - 1.
    soft_prompt.requires_grad = True
    return soft_prompt

def prep_batch(model, soft_prompt, batch, device):
    '''Prepend soft prompt, snip last token off input_ids, and fix labels'''
    word_text_embedding_layer = model._modules['transformer']._modules['wte']
    sub_batch_size = batch['input_ids'].shape[0]
    batch['input_embeddings'] = torch.concat((torch.broadcast_to(soft_prompt, size=(sub_batch_size,)+soft_prompt.shape), word_text_embedding_layer(batch['input_ids'])), dim=1)
    batch['labels'] = torch.concat((torch.full(size=(sub_batch_size, soft_prompt.shape[0]), fill_value=-100, device=device), batch['labels']), dim=1)
    return batch

def write_haiku(word_list, tokenizer, soft_prompt, model):
    word_list = deepcopy(word_list)
    seed_words = []
    for _ in range(random.randint(1, 3)):
        word = random.choice(word_list)
        seed_words.append(word)
        word_list.remove(word)

    nl = '\n'
    prompt = f'Words:{nl}{", ".join(seed_words)}{nl}Haiku:{nl}'
    prompt_token_ids = tokenizer(prompt, return_tensors='pt', max_length=256, truncation=True).input_ids[0]

    # replace last 20 embeddings with our soft prompt embeddings
    model._modules['transformer']._modules['wte'].weight.data[-len(soft_prompt):, :] = soft_prompt
    # get index of first soft prompt token
    soft_prompt_start = model._modules['transformer']._modules['wte'].weight.shape[0] - len(soft_prompt)
    # prepend soft prompt token ids to prompt tokens
    soft_prompt_token_ids = torch.arange(soft_prompt_start, soft_prompt_start + len(soft_prompt), dtype=torch.int64)
    prompt_token_ids = torch.cat((soft_prompt_token_ids, prompt_token_ids)).unsqueeze(0)
    # generate stuff!
    prompt_token_ids = prompt_token_ids.to(model.device)
    output_ids = model.generate(prompt_token_ids, pad_token_id=50256, max_new_tokens=30)
    # clip off soft prompt ids
    output_ids = output_ids[0][len(soft_prompt):]
    # decode generated stuff
    output_s = tokenizer.decode(output_ids)
    print()
    print(output_s)

    return output_s


class HaikuSoftPromptDataset(Dataset):
    def __init__(self, tokenizer, ds):
        super().__init__()
        self.ds = ds
        self.tokenizer = tokenizer

    def __getitem__(self, index):
        original_sample = self.ds[index].replace(' $', '')

        # get some words
        seed_words = []
        candidate_words = original_sample.replace(' /', '').split(' ')
        for _ in range(random.randint(1, 3)):
            word = random.choice(candidate_words)
            seed_words.append(word)
            candidate_words.remove(word)
        seed_words = ', '.join(seed_words)

        # put haiku on mulitple lines
        nl = '\n'
        haiku = original_sample.replace(' / ', '-')

        # tokenize
        token_budget = 64
        prompt = f'Words:{nl}{seed_words}{nl}Haiku:{nl}'
        haiku_token_ids = self.tokenizer(haiku, return_tensors='np').input_ids[0]
        prompt_token_ids = self.tokenizer(prompt, return_tensors='np', max_length=256, truncation=True).input_ids[0]
        n_pad_tokens = token_budget - (haiku_token_ids.shape[0] + prompt_token_ids.shape[0])
        transformed_sample = {
            'input_ids': np.concatenate((prompt_token_ids, haiku_token_ids, np.full(shape=(n_pad_tokens,), fill_value=self.tokenizer.pad_token_id))),
            'labels': np.concatenate((np.full(shape=(prompt_token_ids.shape), fill_value=-100), haiku_token_ids, np.full(shape=(n_pad_tokens,), fill_value=-100))),
            'prompt': prompt,
            'haiku': haiku,
        }
        return transformed_sample

    def __len__(self):
        return len(self.ds)


class HaikuDataModule:
    def __init__(self, tokenizer, sub_batch_size, num_workers=0, train_size=0.8):
        self.tokenizer = tokenizer
        self.batch_size = sub_batch_size
        self.num_workers = num_workers

        with open("haikus.txt") as file:
            lines = file.readlines()

        n_train_samples = int(train_size * len(lines))
        train_lines = lines[:n_train_samples]
        val_lines = lines[n_train_samples:]

        self.train_ds = HaikuSoftPromptDataset(self.tokenizer, train_lines)
        self.val_ds = HaikuSoftPromptDataset(self.tokenizer, val_lines)

    def train_loader(self):
        return DataLoader(self.train_ds, self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_loader(self):
        return DataLoader(self.val_ds, self.batch_size, num_workers=self.num_workers, shuffle=True)

if __name__ == '__main__':
    main()