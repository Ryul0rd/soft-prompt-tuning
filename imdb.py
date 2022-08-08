import torch
import numpy as np
import pandas as pd
import optuna
from optuna.samplers import TPESampler
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from datetime import datetime
from torchmetrics import Accuracy, MeanMetric
from tqdm import tqdm
from fixed_summary_writer import FixedSummaryWriter

def main():
    study = optuna.create_study(sampler=TPESampler(), direction='maximize')
    study.optimize(objective, n_trials=1)

def objective(trial):
    rng_seed = 42
    train_size = 2048
    val_size = 100
    log_every = 1
    validate_every = 10
    log_n_closest_words = 3
    n_steps = 100

    batch_size = 32 #trial.suggest_categorical('batch_size', [2, 4, 6, 8]) # 4
    sub_batch_size = 4
    accumulate_grad_steps = batch_size // sub_batch_size
    n_iterations = n_steps * accumulate_grad_steps

    lr = 1e1 #trial.suggest_float('lr', 1e-1, 1e2, log=True) # 1e1 ?
    weight_decay = 0. #trial.suggest_float('weight_decay', 1e-4, 1e-2, log=True) # 6e-3?
    soft_prompt_length = 32 #trial.suggest_categorical('soft_prompt_length', [1, 2, 4, 8, 16, 24, 32]) # 4

    #word_list = ['movie', 'sentiment', 'classify', 'positive', 'negative'] * 4
    word_list = ['positive', 'negative'] * 2
    hparam_dict = {
        'batch_size': batch_size,
        'lr': lr,
        'weight_decay': weight_decay,
        'soft_prompt_length': soft_prompt_length,
    }

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = GPT2LMHeadModel.from_pretrained('distilgpt2', low_cpu_mem_usage=True)
    model.to(device)
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    embedding_matrix = model._modules['transformer']._modules['wte'].weight
    data = IMDBDataModule(tokenizer, sub_batch_size, train_size=train_size, val_size=val_size)

    d_model = model.config.n_embd
    soft_prompt = init_soft_prompt(soft_prompt_length, d_model, device, tokenizer, embedding_matrix, strategy='uniform', word_list=word_list)
    optimizer = torch.optim.Adam((soft_prompt,), lr=lr, weight_decay=weight_decay)

    train_loss = MeanMetric().to(device)
    val_loss = MeanMetric().to(device)
    train_accuracy = Accuracy(ignore_index=-100).to(device)
    val_accuracy = Accuracy(ignore_index=-100).to(device)

    start_time = datetime.now().strftime('%y-%m-%d_%H-%M-%S')
    logger = FixedSummaryWriter(log_dir=f'logs/soft-prompt-tuning/{start_time}')

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
        output = model(inputs_embeds=batch['input_embeddings'], labels=batch['labels']).logits.transpose(1, 2)
        loss = torch.nn.functional.cross_entropy(output, batch['labels'], ignore_index=-100)
        train_loss(loss)
        train_accuracy(output, batch['labels'])

        if (current_iteration + 1) % (log_every * accumulate_grad_steps) == 0:
            logger.add_scalar('train/loss', train_loss.compute(), current_iteration)
            logger.add_scalar('train/accuracy', train_accuracy.compute(), current_iteration)
            train_loss.reset()
            train_accuracy.reset()
        loss /= accumulate_grad_steps
        loss.backward()
        if (current_iteration + 1) % accumulate_grad_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        # Validate
        if (current_iteration % validate_every) == (validate_every - 1):
            val_loss.reset()
            val_accuracy.reset()
            for batch in tqdm(data.val_loader(), desc=f'Validation', leave=False, ncols=160):
                batch['input_ids'] = batch['input_ids'].to(device)
                batch['labels'] = batch['labels'].to(device)
                batch = prep_batch(model, soft_prompt, batch, device)
                with torch.no_grad():
                    output = model(inputs_embeds=batch['input_embeddings'])
                    logits_t = output.logits.transpose(1, 2)
                    loss = torch.nn.functional.cross_entropy(logits_t, batch['labels'], ignore_index=-100)
                    val_loss(loss)
                    val_accuracy(logits_t, batch['labels'])
            logger.add_hparams(
                hparam_dict=hparam_dict,
                metric_dict={'val/accuracy': val_accuracy.compute(), 'val/loss': val_loss.compute()},
                global_step=current_iteration,
            )
            nearest_prompt_words = {}
            for index, word in enumerate(soft_prompt):
                nearest_prompt_words[f'word {index+1}'] = [tokenizer.decode(word_idx) for word_idx in k_nearest_words(embedding_matrix, word, manhattan_distance, k=log_n_closest_words)]
            logger.add_text('nearest prompt words', pd.DataFrame(nearest_prompt_words).to_markdown(), current_iteration)
            examples = {'output': [], 'prediction': [], 'label': []}
            for i in range(len(batch['text_label'])):
                examples['output'].append(batch['text'][i])
                prediction_index = (batch['labels'][i] != -100).nonzero(as_tuple=True)[0]
                examples['prediction'].append(tokenizer.decode(torch.argmax(output.logits[i, prediction_index])))
                examples['label'].append(batch['text_label'][i])
            logger.add_text('examples', pd.DataFrame(examples).to_markdown(maxcolwidths=[None, 160, None, None]), current_iteration)
    logger.flush()
    logger.close()
    return val_accuracy.compute()

class IMDBSoftPromptDataset(Dataset):
    def __init__(self, tokenizer, ds):
        super().__init__()
        self.ds = ds
        self.tokenizer = tokenizer
        print(f"Negatives: {len(self.ds.filter(lambda sample: sample['label'] == 0))}")
        print(f"Positives: {len(self.ds.filter(lambda sample: sample['label'] == 1))}")

    def __getitem__(self, index):
        original_sample = self.ds[index]

        token_budget = 288
        text_label = ' positive' if original_sample['label'] else ' negative'
        nl = '\n'
        text = f'{original_sample["text"]}{nl}Sentiment:'
        label_token_ids = self.tokenizer(text_label, return_tensors='np').input_ids[0]
        prompt_token_ids = self.tokenizer(text, return_tensors='np', max_length=256, truncation=True).input_ids[0]
        n_pad_tokens = token_budget - (label_token_ids.shape[0] + prompt_token_ids.shape[0])
        transformed_sample = {
            'input_ids': np.concatenate((prompt_token_ids, label_token_ids, np.full(shape=(n_pad_tokens,), fill_value=self.tokenizer.pad_token_id))),
            'labels': np.concatenate((np.full(shape=(prompt_token_ids.shape), fill_value=-100), label_token_ids, np.full(shape=(n_pad_tokens,), fill_value=-100))),
            'text': original_sample['text'],
            'text_label': text_label,
        }
        return transformed_sample

    def __len__(self):
        return len(self.ds)


class IMDBDataModule:
    def __init__(self, tokenizer, sub_batch_size, num_workers=0, train_size=20000, val_size=5000):
        self.tokenizer = tokenizer
        self.batch_size = sub_batch_size
        self.num_workers = num_workers

        train_val_ds = load_dataset('imdb', split='train').shuffle()
        self.train_ds = IMDBSoftPromptDataset(self.tokenizer, train_val_ds.select(range(val_size, len(train_val_ds))).select(range(train_size)))
        self.val_ds = IMDBSoftPromptDataset(self.tokenizer, train_val_ds.select(range(val_size)))
        self.test_ds = IMDBSoftPromptDataset(self.tokenizer, load_dataset('imdb', split='test'))

    def train_loader(self):
        return DataLoader(self.train_ds, self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_loader(self):
        return DataLoader(self.val_ds, self.batch_size, num_workers=self.num_workers, shuffle=True)

    def test_loader(self):
        return DataLoader(self.test_ds, self.batch_size, num_workers=self.num_workers)

def init_uniform_soft_prompt(soft_prompt_length, d_model, device):
    soft_prompt = torch.rand(size=(soft_prompt_length, d_model), device=device) * 2. - 1.
    soft_prompt.requires_grad = True
    return soft_prompt

def init_word_list_soft_prompt(soft_prompt_length, device, word_list, tokenizer, embedding_matrix):
    prompt = ''.join([f' {word}' for word in word_list])
    prompt_ids = tokenizer(prompt, return_tensors='pt').input_ids[0][:soft_prompt_length]
    soft_prompt = embedding_matrix[prompt_ids].detach()
    soft_prompt.to(device)
    soft_prompt.requires_grad = True
    return soft_prompt

def init_soft_prompt(soft_prompt_length, d_model, device, tokenizer, embedding_matrix, strategy='uniform', word_list=None):
    if strategy == 'uniform':
        return init_uniform_soft_prompt(soft_prompt_length, d_model, device)
    elif strategy == 'word list':
        assert word_list is not None
        return init_word_list_soft_prompt(soft_prompt_length, device, word_list, tokenizer, embedding_matrix)
    else:
        print('Please pick a valid strategy. Options are: \'uniform\' and \'vocab\'.')

def prep_batch(model, soft_prompt, batch, device):
    '''Prepend soft prompt, snip last token off input_ids, and fix labels'''
    word_text_embedding_layer = model._modules['transformer']._modules['wte']
    sub_batch_size = batch['input_ids'].shape[0]
    batch['input_embeddings'] = torch.concat((torch.broadcast_to(soft_prompt, size=(sub_batch_size,)+soft_prompt.shape), word_text_embedding_layer(batch['input_ids'])), dim=1)[:, :-1]
    batch['labels'] = torch.concat((torch.full(size=(sub_batch_size, soft_prompt.shape[0]), fill_value=-100, device=device), batch['labels']), dim=1)[:, 1:]
    return batch

def manhattan_distance(tensor1, tensor2):
    return torch.sum(torch.abs(tensor1 - tensor2), dim=-1)

def tensor_sort_by(tensor, key, descending=False):
    sorting_permutation = torch.argsort(key(tensor), descending=descending)
    return tensor[sorting_permutation]

def k_nearest_neighbors(neighbors, point, distance_fn, k=1):
    sorted_neighbors = tensor_sort_by(neighbors, lambda x: distance_fn(point, x))
    return sorted_neighbors[:k]

def k_nearest_words(embedding_matrix, soft_word, distance_fn, k=1):
    sorted_word_ids = torch.argsort(distance_fn(embedding_matrix, soft_word))
    return sorted_word_ids[:k]

if __name__ == '__main__':
    main()