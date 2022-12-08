from __future__ import annotations
import torch
from torch import tensor, Tensor
from torch import device as Device
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, GPTNeoForCausalLM
from torchmetrics import Accuracy, MeanMetric
from tqdm import tqdm, trange
from itertools import cycle
from dataclasses import dataclass
from typing import List, Tuple, Iterable, Union
import random

def main():
    soft_prompt_len = 10
    n_iterations = 200
    validate_every = 50
    batch_size = 32
    sub_batch_size = 8
    lr=0.001
    weight_decay = 0.
    use_vocab = True
    model_name = [
        'EleutherAI/gpt-neo-125M',
        'EleutherAI/gpt-neo-1.3B',
        'EleutherAI/gpt-neo-2.7B',
    ][0]

    if batch_size % sub_batch_size != 0:
        raise ValueError("batch_size must be an integer multiple of sub_batch_size")
    grad_accumulation_steps = batch_size // sub_batch_size

    device = Device('cuda:0') if torch.cuda.is_available() else Device('cpu')

    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    training_dataloader, validation_dataloader = haiku_dataloaders(tokenizer, batch_size=4, num_workers=0)
    model = GPTNeoForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True).to(device)
    soft_prompt = initialize_soft_prompt(model, soft_prompt_len, use_vocab=use_vocab).to(device)
    optimizer = torch.optim.Adam((soft_prompt,), lr=lr, weight_decay=weight_decay)

    train_loss = MeanMetric().to(device)
    val_loss = MeanMetric().to(device)
    train_accuracy = Accuracy(ignore_index=-100).to(device)
    val_accuracy = Accuracy(ignore_index=-100).to(device)

    training_iterator = cycle(training_dataloader)
    for current_iteration in trange(n_iterations):
        train()
        if current_iteration % validate_every == validate_every - 1:
            validate()

    batch = next(iter(training_dataloader))
    print(batch)
    print(batch[3])
    print(batch[1:3])

def haiku_dataloaders(tokenizer, batch_size: int, num_workers: int, train_size: float=0.8) -> Tuple[DataLoader, DataLoader]:
    """Sets up haiku training and validation dataloaders"""
    with open("haikus.txt") as file:
        haikus = file.readlines()
    random.shuffle(haikus)

    n_training_samples = int(train_size * len(haikus))
    training_haikus = haikus[:n_training_samples]
    validation_haikus = haikus[n_training_samples:]

    training_dataset = HaikuDataset(tokenizer, training_haikus)
    validation_dataset = HaikuDataset(tokenizer, validation_haikus)
    training_dataloader = DataLoader(training_dataset, batch_size=batch_size, num_workers=num_workers, collate_fn=training_dataset.collate)
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, num_workers=num_workers, collate_fn=training_dataset.collate)

    return training_dataloader, validation_dataloader

def initialize_soft_prompt(lm, seq_len: int=20, use_vocab=False):
    if use_vocab:
        indices = torch.randperm(lm.get_input_embeddings().weight.shape[0])[:seq_len]
        soft_prompt = lm.get_input_embeddings().weight[indices].clone().detach()
    else:
        embedding_dim = lm.config.hidden_size
        soft_prompt = torch.distributions.uniform.Uniform(-1., 1.).sample((seq_len, embedding_dim))
    soft_prompt.requires_grad = True
    return soft_prompt

def train():
    pass

def validate():
    pass

@dataclass(order=True, frozen=True)
class HaikuData:
    """A batch or sample of haiku data"""
    haikus: List[str]
    prompts: List[str]
    input_ids: Tensor
    labels: Tensor

    def __getitem__(self, index: Union[int, slice]) -> HaikuData:
        if isinstance(index, int):
            index = slice(index, index + 1)
        return HaikuData(
            haikus = self.haikus[index],
            prompts = self.prompts[index],
            input_ids = self.input_ids[index],
            labels = self.labels[index],
        )

    def __len__(self) -> int:
        return len(self.haikus)

    def to(self, device: Union[Device, str]):
        self.input_ids.to(device)
        self.labels.to(device)

    def __post_init__(self):
        if len(self.prompts) != len(self.haikus):
            raise ValueError("Length of prompts must match length of haikus")
        if len(self.input_ids) != len(self.haikus):
            raise ValueError("Length of input_ids must match length of haikus")
        if len(self.labels) != len(self.haikus):
            raise ValueError("Length of labels must match length of haikus")


class HaikuDataset(Dataset):
    """Produces samples of HaikuData from a list of haikus"""
    def __init__(self, tokenizer, haikus: Iterable[str]):
        super().__init__()
        self.haikus = haikus
        self.tokenizer = tokenizer

    def __getitem__(self, index) -> HaikuData:
        newline = "\n"
        pad_token = self.tokenizer.pad_token_id
        max_length = 64 + 1 # 1 gets snipped off later
        null_label = -100

        haiku = self.haikus[index].replace(" $", "").replace(" / ", "\n")
        seed_words = self.pick_seed_words(haiku)
        prompt = f"Words: {seed_words}{newline}Haiku:{newline}"

        prompt_token_ids = self.tokenizer(prompt, return_tensors='pt').input_ids
        haiku_token_ids = self.tokenizer(haiku, return_tensors='pt').input_ids
        n_prompt_tokens = prompt_token_ids.shape[1]
        n_haiku_tokens = haiku_token_ids.shape[1]

        input_ids = torch.concatenate((
            prompt_token_ids,
            haiku_token_ids,
            torch.full(size=(1, max_length - n_prompt_tokens - n_haiku_tokens), fill_value=pad_token),
        ), dim=1)
        labels = torch.concatenate((
            torch.full(size=(1, n_prompt_tokens), fill_value=null_label),
            haiku_token_ids,
            torch.full(size=(1, max_length - n_prompt_tokens - n_haiku_tokens), fill_value=null_label),
        ), dim=1)

        return HaikuData([haiku], [prompt], input_ids, labels)


    def pick_seed_words(self, haiku: str) -> str:
        candidate_words = haiku.replace("\n", " ").split(" ")
        seed_words = []
        for _ in range(random.randint(1, 3)):
            random_word = random.choice(candidate_words)
            seed_words.append(random_word)
            candidate_words.remove(random_word)
        seed_words = ', '.join(seed_words)
        return seed_words

    def __len__(self):
        return len(self.haikus)

    def collate(self, samples: List[HaikuData]) -> HaikuData:
        return HaikuData(
            haikus = [sample.haikus[0] for sample in samples],
            prompts = [sample.prompts[0] for sample in samples],
            input_ids = torch.concatenate([sample.input_ids for sample in samples]),
            labels = torch.concatenate([sample.labels for sample in samples]),
        )

if __name__ == "__main__":
    main()