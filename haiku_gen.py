from __future__ import annotations
import torch
from torch import tensor, Tensor
from torch import device as Device
from torch.utils.data import Dataset, DataLoader
from torch.optim import Optimizer, Adam
from transformers import (
    PreTrainedTokenizer,
    PreTrainedModel,
    GPT2Tokenizer,
    GPTNeoForCausalLM,
    StoppingCriteriaList,
    MaxLengthCriteria,
    BeamSearchScorer,
    LogitsProcessorList,
    MinLengthLogitsProcessor,
)
from torchmetrics import Accuracy, MeanMetric
from tqdm import tqdm, trange
from dataclasses import dataclass
from typing import List, Tuple, Iterable, Union, Optional
import random

def main():
    soft_prompt_len = 20
    n_iterations = 1000
    validate_every = 100
    batch_size = 32
    sub_batch_size = 8
    lr=0.001
    weight_decay = 0.
    use_vocab = True
    model_name = [
        'EleutherAI/gpt-neo-125M',
        'EleutherAI/gpt-neo-1.3B',
        'EleutherAI/gpt-neo-2.7B',
    ][1]

    if batch_size % sub_batch_size != 0:
        raise ValueError("batch_size must be an integer multiple of sub_batch_size")
    grad_accumulation_steps = batch_size // sub_batch_size

    device = Device('cuda:0') if torch.cuda.is_available() else Device('cpu')

    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    training_dataloader, validation_dataloader = haiku_dataloaders(tokenizer, batch_size=4, num_workers=0)
    model = GPTNeoForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True).to(device)
    soft_prompt = initialize_soft_prompt(model, soft_prompt_len, use_vocab=use_vocab).to(device)
    optimizer = Adam((soft_prompt,), lr=lr, weight_decay=weight_decay)

    # monkey patch model to work with input embeds
    model.prepare_inputs_for_generation = prepare_inputs_for_generation

    train_loss_metric = MeanMetric().to(device)
    validation_loss_metric = MeanMetric().to(device)

    training_iterator = DataIterator(training_dataloader)
    for current_iteration in trange(n_iterations):
        train_loss = train(
            soft_prompt,
            training_iterator,
            optimizer,
            model,
            grad_accumulation_steps,
            device,
            )
        train_loss_metric(train_loss)
        if current_iteration % validate_every == validate_every - 1:
            validation_loss = validate(
                soft_prompt,
                validation_dataloader,
                model,
                device,
                validation_loss_metric,
            )
            print()
            newline = "\n"
            print(f'Validation Loss: {validation_loss}')

            seed_words = [
                "beauty",
                "sunlight",
                "imagination",
                "programming",
                "electric",
            ]
            chosen_seed_words = []
            for _ in range(random.randint(2, 3)):
                chosen_word = random.choice(seed_words)
                chosen_seed_words.append(chosen_word)
                seed_words.remove(chosen_word)
            seed_words = chosen_seed_words

            print(f'Example:{newline}{beam_generate(seed_words, tokenizer, soft_prompt, model)}')

def train(
    soft_prompt: Tensor,
    training_iterator: DataIterator,
    optimizer: Optimizer,
    model: PreTrainedModel,
    grad_accumulation_steps: int,
    device: Device
    ):
    model.train()
    total_loss = 0
    for _ in range(grad_accumulation_steps):
        batch = next(training_iterator).to(device).prepend_soft_prompt(soft_prompt, model)
        output = model(inputs_embeds=batch.input, labels=batch.labels)
        loss = output.loss / grad_accumulation_steps
        loss.backward()
        total_loss += loss.detach()
    optimizer.step()
    optimizer.zero_grad()

    return total_loss

def validate(
    soft_prompt: Tensor,
    validation_dataloader: DataLoader,
    model: PreTrainedModel,
    device: Device,
    loss_metric: MeanMetric,
    ):
    loss_metric.reset()
    model.eval()
    with torch.no_grad():
        for batch in tqdm(validation_dataloader):
            batch = batch.to(device).prepend_soft_prompt(soft_prompt, model)
            output = model(inputs_embeds=batch.input, labels=batch.labels)
            loss = output.loss
            loss_metric(loss)
    
    return loss_metric.compute()

def greedy_generate(
    seed_words: List[str],
    tokenizer: PreTrainedTokenizer,
    soft_prompt: Tensor,
    model: PreTrainedModel,
    ):
    device = model.device
    newline = "\n"
    seed_words = ', '.join(seed_words)

    prompt = f"Words: {seed_words}{newline}Haiku:{newline}"
    prompt_token_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(device)

    word_to_embedding_layer = model.get_input_embeddings()
    word_embeddings = word_to_embedding_layer(prompt_token_ids)
    inputs_embeds = torch.concat((soft_prompt.unsqueeze(dim=0), word_embeddings), dim=1)

    input_ids = torch.LongTensor([[model.config.bos_token_id]]).to(device)
    stopping_criteria = StoppingCriteriaList([MaxLengthCriteria(max_length=30)])

    outputs = model.greedy_search(
        input_ids, inputs_embeds=inputs_embeds, stopping_criteria=stopping_criteria, pad_token_id=model.config.eos_token_id
    )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    prompt_and_generated_text = prompt + generated_text

    return prompt_and_generated_text

def beam_generate(
    seed_words: List[str],
    tokenizer: PreTrainedTokenizer,
    soft_prompt: Tensor,
    model: PreTrainedModel,
    temperature: float = 0.9
    ):
    device = model.device
    newline = "\n"
    seed_words = ', '.join(seed_words)

    prompt = f"Words: {seed_words}{newline}Haiku:{newline}"
    prompt_token_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(device)

    num_beams = 3

    prompt_token_ids = prompt_token_ids.broadcast_to((num_beams, prompt_token_ids.shape[1]))

    word_to_embedding_layer = model.get_input_embeddings()
    word_embeddings = word_to_embedding_layer(prompt_token_ids)
    inputs_embeds = torch.concat(
        (soft_prompt.broadcast_to((num_beams,) + soft_prompt.shape), word_embeddings),
        dim=1,
    )

    input_ids = torch.LongTensor([[model.config.bos_token_id]]).broadcast_to((num_beams, 1)).to(device)
    stopping_criteria = StoppingCriteriaList([MaxLengthCriteria(max_length=30)])
    logits_processor = LogitsProcessorList([MinLengthLogitsProcessor(5, eos_token_id=model.config.eos_token_id),])
    beam_scorer = BeamSearchScorer(
        batch_size=1,
        num_beams=num_beams,
        device=device,
        #length_penalty=generation_config.length_penalty,
        #do_early_stopping=generation_config.early_stopping,
        num_beam_hyps_to_keep=1,
    )

    outputs = model.beam_search(
        input_ids,
        beam_scorer,
        inputs_embeds=inputs_embeds,
        temperature=temperature,
        logits_processor=logits_processor,
        stopping_criteria=stopping_criteria,
        pad_token_id=model.config.eos_token_id
    )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    prompt_and_generated_text = prompt + generated_text

    return prompt_and_generated_text

def initialize_soft_prompt(model: PreTrainedModel, seq_len: int=20, use_vocab=False):
    if use_vocab:
        indices = torch.randperm(model.get_input_embeddings().weight.shape[0])[:seq_len]
        soft_prompt = model.get_input_embeddings().weight[indices].clone().detach()
    else:
        embedding_dim = model.config.hidden_size
        soft_prompt = torch.distributions.uniform.Uniform(-1., 1.).sample((seq_len, embedding_dim))
    soft_prompt.requires_grad = True
    return soft_prompt

def haiku_dataloaders(tokenizer: PreTrainedTokenizer, batch_size: int, num_workers: int, train_size: float=0.8) -> Tuple[DataLoader, DataLoader]:
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


class DataIterator:
    """Iterates through a dataloader indefinitely similar to itertools.cycle but shuffles correctly"""
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.iterator = iter(dataloader)

    def __next__(self):
        try:
            return next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.dataloader)
            return next(self.iterator)


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
            input = torch.concatenate([sample.input for sample in samples]),
            labels = torch.concatenate([sample.labels for sample in samples]),
        )

@dataclass(order=True, frozen=True)
class HaikuData:
    """A batch or sample of haiku data"""
    haikus: List[str]
    prompts: List[str]
    input: Tensor
    labels: Tensor

    def __getitem__(self, index: Union[int, slice]) -> HaikuData:
        if isinstance(index, int):
            index = slice(index, index + 1)
        return HaikuData(
            haikus = self.haikus[index],
            prompts = self.prompts[index],
            input = self.input[index],
            labels = self.labels[index],
        )

    def __len__(self) -> int:
        return len(self.haikus)

    def to(self, device: Union[Device, str]) -> HaikuData:
        return HaikuData(
            haikus = self.haikus,
            prompts = self.prompts,
            input = self.input.to(device),
            labels = self.labels.to(device),
        )

    def prepend_soft_prompt(self, soft_prompt: Tensor, model: PreTrainedModel) -> HaikuData:
        word_to_embedding_layer = model.get_input_embeddings()
        batch_size = self.input.shape[0]
        null_label = -100

        word_embeddings = word_to_embedding_layer(self.input)
        soft_prompts = torch.broadcast_to(soft_prompt, size=(batch_size,)+soft_prompt.shape)
        input_embeddings = torch.concat((soft_prompts, word_embeddings), dim=1)

        soft_prompt_padding = torch.full(size=soft_prompts.shape[:2], fill_value=null_label, device=self.labels.device)
        labels = torch.concat((soft_prompt_padding, self.labels), dim=1)

        return HaikuData(
            haikus = self.haikus,
            prompts = self.prompts,
            input = input_embeddings,
            labels = labels,
        )

    def __post_init__(self):
        if len(self.prompts) != len(self.haikus):
            raise ValueError("Length of prompts must match length of haikus")
        if len(self.input) != len(self.haikus):
            raise ValueError("Length of input_ids must match length of haikus")
        if len(self.labels) != len(self.haikus):
            raise ValueError("Length of labels must match length of haikus")

def prepare_inputs_for_generation(input_ids, past=None, **kwargs):
    token_type_ids = kwargs.get("token_type_ids", None)
    # only last token for inputs_ids if past is defined in kwargs
    if past:
        input_ids = input_ids[:, -1].unsqueeze(-1)
        if token_type_ids is not None:
            token_type_ids = token_type_ids[:, -1].unsqueeze(-1)

    attention_mask = kwargs.get("attention_mask", None)
    position_ids = kwargs.get("position_ids", None)

    if attention_mask is not None and position_ids is None:
        # create position_ids on the fly for batch generation
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        if past:
            position_ids = position_ids[:, -1].unsqueeze(-1)
    else:
        position_ids = None

    # !!!!!!!!!!!!!!!!!!! start: modified vs original, to pass inputs_embeds when they are available
    if "inputs_embeds" in kwargs and past is None:  # we only want to use them in the 1st generation step
        model_inputs = {"inputs_embeds": kwargs["inputs_embeds"]}
    else:
        model_inputs = {"input_ids": input_ids}
    model_inputs.update({
        "past_key_values": past,
        "use_cache": kwargs.get("use_cache"),
        "position_ids": position_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids,
    })
    return model_inputs
    # !!!!!!!!!!!!!!!!!!! end: modified vs original, to pass inputs_embeds when they are available

if __name__ == "__main__":
    main()