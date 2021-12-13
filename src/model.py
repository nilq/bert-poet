import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader, Dataset

import pandas as pd
import numpy as np
import itertools
import json

from transformers import BertTokenizer, BertModel
from dataclasses  import dataclass, field

from tqdm import tqdm
from os.path import join, dirname

@dataclass
class BERTEmbedder:
    model = BertModel.from_pretrained(
        'bert-base-uncased',
        output_hidden_states=True
    )
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def tokenize_sentence(self, sentence, padding=None):
        marked    = f'[CLS] {sentence} [SEP]'
        tokenized = self.tokenizer.tokenize(marked)
        indexed   = self.tokenizer.convert_tokens_to_ids(tokenized)

        padding = padding or len(indexed)
        indexed += [0] * (padding - len(indexed))

        return indexed

    def embed_sentence(self, sentence):
        indexed = self.tokenize_sentence(sentence)

        tensor  = torch.Tensor([indexed])
        outputs = torch.stack(self.model(tensor)[0][2:]).sum(0)

        return outputs

    def embedding_layer(self):
        return list(list(self.model.children())[0].children())[0]

class PoetryRNN(nn.Module):
    def __init__(self, bert):
        super(PoetryRNN, self).__init__()

        self.lstm_size     = 768
        self.num_layers    = 3
        self.bert          = bert

        self.embedding = bert.embedding_layer()

        self.lstm = nn.LSTM(
            input_size=self.lstm_size,
            hidden_size=self.lstm_size,
            num_layers=self.num_layers,
            dropout=0.2
        )

        self.fc = nn.Linear(self.lstm_size, 30522)

    def forward(self, x, prev):
        embeddings = self.embedding(x)
        output, state = self.lstm(embeddings, prev)
        logits = self.fc(output)

        return logits, state

    def init_state(self, seq_length):
        return (
            torch.zeros(self.num_layers, seq_length, self.lstm_size),
            torch.zeros(self.num_layers, seq_length, self.lstm_size)
        )

class PoemDataset(Dataset):
    def __init__(self, bert, seq_length=150):
        poems = json.loads(open(join(dirname(__file__), 'data/result.json')).read())

        self.seq_length = seq_length
        self.data = list(itertools.chain(*list(map(
            bert.tokenize_sentence,
            tqdm(poems, desc='Processing')
        ))))

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, i):
        return (
            torch.Tensor(self.data[i:i + self.seq_length]).long(),
            torch.Tensor(self.data[i + 1:i + 1 + self.seq_length]).long()
        )

def train(model, data, epochs=10, seq_length=150):
    model.train()

    dataloader = DataLoader(data, batch_size=10)
    criterion  = nn.CrossEntropyLoss()
    optimizer  = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(1, epochs):
        state_h, state_c = model.init_state(seq_length)

        loss_total = 0
        batches = 0

        for batch, (x, y) in tqdm(list(enumerate(dataloader)), desc=f'Epoch #{epoch}'):
            optimizer.zero_grad()

            pred, (state_h, state_c) = model(x, (state_h, state_c))
            loss = criterion(pred.transpose(1, 2), y)

            state_h = state_h.detach()
            state_c = state_c.detach()

            loss.backward()
            optimizer.step()

            loss_total += loss.item()
            batches += 1

        print(f'=== EPOCH #{epoch}:\n- average loss: {loss_total / batches * 100:.2f}%')

def predict(model, text, how_many=100):
    model.eval()

    sentence = text

    state_h, state_c = model.init_state(128)

    for i in tqdm(range(0, how_many), desc="Hypertraculating the mainframe-caboratorizer"):
        indexed = model.bert.tokenize_sentence(sentence, 128)

        pred, (state_h, state_c) = model(torch.Tensor([indexed]).int(), (state_h, state_c))
        p = F.softmax(pred[0][-1], dim=0).detach().numpy()

        next_word = model.bert.tokenizer.decode([np.argmax(p)])
        sentence += ' ' + next_word

    return sentence


if __name__ == '__main__':
    bert = BERTEmbedder()

    data = PoemDataset(bert)
    poet = PoetryRNN(bert)

    train(poet, data)

    output = predict(poet, 'what are you doing')

    import pdb
    pdb.set_trace()
