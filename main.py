import discord
from pipeline import pipeline
import torch
from transformers import AutoTokenizer, AutoModel
from dotenv import load_dotenv
import os

from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.nn.functional as F
from torch import nn, optim
load_dotenv()

model_name = 'sentence-transformers/all-distilroberta-v1'
tokenizer = AutoTokenizer.from_pretrained(model_name)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class TextClassifier(nn.Module):
     def __init__(self, n_classes):
       super(TextClassifier, self).__init__()
       self.bert = AutoModel.from_pretrained(model_name, return_dict=False)
       self.drop = nn.Dropout(p=0.3)
       self.L1 = nn.Linear(self.bert.config.hidden_size, n_classes)
       self.out = nn.Softmax(dim=1)
    

     def forward(self, input_ids, attention_mask):
       _, pooled_output = self.bert(
           input_ids = input_ids,
           attention_mask = attention_mask
       )
       output = self.drop(pooled_output)
       output = self.L1(output)
       output = self.out(output)
       return output


client = discord.Client()

@client.event
async def on_ready():
    print('We have logged in as {0.user}'.format(client))

@client.event
async def on_message(message):
    if message.author == client.user:
        return

    if message.content.startswith('$hello'):
        await message.channel.send('Hello!')
        msg = message.content.split(' ', 1)[1]
        
        print(msg)
    
    if message.content.startswith('$classify'):
        msg = message.content.split(' ', 1)[1]
        result = pipeline(msg)
        await message.channel.send(result)

client.run(os.environ.get('token'))
