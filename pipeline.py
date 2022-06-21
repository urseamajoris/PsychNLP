import torch
from transformers import AutoTokenizer, AutoModel

from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.nn.functional as F
from torch import nn, optim

def pipeline(text):

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


    model = torch.load('model.pt', map_location=torch.device('cpu'))
    model = model.to(device)

    


    encoding = tokenizer.encode_plus(
          text,
          add_special_tokens=True,
          max_length=128,
          return_token_type_ids=False,
          padding='max_length',
          truncation=True,
          return_attention_mask=True,
          return_tensors='pt',
        )

    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    output = model(input_ids, attention_mask)

    def result(output):
        if torch.argmax(output, dim=1)[0] == 0:
            out = f'this message indicates normal mood ({output[0][0]*100}% chance of not having depression and suicide)'
        else:
            out = f'this message indicates {output[0][1]*100}% risk of depression, {output[0][2]*100}% of suicide'
        return out


    print(result(output))
    return result(output)


