
import pandas as pd
import torch
from transformers import (
    GPT2Tokenizer, GPT2Model, BertTokenizer, BertModel
)
# from pycaret.classification import *
import numpy as np
# !pip install -U sentence-transformers
# from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
# !pip install datasets==2.6.1
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch.nn.functional as F
from tqdm import tqdm

def preprocess_dataframe(df, indicator_mapping, misconception=False):
  id_vars = ["QuestionText", 'QuestionId']
  var_name = 'Answer'
  answer_melted = df.melt(
      id_vars=id_vars,
      value_vars=[f'Answer{let}Text' for let in ['A', 'B', 'C', 'D']],
      value_name='AnswerText',
      var_name=var_name
  )
  for strr in ['Answer', 'Text']:
    answer_melted[var_name] = answer_melted[var_name].str.replace(strr, '')


  if misconception:
    misconception_melted = df.melt(
        id_vars=id_vars, value_vars=[f'Misconception{let}Id' for let in ['A', 'B', 'C', 'D']],
        value_name='MisconceptionId', var_name=var_name
    )
    for strr in ['Misconception', 'Id']:
      misconception_melted[var_name] = misconception_melted[var_name].str.replace(strr, '')

    df = pd.merge(answer_melted, misconception_melted, on=['QuestionText', var_name])
  else:
    df = answer_melted
  df['Q&A'] = df['QuestionText'] + ' ' + df['AnswerText']

  if misconception:
    df = pd.merge(df, indicator_mapping, on='MisconceptionId')

  df['label'] = 1 # 1 if the texts are related, 0 otherwise
  df = df.dropna(subset=['Q&A'])

  return df


# Custom Dataset class for DataFrame
class TextPairDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=128, col1='Q&A', col2='MisconceptionName'):
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.col1 = col1
        self.col2 = col2

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        text1 = self.dataframe.loc[idx, self.col1].to_list()  # First column is text1
        text2 = self.dataframe.loc[idx, self.col2].to_list()  # Second column is text2
        encoding = self.tokenizer(
            text1,
            text2,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {key: val.squeeze(0) for key, val in encoding.items()}
    

