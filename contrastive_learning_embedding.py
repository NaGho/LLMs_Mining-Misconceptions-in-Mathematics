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
from utils import TextPairDataset, preprocess_dataframe


ONLINE = False
# Load the datasets
data_folder = '/kaggle/input/eedi-mining-misconceptions-in-mathematics'# '/content/drive/MyDrive/eedi-mining-misconceptions-in-mathematics'
train_df = pd.read_csv(f'{data_folder}/train.csv')
test_df = pd.read_csv(f'{data_folder}/test.csv')
indicator_mapping = pd.read_csv(f'{data_folder}/misconception_mapping.csv')

import requests

try:
    response = requests.get("https://huggingface.co", timeout=5)
    print("Internet access is available!")
except requests.exceptions.RequestException as e:
    print(f"No internet access: {e}")


# Tokenization with max_length and padding
def tokenize_function(examples, col1='Q&A', col2='MisconceptionName'):
    return tokenizer(
        examples[col1],
        examples[col2],
        padding="max_length",  # Pad to max_length
        truncation=True,
        max_length=128,  # Adjust max_length as necessary
        return_tensors="pt"
    )

if ONLINE:
    # Load tokenizer and model
    pre_trained_model_name = "distilbert-base-uncased"
    local_pre_trained_name = f"./{pre_trained_model_name}"

    tokenizer = AutoTokenizer.from_pretrained(pre_trained_model_name)
    model = AutoModelForSequenceClassification.from_pretrained(pre_trained_model_name, num_labels=2)
    tokenizer.save_pretrained(local_pre_trained_name)
    model.save_pretrained(local_pre_trained_name)
    # else:
    #     tokenizer = AutoTokenizer.from_pretrained(local_pre_trained_name, local_files_only=True)
    #     model = AutoModelForSequenceClassification.from_pretrained(local_pre_trained_name, local_files_only=True)


    df = preprocess_dataframe(train_df, indicator_mapping, misconception=True)
    # Convert DataFrame to Dataset
    dataset_train = Dataset.from_pandas(df[:len(df)-200])
    dataset_eval = Dataset.from_pandas(df[len(df)-200:])

    

    tokenized_dataset_train = dataset_train.map(tokenize_function, batched=True)
    tokenized_dataset_eval = dataset_eval.map(tokenize_function, batched=True)


    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=32,
        num_train_epochs=1,
        weight_decay=0.01,
    )

    print('Inititalize Trainer ...')
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset_train,
        eval_dataset=tokenized_dataset_eval,
    )
    if False and ONLINE:
        print('Train the model ...')
        # Train the model
        trainer.train()

        print('Save the model ...')
        # Save the model
        trainer.save_model("/kaggle/working/fine_tuned_model")
    #     tokenizer.save_pretrained()


import os
print(os.listdir("/kaggle/input/my_fine_tuned/transformers/default/1/fine_tuned_model"))
print(os.listdir("/kaggle/input/my_distilbert_model/pytorch/default/1/distilbert-base-uncased"))



if ONLINE:
    # Load the fine-tuned model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(local_pre_trained_name, local_files_only=True)
    model_name = "/kaggle/working/fine_tuned_model"  # Path to your fine-tuned model
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
else:
    model_name = "/kaggle/input/my_fine_tuned/transformers/default/1/fine_tuned_model"  # Path to your fine-tuned model
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer_path = "/kaggle/input/my_distilbert_model/pytorch/default/1/distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    

# prepare test data
df = preprocess_dataframe(test_df, indicator_mapping, misconception=False)
indicator_mapping['tmp_col'] = 1
df['tmp_col'] = 1
df = pd.merge(df, indicator_mapping, on='tmp_col', how='outer')
df = df.drop(columns=['tmp_col'])
# display(df)


if False:
    dataset_test = Dataset.from_pandas(df)
    tokenized_dataset_test = dataset_test.map(tokenize_function, batched=True)
else:
    tokenized_dataset_test = TextPairDataset(df, tokenizer)
    

# Create Dataset and DataLoader
batch_size = 32  # Adjust as needed
test_dataloader = DataLoader(tokenized_dataset_test, batch_size=batch_size)

# Move model to device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Inference loop
all_predictions = []
for batch in tqdm(test_dataloader, desc="Processing Batches"):
    # Move batch to the same device as the model
    batch = {key: val.to(device) for key, val in batch.items()}

    # Perform inference
    with torch.no_grad():
        outputs = model(**batch)
    if False:
      # Get the predicted class (or logits if needed)
      predictions = torch.max(outputs.logits, dim=-1)
    else:
      # Apply softmax to get probabilities
      probs = F.softmax(outputs.logits, dim=-1)

      # Get the maximum logistic (probability) value for each sample in the batch
      predictions = torch.max(probs, dim=-1)[0]  # [0] gives the max values (logits), not the indices

    all_predictions.extend(predictions.cpu().numpy())  # Collect predictions

# Append the predictions to the original DataFrame
df['prediction'] = all_predictions

# Print the DataFrame with predictions
display(df)


df['QuestionId_Answer'] = df['QuestionId'].astype(str) + '_' + df['Answer']

df = df.sort_values(by=['QuestionId_Answer', 'prediction'], ascending=[True, False])
df_out = df.groupby(['QuestionId_Answer', 'Answer']).apply(
    lambda x: x['MisconceptionId'].head(25).tolist()
).reset_index().rename(columns={0: 'MisconceptionId'})
display(df_out)
df_out[['QuestionId_Answer', 'MisconceptionId']].to_csv('/kaggle/working/submission.csv', index=False)


# import os
# from langchain_openai import OpenAIEmbeddings

# os.environ['OPENAI_API_KEY'] = '{{OpenAI_API}}' # Set your OpenAI API key in the environment variable


# embeddings_model = OpenAIEmbeddings() # Initialize the embeddings model
# # Generate embeddings for a list of documents
# embeddings = embeddings_model.embed_documents(
#     [
#     "This is the Fundamentals of RAG course.",
#     "Educative is an AI-powered online learning platform.",
#     "There are several Generative AI courses available on Educative.",
#     "I am writing this using my keyboard.",
#     "JavaScript is a good programming language"
#     ]
# )

# print(len(embeddings)) # Print the number of embeddings generated (should be equal to the number of documents)
# print(len((embeddings[0]))) # Print the length of the first embedding vector


# # Create a Chroma database from the documents using OpenAI embeddings
# db = Chroma.from_texts(documents, OpenAIEmbeddings())

# # Configure the database to act as a retriever, setting the search type to
# # similarity and returning the top 1 result
# retriever = db.as_retriever(
#     search_type="similarity",
#     search_kwargs={'k': 1}
# )

# # Perform a similarity search with the given query
# result = retriever.invoke("Where can I see Mona Lisa?")
# print(result)


# # Importing the GPT2 tokenizer from the transformers library.
# tokenizer = GPT2Tokenizer.from_pretrained("gpt2", local_files_only=True)

# # Defining a function to create a dataset of lyrics for training.
# def create_lyrics_dataset(lyrics, prefix_code, limit_length=990):

#     # Encoding each lyric line with the tokenizer.
#     encoded_lyrics = [
#         torch.tensor(tokenizer.encode(f"<|{prefix_code}|>{line[:limit_length]}"))
#         for line in lyrics 
#     ]
#     return encoded_lyrics  

# train_subset = create_lyrics_dataset(folk_songs['Lyric'], "prefix_code")


# # Import required libraries
# import pandas as pd
# import numpy as np
# import torch
# from torch.utils.data import DataLoader
# from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup
# import torch.nn.functional as F
# import os

# # Load the tokenizer pretrained on the GPT-2 model
# tokenizer = GPT2Tokenizer.from_pretrained('gpt2', local_files_only=True)

# # Load the pretrained weights of the GPT-2 model
# model = GPT2LMHeadModel.from_pretrained('gpt2', local_files_only=True)


# def training(dataset, model, tokenizer, batch_size=8, epochs=10, learning_rate=1e-3):

#     # Initialization of training parameters and setting the device
#     compute_device = torch.device("cpu")
#     model.to(compute_device)
#     model.train()

#     # Preparing the data loader for batch processing
#     batch_loader = DataLoader(dataset, batch_size=1, shuffle=True)
#     current_loss, batch_accumulator = 0, 0  
#     processing_tensor = None

#     # Setting up the optimizer and scheduler for training
#     model_optimizer = AdamW(model.parameters(), lr=learning_rate)
#     training_scheduler = get_linear_schedule_with_warmup(
#         model_optimizer, num_warmup_steps=10, num_training_steps=-1
#     )

#     # Looping through each training epoch
#     for current_epoch in range(epochs):
#         print(f"Current training epoch: {current_epoch}")
#         print(current_loss)

#         # Initializing the index for a while loop
#         index = 0
#         batch_iterator = iter(batch_loader)

#         # Processing each batch in the dataset using a while loop
#         while index < len(batch_loader):
#             data_batch = next(batch_iterator)

#             # Handling tensor packing for batch processing
#             if processing_tensor is None or processing_tensor.size()[1] + data_batch.size()[1] > 500: # Max sequence limit for batches
#                 processing_tensor, continue_batch = (data_batch, True) if processing_tensor is None else (processing_tensor, False)
#             else:
#                 processing_tensor = torch.cat([data_batch, processing_tensor[:, 1:]], dim=1)
#                 continue_batch = True

#             # Skipping to next iteration if current batch should not be processed now
#             if continue_batch and index != len(batch_loader) - 1:
#                 index += 1
#                 continue

#             # Training model on the current batch
#             processing_tensor = processing_tensor.to(compute_device)
#             model_output = model(processing_tensor, labels=processing_tensor)
#             current_loss = model_output[0]
#             current_loss.backward()

#             # Optimizer and scheduler steps
#             if (batch_accumulator % batch_size) == 0:
#                 model_optimizer.step()
#                 training_scheduler.step()
#                 model_optimizer.zero_grad()
#                 model.zero_grad()

#             batch_accumulator += 1
#             processing_tensor = None
#             index += 1

#     # Saving the model state after training is complete
#     torch.save(
#         model.state_dict(),
#         os.path.join(".", "model-final.pt"),
#     )

#     return model

# # Training the model on the specific data we have
# model = training(train_subset, model, tokenizer)