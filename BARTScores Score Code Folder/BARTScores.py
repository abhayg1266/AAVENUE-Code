#!/usr/bin/env python
# coding: utf-8

# In[21]:


get_ipython().system('pip install transformers datasets')


# In[22]:


from transformers import BartTokenizer, BartForConditionalGeneration
import pandas as pd
import torch
from google.colab import drive

# Mount Google Drive
drive.mount('/content/drive')

# Load the BART model and tokenizer
model_name = 'facebook/bart-large-cnn'
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)


# In[23]:


# Read the translation files from Google Drive
translations_SST2_df = pd.read_csv('/content/drive/MyDrive/Algoverse/Values Translations/GPT 4o/SST-2_filtered.csv')
translations_BoolQ_df = pd.read_csv('/content/drive/MyDrive/Algoverse/Values Translations/GPT 4o/BoolQ_filtered.csv')
translations_Wsc_df = pd.read_csv('/content/drive/MyDrive/Algoverse/Values Translations/GPT 4o/WSC_filtered.csv')
translations_Copa_df = pd.read_csv('/content/drive/MyDrive/Algoverse/Values Translations/GPT 4o/Copa_Filtered.csv')

# Define the data frames
sst2_data = pd.DataFrame({
    'original': translations_SST2_df['Original'],
    'translated': translations_SST2_df['Translated']
})

boolq_data = pd.DataFrame({
    'original_question': translations_BoolQ_df['Original Question'],
    'translated_question': translations_BoolQ_df['Translated Question'],
    'original_passage': translations_BoolQ_df['Original Passage'],
    'translated_passage': translations_BoolQ_df['Translated Passage']
})

wsc_data = pd.DataFrame({
    'original_passage': translations_Wsc_df['Original Passage'],
    'translated_passage': translations_Wsc_df['Translated Passage']
})

copa_data = pd.DataFrame({
    'original_premise': translations_Copa_df['Original Premise'],
    'translated_premise': translations_Copa_df['Translated Premise'],
    'original_choice1': translations_Copa_df['Original Choice1'],
    'translated_choice1': translations_Copa_df['Translated Choice1'],
    'original_choice2': translations_Copa_df['Original Choice2'],
    'translated_choice2': translations_Copa_df['Translated Choice2']
})


# In[24]:


# Function to calculate BARTScore
def calculate_bartscore(reference, candidate):
    if not isinstance(reference, str) or not isinstance(candidate, str):
        return float('nan')
    inputs = tokenizer([reference, candidate], return_tensors='pt', max_length=1024, truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    target_ids = inputs["input_ids"]
    target_log_probs = log_probs.gather(2, target_ids.unsqueeze(-1)).squeeze(-1)
    sentence_log_prob = target_log_probs.sum(1)
    bart_score = sentence_log_prob.mean().item()
    return bart_score


# In[25]:


# Calculate BARTScores for each dataset
sst2_data['bartscore'] = sst2_data.apply(lambda row: calculate_bartscore(row['original'], row['translated']) if pd.notnull(row['original']) and pd.notnull(row['translated']) else float('nan'), axis=1)

boolq_data['question_bartscore'] = boolq_data.apply(lambda row: calculate_bartscore(row['original_question'], row['translated_question']) if pd.notnull(row['original_question']) and pd.notnull(row['translated_question']) else float('nan'), axis=1)
boolq_data['passage_bartscore'] = boolq_data.apply(lambda row: calculate_bartscore(row['original_passage'], row['translated_passage']) if pd.notnull(row['original_passage']) and pd.notnull(row['translated_passage']) else float('nan'), axis=1)

wsc_data['bartscore'] = wsc_data.apply(lambda row: calculate_bartscore(row['original_passage'], row['translated_passage']) if pd.notnull(row['original_passage']) and pd.notnull(row['translated_passage']) else float('nan'), axis=1)

copa_data['premise_bartscore'] = copa_data.apply(lambda row: calculate_bartscore(row['original_premise'], row['translated_premise']) if pd.notnull(row['original_premise']) and pd.notnull(row['translated_premise']) else float('nan'), axis=1)
copa_data['choice1_bartscore'] = copa_data.apply(lambda row: calculate_bartscore(row['original_choice1'], row['translated_choice1']) if pd.notnull(row['original_choice1']) and pd.notnull(row['translated_choice1']) else float('nan'), axis=1)
copa_data['choice2_bartscore'] = copa_data.apply(lambda row: calculate_bartscore(row['original_choice2'], row['translated_choice2']) if pd.notnull(row['original_choice2']) and pd.notnull(row['translated_choice2']) else float('nan'), axis=1)


# In[26]:


# Calculate average BARTScores, ignoring NaN values
average_sst2_bartscore = sst2_data['bartscore'].mean()
average_boolq_question_bartscore = boolq_data['question_bartscore'].mean()
average_boolq_passage_bartscore = boolq_data['passage_bartscore'].mean()
average_wsc_bartscore = wsc_data['bartscore'].mean()
average_copa_premise_bartscore = copa_data['premise_bartscore'].mean()
average_copa_choice1_bartscore = copa_data['choice1_bartscore'].mean()
average_copa_choice2_bartscore = copa_data['choice2_bartscore'].mean()

# Save the average BARTScores to a summary file
with open('/content/drive/MyDrive/Algoverse/Validation Scores/GPT 4o/BARTScores/average_bartscores.txt', 'w') as f:
    f.write(f"Average BARTScores:\n")
    f.write(f"SST-2: {average_sst2_bartscore}\n")
    f.write(f"BoolQ Question: {average_boolq_question_bartscore}\n")
    f.write(f"BoolQ Passage: {average_boolq_passage_bartscore}\n")
    f.write(f"WSC: {average_wsc_bartscore}\n")
    f.write(f"COPA Premise: {average_copa_premise_bartscore}\n")
    f.write(f"COPA Choice1: {average_copa_choice1_bartscore}\n")
    f.write(f"COPA Choice2: {average_copa_choice2_bartscore}\n")

print("BARTScore calculations complete. Averages saved to a summary file.")

