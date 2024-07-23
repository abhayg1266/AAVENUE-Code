#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install gdown pandas sacrebleu


# In[2]:


from google.colab import drive
drive.mount('/content/drive')


# In[3]:


import gdown
import pandas as pd
import sacrebleu

# Replace 'FILE_ID' with the actual ID of your file
file_id = '1-cmuoQzB78Pa1dIH0wh-cSWQbB8MFVeY'
output = 'translations.csv'

# Construct the URL for gdown
url = f'https://drive.google.com/uc?id={file_id}'

# Download the file
gdown.download(url, output, quiet=False)

# Read the downloaded CSV file
df = pd.read_csv(output)

# Ensure the columns are correctly named
assert 'Original Passage' in df.columns and 'Translated Passage' in df.columns, "CSV must contain 'Original Passage' and 'Translated Passage' columns."

# List to store results
results = []

# Calculate BLEU score for each sentence pair and store results
for index, row in df.iterrows():
    reference = row['Original Passage'].lower()
    hypothesis = row['Translated Passage'].lower()
    bleu_score = sacrebleu.sentence_bleu(hypothesis, [reference])

    if bleu_score.score < 10:
      bleu_under_10 = 1
    else:
      bleu_under_10 = 0
    if bleu_score.score < 20:
      bleu_under_20 = 1
    else:
      bleu_under_20 = 0
    if bleu_score.score < 30:
      bleu_under_30 = 1
    else:
      bleu_under_30 = 0
    if bleu_score.score < 50:
      bleu_under_50 = 1
    else:
      bleu_under_50 = 0
    if bleu_score.score < 70:
      bleu_under_70 = 1
    else:
      bleu_under_70 = 0

    # Append results to the list
    results.append({
        'Reference': reference,
        'Hypothesis': hypothesis,
        'BLEU_score': bleu_score.score,
        'BLEU_under_10': bleu_under_10,
        'BLEU_under_20': bleu_under_20,
        'BLEU_under_30': bleu_under_30,
        'BLEU_under_50': bleu_under_50,
        'BLEU_under_70': bleu_under_70
    })

# Create a DataFrame from the results list
results_df = pd.DataFrame(results)

# Save results to a new CSV file
output_results_csv = '/content/drive/MyDrive/Algoverse/BLEU Scores/results_with_bleu_scores_wsc.csv'
results_df.to_csv(output_results_csv, index=False)

print(f"Results with BLEU scores saved to '{output_results_csv}'.")

