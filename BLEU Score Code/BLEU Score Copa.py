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
file_id = '13KLg7NKleTMya64iNlCzkdm2sMWtecDe'
output = 'translations.csv'

# Construct the URL for gdown
url = f'https://drive.google.com/uc?id={file_id}'

# Download the file
gdown.download(url, output, quiet=False)

# Read the downloaded CSV file
df = pd.read_csv(output)

# Ensure the columns are correctly named
assert 'Premise' in df.columns and 'Translated Premise' in df.columns and 'Choice1' in df.columns and'Translated Choice1' in df.columns and 'Choice2' in df.columns and'Translated Choice2' in df.columns, "CSV must contain these columns."

# List to store results
results = []

# Calculate BLEU score for each sentence pair and store results
for index, row in df.iterrows():
    premise_reference = row['Premise'].lower()
    premise_hypothesis = row['Translated Premise'].lower()
    bleu_score_premise = sacrebleu.sentence_bleu(premise_hypothesis, [premise_reference])

    if bleu_score_premise.score < 10:
      bleu_premise_under_10 = 1
    else:
      bleu_premise_under_10 = 0
    if bleu_score_premise.score < 20:
      bleu_premise_under_20 = 1
    else:
      bleu_premise_under_20 = 0
    if bleu_score_premise.score < 30:
      bleu_premise_under_30 = 1
    else:
      bleu_premise_under_30 = 0
    if bleu_score_premise.score < 50:
      bleu_premise_under_50 = 1
    else:
      bleu_premise_under_50 = 0
    if bleu_score_premise.score < 70:
      bleu_premise_under_70 = 1
    else:
      bleu_premise_under_70 = 0

    choice1_reference = row['Choice1'].lower()
    choice1_hypothesis = row['Translated Choice1'].lower()
    bleu_score_choice1 = sacrebleu.sentence_bleu(choice1_hypothesis, [choice1_reference])

    if bleu_score_choice1.score < 10:
      bleu_choice1_under_10 = 1
    else:
      bleu_choice1_under_10 = 0
    if bleu_score_choice1.score < 20:
      bleu_choice1_under_20 = 1
    else:
      bleu_choice1_under_20 = 0
    if bleu_score_choice1.score < 30:
      bleu_choice1_under_30 = 1
    else:
      bleu_choice1_under_30 = 0
    if bleu_score_choice1.score < 50:
      bleu_choice1_under_50 = 1
    else:
      bleu_choice1_under_50 = 0
    if bleu_score_choice1.score < 70:
      bleu_choice1_under_70 = 1
    else:
      bleu_choice1_under_70 = 0

    choice2_reference = row['Choice2'].lower()
    choice2_hypothesis = row['Translated Choice2'].lower()
    bleu_score_choice2 = sacrebleu.sentence_bleu(choice2_hypothesis, [choice2_reference])

    if bleu_score_choice2.score < 10:
      bleu_choice2_under_10 = 1
    else:
      bleu_choice2_under_10 = 0
    if bleu_score_choice2.score < 20:
      bleu_choice2_under_20 = 1
    else:
      bleu_choice2_under_20 = 0
    if bleu_score_choice2.score < 30:
      bleu_choice2_under_30 = 1
    else:
      bleu_choice2_under_30 = 0
    if bleu_score_choice2.score < 50:
      bleu_choice2_under_50 = 1
    else:
      bleu_choice2_under_50 = 0
    if bleu_score_choice2.score < 70:
      bleu_choice2_under_70 = 1
    else:
      bleu_choice2_under_70 = 0

    # Append results to the list
    results.append({
        'Premise Reference': premise_reference,
        'Premise Hypothesis': premise_hypothesis,
        'Choice1 Reference': choice1_reference,
        'Choice1 Hypothesis': choice1_hypothesis,
        'Choice2 Reference': choice2_reference,
        'Choice2 Hypothesis': choice2_hypothesis,
        'Premise BLEU_score': bleu_score_premise.score,
        'Choice1 BLEU_score': bleu_score_choice1.score,
        'Choice2 BLEU_score': bleu_score_choice2.score,
        'Premise BLEU_score_under_10': bleu_premise_under_10,
        'Premise BLEU_score_under_20': bleu_premise_under_20,
        'Premise BLEU_score_under_30': bleu_premise_under_30,
        'Premise BLEU_score_under_50': bleu_premise_under_50,
        'Premise BLEU_score_under_70': bleu_premise_under_70,
        'Choice1 BLEU_score_under_10': bleu_choice1_under_10,
        'Choice1 BLEU_score_under_20': bleu_choice1_under_20,
        'Choice1 BLEU_score_under_30': bleu_choice1_under_30,
        'Choice1 BLEU_score_under_50': bleu_choice1_under_50,
        'Choice1 BLEU_score_under_70': bleu_choice1_under_70,
        'Choice2 BLEU_score_under_10': bleu_choice2_under_10,
        'Choice2 BLEU_score_under_20': bleu_choice2_under_20,
        'Choice2 BLEU_score_under_30': bleu_choice2_under_30,
        'Choice2 BLEU_score_under_50': bleu_choice2_under_50,
        'Choice2 BLEU_score_under_70': bleu_choice2_under_70
    })

# Create a DataFrame from the results list
results_df = pd.DataFrame(results)

# Save results to a new CSV file
output_results_csv = '/content/drive/MyDrive/Algoverse/BLEU Scores/results_with_bleu_scores_copa.csv'
results_df.to_csv(output_results_csv, index=False)

print(f"Results with BLEU scores saved to '{output_results_csv}'.")

