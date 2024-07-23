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
file_id = '1IHm5mQFvhKb1cAggwo6PcUYcN_C4fcAR'
output = 'translations.csv'

# Construct the URL for gdown
url = f'https://drive.google.com/uc?id={file_id}'

# Download the file
gdown.download(url, output, quiet=False)

# Read the downloaded CSV file
df = pd.read_csv(output)

# Ensure the columns are correctly named
assert 'Paragraph' in df.columns and 'Translated Paragraph' in df.columns and 'Question' in df.columns and 'Translated Question' in df.columns and 'Answer' in df.columns and 'Translated Answer' in df.columns, "CSV must contain 'Original Passage' and 'Translated Passage' columns."

# List to store results
results = []

# Calculate BLEU score for each sentence pair and store results
for index, row in df.iterrows():
    passage_reference = row['Paragraph'].lower()
    passage_hypothesis = row['Translated Paragraph'].lower()
    bleu_score_passage = sacrebleu.sentence_bleu(passage_hypothesis, [passage_reference])

    if bleu_score_passage.score < 10:
       bleu_passage_under_10 = 1
    else:
       bleu_passage_under_10 = 0
    if bleu_score_passage.score < 20:
       bleu_passage_under_20 = 1
    else:
       bleu_passage_under_20 = 0
    if bleu_score_passage.score < 30:
       bleu_passage_under_30 = 1
    else:
       bleu_passage_under_30 = 0
    if bleu_score_passage.score < 50:
       bleu_passage_under_50 = 1
    else:
       bleu_passage_under_50 = 0
    if bleu_score_passage.score < 70:
       bleu_passage_under_70 = 1
    else:
       bleu_passage_under_70 = 0

    question_reference = row['Question'].lower()
    question_hypothesis = row['Translated Question'].lower()
    if pd.notna(question_hypothesis):
        bleu_score_question = sacrebleu.sentence_bleu(question_hypothesis, [question_reference])
    else:
        bleu_score_question = sacrebleu.sentence_bleu("a", ["b"])

    if bleu_score_question.score < 10:
       bleu_question_under_10 = 1
    else:
       bleu_question_under_10 = 0
    if bleu_score_question.score < 20:
       bleu_question_under_20 = 1
    else:
       bleu_question_under_20 = 0
    if bleu_score_question.score < 30:
       bleu_question_under_30 = 1
    else:
       bleu_question_under_30 = 0
    if bleu_score_question.score < 50:
       bleu_question_under_50 = 1
    else:
       bleu_question_under_50 = 0
    if bleu_score_question.score < 70:
       bleu_question_under_70 = 1
    else:
       bleu_question_under_70 = 0

    answer_reference = row['Answer'].lower() if pd.notna(row['Answer']) else ""
    answer_hypothesis = row['Translated Answer'].lower()
    bleu_score_answer = sacrebleu.sentence_bleu(answer_hypothesis, [answer_reference])

    if bleu_score_answer.score < 10:
       bleu_answer_under_10 = 1
    else:
       bleu_answer_under_10 = 0
    if bleu_score_answer.score < 20:
       bleu_answer_under_20 = 1
    else:
       bleu_answer_under_20 = 0
    if bleu_score_answer.score < 30:
       bleu_answer_under_30 = 1
    else:
       bleu_answer_under_30 = 0
    if bleu_score_answer.score < 50:
       bleu_answer_under_50 = 1
    else:
       bleu_answer_under_50 = 0
    if bleu_score_answer.score < 70:
       bleu_answer_under_70 = 1
    else:
       bleu_answer_under_70 = 0

    # Append results to the list
    results.append({
        'Passage Reference': passage_reference,
        'Passage Hypothesis': passage_hypothesis,
        'Question Reference': question_reference,
        'Question Hypothesis': question_hypothesis,
        'Answer Reference': answer_reference,
        'Answer Hypothesis': answer_hypothesis,
        'Passage BLEU_score': bleu_score_passage.score,
        'Question BLEU_score': bleu_score_question.score,
        'Answer BLEU_score': bleu_score_answer.score,
        'Passage BLEU_score_under_10': bleu_passage_under_10,
        'Passage BLEU_score_under_20': bleu_passage_under_20,
        'Passage BLEU_score_under_30': bleu_passage_under_30,
        'Passage BLEU_score_under_50': bleu_passage_under_50,
        'Passage BLEU_score_under_70': bleu_passage_under_70,
        'Question BLEU_score_under_10': bleu_question_under_10,
        'Question BLEU_score_under_20': bleu_question_under_20,
        'Question BLEU_score_under_30': bleu_question_under_30,
        'Question BLEU_score_under_50': bleu_question_under_50,
        'Question BLEU_score_under_70': bleu_question_under_70,
        'Answer BLEU_score_under_10': bleu_answer_under_10,
        'Answer BLEU_score_under_20': bleu_answer_under_20,
        'Answer BLEU_score_under_30': bleu_answer_under_30,
        'Answer BLEU_score_under_50': bleu_answer_under_50,
        'Answer BLEU_score_under_70': bleu_answer_under_70

    })

# Create a DataFrame from the results list
results_df = pd.DataFrame(results)

# Save results to a new CSV file
output_results_csv = '/content/drive/MyDrive/Algoverse/BLEU Scores/results_with_bleu_scores_multirc.csv'
results_df.to_csv(output_results_csv, index=False)

print(f"Results with BLEU scores saved to '{output_results_csv}'.")
