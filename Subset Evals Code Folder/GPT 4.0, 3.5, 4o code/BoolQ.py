#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Installations needed before, run this first

get_ipython().system('pip install datasets')
get_ipython().system('pip install datasets openai')
get_ipython().system('pip install openai==0.28')
get_ipython().system('pip install datasets openai pandas google-colab')
get_ipython().system('pip install pandas')


# In[ ]:


# Imports

import openai
import pandas as pd
from datasets import load_dataset
from google.colab import drive

# Set up your OpenAI API key
openai.api_key = "API KEY"

# Set up OpenAI organization key
openai.organization = "Organization"


# In[ ]:


from google.colab import drive
drive.mount('/content/drive')


# In[ ]:


import pandas as pd
def query_gpt(passage, question):
    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are reading a passage, interpreting the information, and answering the question as 'True' or 'False' only. No other characters should be added in your response."},
            {"role": "user", "content": f"Passage: \"{passage}\"\nQuestion: \"{question}\""}
        ],
        max_tokens=50
    )
    return response.choices[0].message['content'] if response.choices else "No translation found."

# Function to convert label to True/False
def convert(answer):
    return ["True", "True.", "true", "true."] if answer == 1 else ["False", "False.", "false", "false."]

# Function to check correctness
def check_correctness(predicted, actual):
    return 1 if predicted in actual else 0

# Load translations from CSV
csv_file_path = '/content/drive/MyDrive/BoolQA_superglue_1000_4-0.csv'
translations_df = pd.read_csv(csv_file_path)

# Initialize results list
results = []

try:
    for index, row in translations_df.head(999).iterrows():  # Process only the first 5 rows
        question = row['se question']
        passage = row['se passage']
        answer = row['actual answer']  # Actual answer from CSV (0 or 1)
        actual_answer = convert(answer)  # Convert answer to list of possible True/False values

        # Read AAVE translations
        aave_question = row['aave qustion']
        aave_passage = row['aave passage']

        # Debug prints
        print(f"Processing row {index + 1}")
        print(f"SE Question: {question}")
        print(f"AAVE Question: {aave_question}")
        print(f"SE Passage: {passage}")
        print(f"AAVE Passage: {aave_passage}")

        # Determine answers
        se_answer_new = query_gpt(passage, question)
        aave_answer = query_gpt(aave_passage, aave_question)

        # Check correctness
        se_correct = check_correctness(se_answer_new, actual_answer)
        aave_correct = check_correctness(aave_answer, actual_answer)

        # Append results
        results.append({
            'se question': question,
            'aave question': aave_question,
            'actual answer': actual_answer[0],  # Save as the primary True/False value
            'se passage': passage,
            'se answer': se_answer_new,
            'aave passage': aave_passage,
            'aave answer': aave_answer,
            'se correct': se_correct,
            'aave correct': aave_correct
        })

        # Print progress indicator
        print(index + 1)
except Exception as e:
    print(f"Error processing translations: {e}")

# Convert the list to a DataFrame
results_df = pd.DataFrame(results)

# Save the DataFrame to a CSV file
output_file_path = '/content/drive/MyDrive/BoolQA_superglue_1000_4o.csv'
results_df.to_csv(output_file_path, index=False)
print("Evaluation results saved to CSV file.")


# In[ ]:


## Accuracy calculator
import pandas as pd
from statsmodels.stats.proportion import proportions_ztest
import numpy as np

# Define the file path for the cleaned BoolQA data
cleaned_boolqa_file_path_new = '/content/drive/MyDrive/Algoverse/AAVE Translations/GPT 4.0 Translations, Gemini-1.5-Pro Evaluation/BoolQ/BoolQA_superglue_1000_gemini.csv'

# Load the BoolQA data
boolqa_df_new = pd.read_csv(cleaned_boolqa_file_path_new)

# Ensure the 'actual answer' column is correctly parsed as booleans
boolqa_df_new['actual answer'] = boolqa_df_new['actual answer'].astype(str).str.strip().str.capitalize()
boolqa_df_new['actual answer'] = boolqa_df_new['actual answer'].map({'True': True, 'False': False})

# Standardize the 'se answer' and 'aave answer' columns
boolqa_df_new['se answer'] = boolqa_df_new['se answer'].astype(str).str.strip().str.capitalize()
boolqa_df_new['se answer'] = boolqa_df_new['se answer'].map({'True': True, 'False': False})

boolqa_df_new['aave answer'] = boolqa_df_new['aave answer'].astype(str).str.strip().str.capitalize()
boolqa_df_new['aave answer'] = boolqa_df_new['aave answer'].map({'True': True, 'False': False})

# Calculate accuracies for SE (Standard English) and AAVE
se_correct_boolqa_new = sum(boolqa_df_new['se answer'] == boolqa_df_new['actual answer'])
aave_correct_boolqa_new = sum(boolqa_df_new['aave answer'] == boolqa_df_new['actual answer'])

se_accuracy_boolqa_new = se_correct_boolqa_new / len(boolqa_df_new)
aave_accuracy_boolqa_new = aave_correct_boolqa_new / len(boolqa_df_new)

print(f"Standard English accuracy (BoolQA): {se_accuracy_boolqa_new}")
print(f"AAVE accuracy (BoolQA): {aave_accuracy_boolqa_new}")

count_boolqa_new = np.array([se_correct_boolqa_new, aave_correct_boolqa_new])

# Number of observations for BoolQA
nobs_boolqa_new = np.array([len(boolqa_df_new), len(boolqa_df_new)])

# Perform z-test for BoolQA
stat_boolqa_new, pval_boolqa_new = proportions_ztest(count_boolqa_new, nobs_boolqa_new)
print(f'z-statistic (BoolQA): {stat_boolqa_new}, p-value (BoolQA): {pval_boolqa_new}')


# In[ ]:


### File merger

import pandas as pd

# Define file paths
existing_file_path = '/content/drive/MyDrive/BoolQA_superglue_1000.csv'
new_file_path = '/content/drive/MyDrive/BoolQA_superglue_1000_Last_105.csv'  # Replace this with the actual path
combined_file_path = '/content/drive/MyDrive/BoolQA_superglue_combined.csv'

# Load existing and new data
existing_df = pd.read_csv(existing_file_path)
new_df = pd.read_csv(new_file_path)

# Combine the data
combined_df = pd.concat([existing_df, new_df], ignore_index=True)

# Save the combined data to a new CSV file
combined_df.to_csv(combined_file_path, index=False)
print("Combined data saved to CSV file.")

