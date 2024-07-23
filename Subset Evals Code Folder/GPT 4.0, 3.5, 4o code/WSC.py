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
# Function to determine the pronoun reference
def determine_pronoun_reference(passage, pronoun, span1, span2):
    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You can only respond with 1 of these 2 things: '0' and '1'. You are reading a passage and determining the antecedent of a pronoun. Only respond with '0' if the pronoun refers to Span1 or '1' if the pronoun refers to Span2. Only 1 number should be your response, '0' or '1'."},
            {"role": "user", "content": f"Passage: \"{passage}\"\nPronoun: \"{pronoun}\"\nSpan1: \"{span1}\"\nSpan2: \"{span2}\""}
        ],
        max_tokens=50
    )
    return response.choices[0].message['content'].strip() if response.choices else "No response found."

# Load translations from CSV
csv_file_path = '/content/WSC_630_4.0.csv'
translations_df = pd.read_csv(csv_file_path)

# Initialize results list
results = []

try:
    for index, row in translations_df.iterrows():  # Process the entire CSV
        passage = row['Original Passage']
        translated_passage = row['Translated Passage']
        pronoun = row['Pronoun']
        span1 = row['Span1']
        translated_span1 = row['Translated Span1']
        span2 = row['Span2']
        translated_span2 = row['Translated Span2']
        actual_reference = str(row['Actual Reference'])

        # Determine references
        se_reference = determine_pronoun_reference(passage, pronoun, span1, span2)
        aave_reference = determine_pronoun_reference(translated_passage, pronoun, translated_span1, translated_span2)

        # Append results
        results.append({
            'Original Passage': passage,
            'Translated Passage': translated_passage,
            'Pronoun': pronoun,
            'Span1': span1,
            'Translated Span1': translated_span1,
            'Span2': span2,
            'Translated Span2': translated_span2,
            'Actual Reference': actual_reference,
            'SE Reference': se_reference,
            'AAVE Reference': aave_reference
        })

        # Print progress indicator
        print(index + 1)
except Exception as e:
    print(f"Error processing translations: {e}")

# Convert the list to a DataFrame
results_df = pd.DataFrame(results)

# Save the DataFrame to a CSV file
output_file_path = '/content/WSC_630_4o.csv'
results_df.to_csv(output_file_path, index=False)
print("Evaluation results saved to CSV file.")


# In[ ]:


import pandas as pd
from statsmodels.stats.proportion import proportions_ztest
import numpy as np

# Define the file path for the cleaned WSC data
cleaned_wsc_file_path = '/content/drive/MyDrive/Algoverse/AAVE Translations/GPT 4.0 Translation, 4o evaluation/WSC/WSC_630_4o.csv'

# Load the WSC data
wsc_df = pd.read_csv(cleaned_wsc_file_path)

# Ensure the 'Actual Reference' column is correctly parsed as integers
wsc_df['Actual Reference'] = pd.to_numeric(wsc_df['Actual Reference'], errors='coerce').fillna(0).astype(int)

# Ensure the 'SE Reference' and 'AAVE Reference' columns are correctly parsed as integers
wsc_df['SE Reference'] = pd.to_numeric(wsc_df['SE Reference'], errors='coerce').fillna(0).astype(int)
wsc_df['AAVE Reference'] = pd.to_numeric(wsc_df['AAVE Reference'], errors='coerce').fillna(0).astype(int)

# Calculate accuracies for SE (Standard English) and AAVE
se_correct_wsc = sum(wsc_df['SE Reference'] == wsc_df['Actual Reference'])
aave_correct_wsc = sum(wsc_df['AAVE Reference'] == wsc_df['Actual Reference'])

se_accuracy_wsc = se_correct_wsc / len(wsc_df)
aave_accuracy_wsc = aave_correct_wsc / len(wsc_df)

print(f"Standard English accuracy (WSC): {se_accuracy_wsc}")
print(f"AAVE accuracy (WSC): {aave_accuracy_wsc}")

count_wsc = np.array([se_correct_wsc, aave_correct_wsc])

# Number of observations for WSC
nobs_wsc = np.array([len(wsc_df), len(wsc_df)])

# Perform z-test for WSC
stat_wsc, pval_wsc = proportions_ztest(count_wsc, nobs_wsc)
print(f'z-statistic (WSC): {stat_wsc}, p-value (WSC): {pval_wsc}')


# In[ ]:


### File merger

import pandas as pd

# Define file paths
existing_file_path = '/content/drive/MyDrive/Algoverse Folder Only/WSC_validation_4-3.5.csv'
new_file_path = '/content/drive/MyDrive/Algoverse Folder Only/WSC_train_3.5.csv'  # Replace this with the actual path
combined_file_path = '/content/drive/MyDrive/Algoverse Folder Only/WSC_630_4-3.5.csv'

# Load existing and new data
existing_df = pd.read_csv(existing_file_path)
new_df = pd.read_csv(new_file_path)

# Combine the data
combined_df = pd.concat([existing_df, new_df], ignore_index=True)

# Save the combined data to a new CSV file
combined_df.to_csv(combined_file_path, index=False)
print("Combined data saved to CSV file.")

