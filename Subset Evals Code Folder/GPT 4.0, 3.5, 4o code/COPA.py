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
# Function to query GPT for cause-effect prediction
def predict_cause_effect(premise, choices):
    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You must determine if the presented choice is the cause or the effect of the premise given. Only respond with '0' if the first choice is correct or '1' if the second choice is correct. You're only allowed to say those 2 things"},
            {"role": "user", "content": f"Premise: \"{premise}\"\nChoice1: \"{choices[0]}\"\nChoice2: \"{choices[1]}\""}
        ],
        max_tokens=25
    )
    return response.choices[0].message['content'] if response.choices else "No response found."

# Load translations from CSV
csv_file_path = '/content/Copa_superglue_500_4.0.csv'
translations_df = pd.read_csv(csv_file_path)

# Initialize results list
results = []

try:
    for index, row in translations_df.iterrows():  # Process the entire CSV
        premise = row['Premise']
        translated_premise = row['Translated Premise']
        choice1 = row['Choice1']
        translated_choice1 = row['Translated Choice1']
        choice2 = row['Choice2']
        translated_choice2 = row['Translated Choice2']
        actual_label = row['Actual Label']

        # Predictions
        se_response = predict_cause_effect(premise, [choice1, choice2])
        aave_response = predict_cause_effect(translated_premise, [translated_choice1, translated_choice2])

        # Append results
        results.append({
            'Premise': premise,
            'Translated Premise': translated_premise,
            'Choice1': choice1,
            'Translated Choice1': translated_choice1,
            'Choice2': choice2,
            'Translated Choice2': translated_choice2,
            'Actual Label': actual_label,
            'SE Response': se_response,
            'AAVE Response': aave_response
        })

        # Print progress indicator
        print(index + 1)
except Exception as e:
    print(f"Error processing translations: {e}")

# Convert the list to a DataFrame
results_df = pd.DataFrame(results)

# Save the DataFrame to a CSV file
output_file_path = '/content/Copa_superglue_500_4o.csv'
results_df.to_csv(output_file_path, index=False)
print("Evaluation results saved to CSV file.")


# In[ ]:


## Accuracy calculator
import pandas as pd
from statsmodels.stats.proportion import proportions_ztest
import numpy as np

# Define the file path for the cleaned COPA data
cleaned_copa_file_path = '/content/drive/MyDrive/Algoverse/AAVE Translations/GPT 4.0 Translation, 4o evaluation/COPA/Copa_superglue_500_4o.csv'

# Load the COPA data
copa_df = pd.read_csv(cleaned_copa_file_path)

# Calculate accuracies for SE (Standard English) and AAVE
se_correct_copa = sum(copa_df['SE Response'] == copa_df['Actual Label'])
aave_correct_copa = sum(copa_df['AAVE Response'] == copa_df['Actual Label'])

se_accuracy_copa = se_correct_copa / len(copa_df)
aave_accuracy_copa = aave_correct_copa / len(copa_df)

print(f"Standard English accuracy (COPA): {se_accuracy_copa}")
print(f"AAVE accuracy (COPA): {aave_accuracy_copa}")

count_copa = np.array([se_correct_copa, aave_correct_copa])

# Number of observations for COPA
nobs_copa = np.array([len(copa_df), len(copa_df)])

# Perform z-test for COPA
stat_copa, pval_copa = proportions_ztest(count_copa, nobs_copa)
print(f'z-statistic (COPA): {stat_copa}, p-value (COPA): {pval_copa}')


# In[ ]:


### File merger

import pandas as pd

# Define file paths
existing_file_path = '/content/drive/MyDrive/Copa_400_4o.csv'
new_file_path = '/content/drive/MyDrive/Copa_100_4o.csv'  # Replace this with the actual path
combined_file_path = '/content/drive/MyDrive/Copa_500_4o.csv'

# Load existing and new data
existing_df = pd.read_csv(existing_file_path)
new_df = pd.read_csv(new_file_path)

# Combine the data
combined_df = pd.concat([existing_df, new_df], ignore_index=True)

# Save the combined data to a new CSV file
combined_df.to_csv(combined_file_path, index=False)
print("Combined data saved to CSV file.")

