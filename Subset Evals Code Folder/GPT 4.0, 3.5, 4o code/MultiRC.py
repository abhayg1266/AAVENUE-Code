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


def determine_answer(paragraph, question, answer):
    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You can only say 2 things, '0' & '1'. Determine if the answer to the question based on the given paragraph is true or false. Only respond with '0' for false or '1' for true."},
            {"role": "user", "content": f"Paragraph: \"{paragraph}\"\nQuestion: \"{question}\"\nAnswer: \"{answer}\""}
        ],
        max_tokens=25
    )
    return response.choices[0].message['content'] if response.choices else "No response found."

# Function to check correctness
def check_correctness(predicted, actual):
    return '✔️' if predicted == actual else '❌'

# Load translations from CSV
csv_file_path = '/content/drive/MyDrive/Algoverse/AAVE Translations/GPT 4.0 Translations/Multi-RC/4.0/MultiRC_1000_4.0.csv'
translations_df = pd.read_csv(csv_file_path)

# Initialize results list
results = []

try:
    for index, row in translations_df.iterrows():  # Process the entire CSV
        paragraph = row['Paragraph']
        translated_paragraph = row['Translated Paragraph']
        question = row['Question']
        translated_question = row['Translated Question']
        answer = row['Answer']
        translated_answer = row['Translated Answer']
        actual_label = row['Actual Label']

        # Determine answers
        se_response = determine_answer(paragraph, question, answer)
        aave_response = determine_answer(translated_paragraph, translated_question, translated_answer)

        # Append results
        results.append({
            'Paragraph': paragraph,
            'Translated Paragraph': translated_paragraph,
            'Question': question,
            'Translated Question': translated_question,
            'Answer': answer,
            'Translated Answer': translated_answer,
            'Actual Label': str(actual_label),
            'SE Response': se_response,
            'AAVE Response': aave_response,
            'SE Correct': check_correctness(se_response, str(actual_label)),
            'AAVE Correct': check_correctness(aave_response, str(actual_label))
        })

        # Print progress indicator
        print(index + 1)
except Exception as e:
    print(f"Error processing translations: {e}")

# Convert the list to a DataFrame
results_df = pd.DataFrame(results)

# Save the DataFrame to a CSV file
output_file_path = '/content/drive/MyDrive/Algoverse/AAVE Translations/GPT 4.0 Translation, 4o evaluation/MultiRC/MultiRC_1000_4o.csv'
results_df.to_csv(output_file_path, index=False)
print("Evaluation results saved to CSV file.")


# In[ ]:


import pandas as pd
from statsmodels.stats.proportion import proportions_ztest
import numpy as np

# Define the file path for the results
results_file_path = '/content/drive/MyDrive/Algoverse/AAVE Translations/GPT 4.0 Translation, 4o evaluation/MultiRC/MultiRC_1000_4o.csv'

# Load the results data
results_df = pd.read_csv(results_file_path)

# Calculate accuracies for SE (Standard English) and AAVE
se_correct = sum(results_df['SE Correct'] == '✔️')
aave_correct = sum(results_df['AAVE Correct'] == '✔️')

se_accuracy = se_correct / len(results_df)
aave_accuracy = aave_correct / len(results_df)

print(f"Standard English accuracy (Multi-RC): {se_accuracy:.3f}")
print(f"AAVE accuracy (Multi-RC): {aave_accuracy:.3f}")

count = np.array([se_correct, aave_correct])

# Number of observations for Multi-RC
nobs = np.array([len(results_df), len(results_df)])

# Perform z-test for Multi-RC
stat, pval = proportions_ztest(count, nobs)
print(f'z-statistic (Multi-RC): {stat}, p-value (Multi-RC): {pval}')


# In[ ]:


### File merger

import pandas as pd

# Define file paths
existing_file_path = '/content/drive/MyDrive/Algoverse Folder Only/Copa_train_400_4_3.5.csv'
new_file_path = '/content/drive/MyDrive/Algoverse Folder Only/Copa_validation_100_4_3.5.csv'  # Replace this with the actual path
combined_file_path = '/content/drive/MyDrive/Algoverse Folder Only/Copa_superglue_4-3.5.csv'

# Load existing and new data
existing_df = pd.read_csv(existing_file_path)
new_df = pd.read_csv(new_file_path)

# Combine the data
combined_df = pd.concat([existing_df, new_df], ignore_index=True)

# Save the combined data to a new CSV file
combined_df.to_csv(combined_file_path, index=False)
print("Combined data saved to CSV file.")

