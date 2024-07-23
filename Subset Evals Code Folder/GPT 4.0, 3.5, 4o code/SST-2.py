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
def determine_sentiment(passage):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Assess the sentiment of the passage and respond with 'Positive' or 'Negative'. Ensure the response contains only these terms without any additional characters or explanations, regardless of passage length. If passage is small still reply with 'Positive' or 'Negative' those are the only 2 things you're allowed to output"},
            {"role": "user", "content": f"Analyze the sentiment of the passage:\n\"{passage}\""}
        ],
        max_tokens=50
    )
    return response.choices[0].message['content'] if response.choices else "No translation found."

# Function to convert numerical sentiment label to text
def convert(answer):  # turns number 0/1 into positive/negative
    return "Positive" if answer == 1 else "Negative"

# Function to check correctness
def check_correctness(predicted, actual):
    return 1 if predicted.lower() == actual.lower() else 0

# Load translations from CSV
csv_file_path = '/content/drive/MyDrive/Algoverse/AAVE Translations/GPT 4.0 Translations, 4.0 Evaluations/SST-2/SST-2_super-glue_1000_4.0.csv'
translations_df = pd.read_csv(csv_file_path)

# Initialize results list
results = []

try:
    for index, row in translations_df.iterrows():  # Process the entire CSV
        sentence = row['Original']
        translated_sentence = row['Translated']
        actual_sentiment = row['Original Sentiment']  # Actual sentiment from CSV
        se_correct_existing = row['SE Correct']  # Existing SE Correct value
        aave_correct_existing = row['AAVE Correct']  # Existing AAVE Correct value

        # Determine sentiments
        se_sentiment = determine_sentiment(sentence)
        aave_sentiment = determine_sentiment(translated_sentence)

        # Check correctness
        se_correct = check_correctness(se_sentiment, actual_sentiment)
        aave_correct = check_correctness(aave_sentiment, actual_sentiment)

        # Append results
        results.append({
            'Original': sentence,
            'Translated': translated_sentence,
            'Original Sentiment': actual_sentiment,
            'SE Sentiment': se_sentiment,
            'AAVE Sentiment': aave_sentiment,
            'SE Correct Existing': se_correct_existing,
            'AAVE Correct Existing': aave_correct_existing,
            'SE Correct New': se_correct,
            'AAVE Correct New': aave_correct
        })

        # Print progress indicator
        print(index + 1)
except Exception as e:
    print(f"Error processing translations: {e}")

# Exporting results to CSV
results_df = pd.DataFrame(results)
output_file_path = '/content/drive/MyDrive/Algoverse/AAVE Translations/GPT 4.0 Translations, 3.5 Evaluations/SST-2/SST2_1000_4_3.5.csv'
results_df.to_csv(output_file_path, index=False)
print("Evaluation results saved to CSV file.")


# In[ ]:


import pandas as pd

def calculate_accuracy(csv_file_path):
    # Read the CSV file
    df = pd.read_csv(csv_file_path)

    # Calculate the total number of rows
    total_rows = len(df)

    # Calculate the number of correct and incorrect predictions for SE and AAVE
    se_correct_count = df['SE Correct New'].sum()
    aave_correct_count = df['AAVE Correct New'].sum()

    se_incorrect_count = total_rows - se_correct_count
    aave_incorrect_count = total_rows - aave_correct_count

    # Calculate the percentages
    se_correct_percentage = (se_correct_count / total_rows) * 100
    aave_correct_percentage = (aave_correct_count / total_rows) * 100
    se_incorrect_percentage = (se_incorrect_count / total_rows) * 100
    aave_incorrect_percentage = (aave_incorrect_count / total_rows) * 100

    # Print the results
    print(f"SE Correct: {se_correct_percentage:.2f}%")
    print(f"SE Incorrect: {se_incorrect_percentage:.2f}%")
    print(f"AAVE Correct: {aave_correct_percentage:.2f}%")
    print(f"AAVE Incorrect: {aave_incorrect_percentage:.2f}%")

    return {
        'se_correct_percentage': se_correct_percentage,
        'se_incorrect_percentage': se_incorrect_percentage,
        'aave_correct_percentage': aave_correct_percentage,
        'aave_incorrect_percentage': aave_incorrect_percentage
    }

# Usage example
csv_file_path = '/content/drive/MyDrive/Algoverse/AAVE Translations/GPT 4.0 Translations, 3.5 Evaluations/SST-2/SST2_1000_4_3.5.csv'
accuracy_results = calculate_accuracy(csv_file_path)


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

