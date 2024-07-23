#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install datasets')
get_ipython().system('pip install pandas')
get_ipython().system('pip install google-colab')


# In[ ]:


from google.colab import drive
drive.mount('/content/drive')


# In[ ]:


import pathlib
import textwrap
import pandas as pd
from IPython.display import display, Markdown
from google.colab import drive
import requests
import time

# Claude API configuration
API_KEY = 'API KEY'
HEADERS = {
    'x-api-key': API_KEY,
    'Content-Type': 'application/json',
    'anthropic-version': '2023-06-01'
}

MAX_REQUESTS_PER_MINUTE = 50

def compare_translations(text1, text2, row_number, request_count):
    while True:
        if request_count >= MAX_REQUESTS_PER_MINUTE:
            print("Rate limit reached. Waiting for 12 seconds...")
            time.sleep(12)
            request_count = 0  # Reset the request count after waiting

        try:
            messages = [
                {
                    "role": "user",
                    "content": f"Compare the following two translations and indicate which one is more accurate or if they are about the same. Translation 1: {text1}. Translation 2: {text2}. Which translation is more accurate in reflecting AAVE? Respond with 'Translation 1', 'Translation 2', or 'About the same'. You must choose one, don't say neither. Do not provide any additional text or explanation."
                }
            ]
            payload = {
                "model": "claude-3-5-sonnet-20240620",
                "system": "You are a helpful assistant.",
                "messages": messages,
                "max_tokens": 50
            }
            response = requests.post('https://api.anthropic.com/v1/messages', headers=HEADERS, json=payload)
            request_count += 1
            response_data = response.json()
            if 'content' in response_data:
                # Extract the actual message text
                message_content = response_data['content'][0]['text']
                print(f"Processed Row {row_number + 1}: {message_content.strip()}")
                return message_content.strip(), request_count
            else:
                print("Error: 'content' key not found in response_data")
                print("Response Data:", response_data)
                return "Comparison failed", request_count
        except requests.exceptions.RequestException as e:
            print(f"Request exception: {e}")
            return "Comparison failed", request_count
        except Exception as e:
            if 'rate_limit_error' in str(e):
                print(f"Rate limit error encountered. Waiting for 12 seconds before retrying...")
                time.sleep(12)
                request_count = 0  # Reset the request count after waiting
            else:
                print(f"Error comparing translations: {e}")
                return "Comparison failed", request_count
    print("Max attempts reached. Exiting...")
    return "Comparison failed", request_count

# Function to inspect column names and adjust DataFrame creation
def adjust_columns(df, expected_columns):
    columns = {col.lower().replace(' ', '_'): col for col in df.columns}
    adjusted_columns = {expected_col: columns.get(expected_col.lower(), expected_col) for expected_col in expected_columns}
    return df.rename(columns=adjusted_columns)

# Define expected columns for each dataset
expected_columns_boolq = ['Translated Passage', 'Translated Question']
expected_columns_copa = ['Translated Premise', 'Translated Choice1', 'Translated Choice2']
expected_columns_sst2 = ['Translated']
expected_columns_wsc = ['Translated Passage']

# Read the translation files
aave_translation_boolq = pd.read_csv('/content/drive/MyDrive/Algoverse/Values Translations/GPT 4.0 + Value/BoolQ/BoolQ_GPT.csv')
value_translation_boolq = pd.read_csv('/content/drive/MyDrive/Algoverse/Values Translations/GPT 4.0 + Value/BoolQ/BoolQ_value.csv')

value_translation_copa = pd.read_csv('/content/drive/MyDrive/Algoverse/Values Translations/GPT 4.0 + Value/Copa/COPA_Value(1) - Sheet1.csv')
aave_translation_copa = pd.read_csv('/content/drive/MyDrive/Algoverse/Values Translations/GPT 4.0 + Value/Copa/Copa_Translations.csv')

value_translation_sst2 = pd.read_csv('/content/drive/MyDrive/Algoverse/Values Translations/GPT 4.0 + Value/SST-2/SST-2_value.csv')
aave_translation_sst2 = pd.read_csv('/content/drive/MyDrive/Algoverse/Values Translations/GPT 4.0 + Value/SST-2/SST-2_translations.csv')

value_translation_wsc = pd.read_csv('/content/drive/MyDrive/Algoverse/Values Translations/GPT 4.0 + Value/Wsc/WSC_Value.csv')
aave_translation_wsc = pd.read_csv('/content/drive/MyDrive/Algoverse/Values Translations/GPT 4.0 + Value/Wsc/WSC_Translations.csv')

# Adjust the column names
adjusted_aave_boolq = adjust_columns(aave_translation_boolq, expected_columns_boolq)
adjusted_value_boolq = adjust_columns(value_translation_boolq, expected_columns_boolq)

adjusted_aave_copa = adjust_columns(aave_translation_copa, expected_columns_copa)
adjusted_value_copa = adjust_columns(value_translation_copa, expected_columns_copa)

adjusted_aave_sst2 = adjust_columns(aave_translation_sst2, expected_columns_sst2)
adjusted_value_sst2 = adjust_columns(value_translation_sst2, expected_columns_sst2)

adjusted_aave_wsc = adjust_columns(aave_translation_wsc, expected_columns_wsc)
adjusted_value_wsc = adjust_columns(value_translation_wsc, expected_columns_wsc)

# Function to process a dataset
def process_dataset(dataset_name, adjusted_aave, adjusted_value, expected_columns):
    print(f"Processing {dataset_name} dataset")
    data = pd.DataFrame()
    request_count = 0
    for idx, col in enumerate(expected_columns):
        print(f"Comparing column: {col}")
        try:
            aave_col = adjusted_aave[col]
            value_col = adjusted_value[col]
            data[f'aave_{col.lower().replace(" ", "_")}'] = aave_col
            data[f'value_{col.lower().replace(" ", "_")}'] = value_col
            result = []
            for i in range(len(data)):
                result_text, request_count = compare_translations(aave_col[i], value_col[i], i, request_count)
                result.append(result_text)
            data[f'{col.lower().replace(" ", "_")}_comparison_result'] = result
            print(f"Finished comparing column: {col}")
        except Exception as e:
            print(f"Error processing column {col}: {e}")
    data.to_csv(f'/content/drive/MyDrive/Algoverse/Validation Scores/Comparison GPT 4.0 vs Value (Claude-Sonnet-3.5)/{dataset_name}_comparison_results.csv', index=False)
    print(f"{dataset_name} comparison completed")

# Process all datasets
process_dataset('BoolQ', adjusted_aave_boolq, adjusted_value_boolq, expected_columns_boolq)
process_dataset('COPA', adjusted_aave_copa, adjusted_value_copa, expected_columns_copa)
process_dataset('SST2', adjusted_aave_sst2, adjusted_value_sst2, expected_columns_sst2)
process_dataset('WSC', adjusted_aave_wsc, adjusted_value_wsc, expected_columns_wsc)

print("All comparisons completed")


# In[ ]:


import pandas as pd

def calculate_average_percentages(comparison_file, columns):
    # Read the comparison results CSV file
    df = pd.read_csv(comparison_file)

    # Initialize counters
    total = 0
    translation_1_count = 0
    translation_2_count = 0
    about_the_same_count = 0

    # Iterate over the specified comparison result columns
    for col in columns:
        if 'comparison_result' in col:
            col_total = df[col].count()
            total += col_total
            translation_1_count += (df[col] == 'Translation 1').sum()
            translation_2_count += (df[col] == 'Translation 2').sum()
            about_the_same_count += ((df[col] != 'Translation 1') & (df[col] != 'Translation 2')).sum()

    # Calculate percentages
    translation_1_percentage = (translation_1_count / total) * 100 if total != 0 else 0
    translation_2_percentage = (translation_2_count / total) * 100 if total != 0 else 0
    about_the_same_percentage = (about_the_same_count / total) * 100 if total != 0 else 0

    return translation_1_percentage, translation_2_percentage, about_the_same_percentage

# Define the comparison result files and their respective columns
comparison_files = {
    'BoolQ': {
        'file': '/content/drive/MyDrive/Algoverse/Validation Scores/Comparison GPT 4.0 vs Value (Gemini-1.5-Pro)/BoolQ_comparison_results.csv',
        'columns': ['translated_passage_comparison_result', 'translated_question_comparison_result']
    },
    'COPA': {
        'file': '/content/drive/MyDrive/Algoverse/Validation Scores/Comparison GPT 4.0 vs Value (Gemini-1.5-Pro)/COPA_comparison_results.csv',
        'columns': ['translated_premise_comparison_result', 'translated_choice1_comparison_result', 'translated_choice2_comparison_result']
    },
    'SST2': {
        'file': '/content/drive/MyDrive/Algoverse/Validation Scores/Comparison GPT 4.0 vs Value (Gemini-1.5-Pro)/SST2_comparison_results.csv',
        'columns': ['translated_comparison_result']
    },
    'WSC': {
        'file': '/content/drive/MyDrive/Algoverse/Validation Scores/Comparison GPT 4.0 vs Value (Gemini-1.5-Pro)/WSC_comparison_results.csv',
        'columns': ['translated_passage_comparison_result']
    }
}

# Calculate and print the results to the output
for dataset_name, info in comparison_files.items():
    file_path = info['file']
    columns = info['columns']
    translation_1_percentage, translation_2_percentage, about_the_same_percentage = calculate_average_percentages(file_path, columns)
    print(f"{dataset_name}:")
    print(f"Percentage preferring AAVE translation: {translation_1_percentage:.2f}%")
    print(f"Percentage preferring VALUE translation: {translation_2_percentage:.2f}%")
    print(f"Percentage saying 'About the same': {about_the_same_percentage:.2f}%\n")

print("Comparison percentages calculated and printed to output")

