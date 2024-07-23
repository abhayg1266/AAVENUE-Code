#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install -q -U google-generativeai')
get_ipython().system('pip install datasets')
get_ipython().system('pip install pandas')
get_ipython().system('pip install google-colab')


# In[ ]:


import pathlib
import textwrap

import google.generativeai as genai
from google.colab import drive
import pandas as pd
from IPython.display import display, Markdown


# In[ ]:


# Function to format text as Markdown
def to_markdown(text):
    text = text.replace('•', '  *')
    return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))

# Used to securely store your API key
from google.colab import userdata

# Fetch the API key
# GOOGLE_API_KEY = userdata.get('GOOGLE_API_KEY')
GOOGLE_API_KEY = 'API KEY'
genai.configure(api_key=GOOGLE_API_KEY)


# In[ ]:


model = genai.GenerativeModel('gemini-1.5-pro')


# In[ ]:


from google.colab import drive
drive.mount('/content/drive')


# In[ ]:


import pandas as pd

def compare_translations(text1, text2):
    try:
        response = model.generate_content(f"Compare the following two translations and indicate which one is more accurate or if they are about the same. Translation 1: {text1}. Translation 2: {text2}. Which translation is more accurate in reflecting AAVE? Respond with 'Translation 1', 'Translation 2', or 'About the same'. You must choose one, don't say neither. Do not provide any additional text or explanation")
        print("Response received from API:", response.text)
        return response.text.strip()
    except Exception as e:
        print(f"Error comparing translations: {e}")
        return "Comparison failed"

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
aave_translation_boolq = pd.read_csv('/content/drive/MyDrive/Algoverse/Values Translations/GPT 4.0 + Value/BoolQ/BoolQ_Translations.csv')
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
    for col in expected_columns:
        print(f"Comparing column: {col}")
        try:
            aave_col = adjusted_aave[col]
            value_col = adjusted_value[col]
            data[f'aave_{col.lower().replace(" ", "_")}'] = aave_col
            data[f'value_{col.lower().replace(" ", "_")}'] = value_col
            data[f'{col.lower().replace(" ", "_")}_comparison_result'] = data.apply(
                lambda row: compare_translations(row[f'aave_{col.lower().replace(" ", "_")}'], row[f'value_{col.lower().replace(" ", "_")}']), axis=1
            )
            print(f"Finished comparing column: {col}")
        except Exception as e:
            print(f"Error processing column {col}: {e}")
    data.to_csv(f'/content/drive/MyDrive/Algoverse/{dataset_name}_comparison_results.csv', index=False)
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
            valid_values = df[col].isin(['Translation 1', 'Translation 2', 'About the same'])
            col_total = valid_values.sum()
            total += col_total
            translation_1_count += (df[col] == 'Translation 1').sum()
            translation_2_count += (df[col] == 'Translation 2').sum()
            about_the_same_count += (df[col] == 'About the same').sum()

    # Calculate percentages
    translation_1_percentage = (translation_1_count / total) * 100 if total != 0 else 0
    translation_2_percentage = (translation_2_count / total) * 100 if total != 0 else 0
    about_the_same_percentage = (about_the_same_count / total) * 100 if total != 0 else 0

    return translation_1_percentage, translation_2_percentage, about_the_same_percentage

# Define the comparison result files and their respective columns
comparison_files = {
    'BoolQ': {
        'file': '/content/drive/MyDrive/Algoverse/Validation Scores/Comparison GPT 4.0 vs Value (Claude-Sonnet-3.5)/BoolQ_comparison_results.csv',
        'columns': ['translated_passage_comparison_result', 'translated_question_comparison_result']
    },
    'COPA': {
        'file': '/content/drive/MyDrive/Algoverse/Validation Scores/Comparison GPT 4.0 vs Value (Claude-Sonnet-3.5)/COPA_comparison_results.csv',
        'columns': ['translated_premise_comparison_result', 'translated_choice1_comparison_result', 'translated_choice2_comparison_result']
    },
    'SST2': {
        'file': '/content/drive/MyDrive/Algoverse/Validation Scores/Comparison GPT 4.0 vs Value (Claude-Sonnet-3.5)/SST2_comparison_results.csv',
        'columns': ['translated_comparison_result']
    },
    'WSC': {
        'file': '/content/drive/MyDrive/Algoverse/Validation Scores/Comparison GPT 4.0 vs Value (Claude-Sonnet-3.5)/WSC_comparison_results.csv',
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

