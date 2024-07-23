#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install openai==0.28')
get_ipython().system('pip install pandas google-colab gdown')

import openai
import pandas as pd
from google.colab import drive
import numpy as np

openai.api_key = "API KEY"
openai.organization = "Organization"

# Mount Google Drive
drive.mount('/content/drive')


# In[ ]:


import openai
import pandas as pd
from google.colab import drive
import numpy as np

# Function to compare translations using GPT-4.0-turbo
def compare_translations(text1, text2):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an expert in AAVE (African American Vernacular English). Compare the following two translations and indicate which one is more accurate or if they are about the same."},
                {"role": "user", "content": f"Translation 1: {text1}\nTranslation 2: {text2}\nWhich translation is more accurate in reflecting AAVE? Respond with 'Translation 1', 'Translation 2', or 'About the same'. You must choose one, don't say neither. Do not provide any additional text or explanation"}
            ]
        )
        return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        print(f"Error comparing translations: {e}")
        return "Comparison failed"

# Function to process a dataset and save comparison results
def process_dataset(dataset_name, aave_translation_file, value_translation_file, columns_mapping):
    print(f"Processing {dataset_name} dataset")

    # Load translations
    aave_translation = pd.read_csv(aave_translation_file)
    value_translation = pd.read_csv(value_translation_file)

    # Create DataFrame with necessary columns
    data = pd.DataFrame({
        f'aave_{key}': aave_translation[val] for key, val in columns_mapping.items()
    })
    for key, val in columns_mapping.items():
        data[f'value_{key}'] = value_translation[val]

    # Process the entire CSV file
    processed_data = []
    for i, row in data.iterrows():
        comparison_results = {}
        for key in columns_mapping.keys():
            comparison_results[f'{key}_comparison_result'] = compare_translations(row[f'aave_{key}'], row[f'value_{key}'])
        processed_data.append({
            **{f'aave_{key}': row[f'aave_{key}'] for key in columns_mapping.keys()},
            **{f'value_{key}': row[f'value_{key}'] for key in columns_mapping.keys()},
            **comparison_results
        })
        print(f"Processed row {i + 1}")

    # Create a DataFrame from the processed data
    processed_df = pd.DataFrame(processed_data)

    # Save the resulting DataFrame to a new CSV file
    output_file_path = f'/content/drive/MyDrive/Algoverse/Validation Scores/Comparison GPT 4.0 vs Value (GPT 3.5)/{dataset_name}_comparison_results.csv'
    processed_df.to_csv(output_file_path, index=False)
    print(f"{dataset_name} results saved to CSV file at {output_file_path}")

# Process each dataset
datasets = {
    'BoolQ': {
        'aave_translation_file': '/content/drive/MyDrive/Algoverse/Values Translations/GPT 4.0 + Value/BoolQ/BoolQ_GPT.csv',
        'value_translation_file': '/content/drive/MyDrive/Algoverse/Values Translations/GPT 4.0 + Value/BoolQ/BoolQ_value.csv',
        'columns_mapping': {
            'passage': 'Translated Passage',
            'question': 'Translated Question'
        }
    },
    'COPA': {
        'aave_translation_file': '/content/drive/MyDrive/Algoverse/Values Translations/GPT 4.0 + Value/Copa/Copa_Translations.csv',
        'value_translation_file': '/content/drive/MyDrive/Algoverse/Values Translations/GPT 4.0 + Value/Copa/COPA_Value(1) - Sheet1.csv',
        'columns_mapping': {
            'premise': 'Translated Premise',
            'choice1': 'Translated Choice1',
            'choice2': 'Translated Choice2'
        }
    },
    'SST2': {
        'aave_translation_file': '/content/drive/MyDrive/Algoverse/Values Translations/GPT 4.0 + Value/SST-2/SST-2_translations.csv',
        'value_translation_file': '/content/drive/MyDrive/Algoverse/Values Translations/GPT 4.0 + Value/SST-2/SST-2_value.csv',
        'columns_mapping': {
            'translation': 'Translated'
        }
    },
    'WSC': {
        'aave_translation_file': '/content/drive/MyDrive/Algoverse/Values Translations/GPT 4.0 + Value/Wsc/WSC_Translations.csv',
        'value_translation_file': '/content/drive/MyDrive/Algoverse/Values Translations/GPT 4.0 + Value/Wsc/WSC_Value.csv',
        'columns_mapping': {
            'passage': 'Translated Passage'
        }
    }
}

for dataset_name, info in datasets.items():
    process_dataset(dataset_name, info['aave_translation_file'], info['value_translation_file'], info['columns_mapping'])

print("All dataset comparisons completed")


# In[ ]:


# Function to calculate average percentages
def calculate_average_percentages(comparison_file, columns):
    df = pd.read_csv(comparison_file)

    total = 0
    translation_1_count = 0
    translation_2_count = 0
    about_the_same_count = 0

    for col in columns:
        if 'comparison_result' in col:
            col_total = df[col].count()
            total += col_total
            translation_1_count += (df[col] == 'Translation 1').sum()
            translation_2_count += (df[col] == 'Translation 2').sum()
            about_the_same_count += ((df[col] != 'Translation 1') & (df[col] != 'Translation 2')).sum()

    translation_1_percentage = (translation_1_count / total) * 100 if total != 0 else 0
    translation_2_percentage = (translation_2_count / total) * 100 if total != 0 else 0
    about_the_same_percentage = (about_the_same_count / total) * 100 if total != 0 else 0

    return translation_1_percentage, translation_2_percentage, about_the_same_percentage

# Calculate and print results for each dataset
comparison_files = {
    'BoolQ': {
        'file': '/content/drive/MyDrive/Algoverse/Validation Scores/Comparison GPT 4.0 vs Value (GPT 3.5)/BoolQ_comparison_results.csv',
        'columns': ['passage_comparison_result', 'question_comparison_result']
    },
    'COPA': {
        'file': '/content/drive/MyDrive/Algoverse/Validation Scores/Comparison GPT 4.0 vs Value (GPT 3.5)/COPA_comparison_results.csv',
        'columns': ['premise_comparison_result', 'choice1_comparison_result', 'choice2_comparison_result']
    },
    'SST2': {
        'file': '/content/drive/MyDrive/Algoverse/Validation Scores/Comparison GPT 4.0 vs Value (GPT 3.5)/SST2_comparison_results.csv',
        'columns': ['translation_comparison_result']
    },
    'WSC': {
        'file': '/content/drive/MyDrive/Algoverse/Validation Scores/Comparison GPT 4.0 vs Value (GPT 3.5)/WSC_comparison_results.csv',
        'columns': ['passage_comparison_result']
    }
}

for dataset_name, info in comparison_files.items():
    file_path = info['file']
    columns = info['columns']
    translation_1_percentage, translation_2_percentage, about_the_same_percentage = calculate_average_percentages(file_path, columns)
    print(f"{dataset_name}:")
    print(f"Percentage preferring AAVE translation: {translation_1_percentage:.2f}%")
    print(f"Percentage preferring VALUE translation: {translation_2_percentage:.2f}%")
    print(f"Percentage saying 'About the same': {about_the_same_percentage:.2f}%\n")

print("Comparison percentages calculated and printed to output")

