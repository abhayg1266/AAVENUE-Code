#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install gdown pandas


# In[2]:


from google.colab import drive
drive.mount ('/content/drive')


# In[ ]:


import gdown
import pandas as pd

results = []

# Replace 'your_file_id' with the actual file ID of your CSV file
file_id = '1XOQ-_sMJNfk7eFk-mejtU7Tt-0KOMvtD'
url = f'https://drive.google.com/uc?id={file_id}'

# Download the file
output = 'downloaded_file.csv'
gdown.download(url, output, quiet=False)

# Load the CSV file into a pandas DataFrame
df = pd.read_csv(output)

# Specify the column name that contains the 0s and 1s
column_name = 'Passage BLEU_score_under_10'
column_name1 = 'Passage BLEU_score_under_20'
column_name2 = 'Passage BLEU_score_under_30'
column_name3 = 'Passage BLEU_score_under_50'
column_name4 = 'Passage BLEU_score_under_70'
column_name5 = 'Question BLEU_score_under_10'
column_name6 = 'Question BLEU_score_under_20'
column_name7 = 'Question BLEU_score_under_30'
column_name8 = 'Question BLEU_score_under_50'
column_name9 = 'Question BLEU_score_under_70'

# Calculate the sum of the values in the specified column
sum_values = df[column_name].sum()
sum_values1 = df[column_name1].sum()
sum_values2 = df[column_name2].sum()
sum_values3 = df[column_name3].sum()
sum_values4 = df[column_name4].sum()
sum_values5 = df[column_name5].sum()
sum_values6 = df[column_name6].sum()
sum_values7 = df[column_name7].sum()
sum_values8 = df[column_name8].sum()
sum_values9 = df[column_name9].sum()

# Calculate the total number of values in the specified column
total_values = df[column_name].count()
total_values1 = df[column_name1].count()
total_values2 = df[column_name2].count()
total_values3 = df[column_name3].count()
total_values4 = df[column_name4].count()
total_values5 = df[column_name5].count()
total_values6 = df[column_name6].count()
total_values7 = df[column_name7].count()
total_values8 = df[column_name8].count()
total_values9 = df[column_name9].count()

# Calculate the result by dividing the sum by the total number of values
result = sum_values / total_values
result1 = sum_values1 / total_values1
result2 = sum_values2 / total_values2
result3 = sum_values3 / total_values3
result4 = sum_values4 / total_values4
result5 = sum_values5 / total_values5
result6 = sum_values6 / total_values6
result7 = sum_values7 / total_values7
result8 = sum_values8 / total_values8
result9 = sum_values9 / total_values9

# Print the result
print(f"Sum of values: {sum_values}")
print(f"Total number of values: {total_values}")
print(f"Result: {result}")

results.append({
    'Passage BLEU_under_10': str(result*100) + "%",
    'Passage BLEU_under_20': str(result1*100) + "%",
    'Passage BLEU_under_30': str(result2*100) + "%",
    'Passage BLEU_under_50': str(result3*100) + "%",
    'Passage BLEU_under_70': str(result4*100) + "%",
    'Question BLEU_under_10': str(result5*100) + "%",
    'Question BLEU_under_20': str(result6*100) + "%",
    'Question BLEU_under_30': str(result7*100) + "%",
    'Question BLEU_under_50': str(result8*100) + "%",
    'Question BLEU_under_70': str(result9*100) + "%"

})

# Create a DataFrame from the results list
results_df = pd. DataFrame (results)
# Save results to a new CSV file
output_results_csv = '/content/drive/MyDrive/Algoverse/BLEU Scores/percentage_boolq.csv'
results_df.to_csv(output_results_csv, index=False)
print(f"Results with BLEU scores saved to '{output_results_csv}'.")


# In[ ]:




