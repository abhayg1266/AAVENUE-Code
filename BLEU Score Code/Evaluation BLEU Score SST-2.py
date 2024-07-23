#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install sacrebleu


# In[2]:


from google.colab import drive
drive.mount('/content/drive')


# In[8]:


# Required libraries
import gdown
import pandas as pd
import sacrebleu

# File IDs for downloading from Google Drive
file_id = '1QW0niXLpsi9S5kGl530hXXCOfo_nd4NY'  # gpt 4.0
file_id1 = '1-2TJA-hKAQ3xwkptL9VJcdAg71LZpuiI'  # gemini pro
file_id2 = '1-J5aeRMjMrW67QIFz7cLVCs7w3OprLAb'  # gemini flash
file_id3 = '1-C6OzMv2xGex-DQ7f_OzHG5xTahErVqD'  # gpt 4o
file_id4 = '1yPev_A-EC96SBNWhP-xEkCQ_KpCorJd0'  # gpt 3.5

# Download URL templates
url = f'https://drive.google.com/uc?id={file_id}'
url1 = f'https://drive.google.com/uc?id={file_id1}'
url2 = f'https://drive.google.com/uc?id={file_id2}'
url3 = f'https://drive.google.com/uc?id={file_id3}'
url4 = f'https://drive.google.com/uc?id={file_id4}'

# Download the files
output = 'file.csv'
output1 = 'file1.csv'
output2 = 'file2.csv'
output3 = 'file3.csv'
output4 = 'file4.csv'

gdown.download(url, output, quiet=False)
gdown.download(url1, output1, quiet=False)
gdown.download(url2, output2, quiet=False)
gdown.download(url3, output3, quiet=False)
gdown.download(url4, output4, quiet=False)

# Read the downloaded CSV files
df = pd.read_csv(output)
df1 = pd.read_csv(output1)
df2 = pd.read_csv(output2)
df3 = pd.read_csv(output3)
df4 = pd.read_csv(output4)

# Ensure the columns are correctly named
assert 'SE Correct' in df.columns and 'AAVE Correct' in df.columns and'Original' in df.columns and 'Translated' in df.columns, "CSV must contain 'Original' and 'Translated' columns."
assert 'SE Correct' in df1.columns and 'AAVE Correct' in df1.columns and 'Original' in df1.columns and 'Translated' in df1.columns, "CSV must contain 'Original' and 'Translated' columns."
assert 'SE Correct' in df2.columns and 'AAVE Correct' in df2.columns and 'Original' in df2.columns and 'Translated' in df2.columns, "CSV must contain 'Original' and 'Translated' columns."
assert 'SE Correct New' in df3.columns and 'AAVE Correct New' in df3.columns and 'Original' in df3.columns and 'Translated' in df3.columns, "CSV must contain 'Original' and 'Translated' columns."
assert 'SE Correct New' in df4.columns and 'AAVE Correct New' in df4.columns and 'Original' in df4.columns and 'Translated' in df4.columns, "CSV must contain 'Original' and 'Translated' columns."

# List to store results
results = []

# Calculate BLEU score for each sentence pair and store results
for index, (row, row1, row2, row3, row4) in enumerate(zip(df.iterrows(), df1.iterrows(), df2.iterrows(), df3.iterrows(), df4.iterrows())):
    reference = row[1]['Original'].lower()
    hypothesis = row[1]['Translated'].lower()
    bleu_score = sacrebleu.sentence_bleu(hypothesis, [reference])

    geminiSE = row1[1]['Original'].lower()
    geminiAAVE = row1[1]['Translated'].lower()

    geminiflashSE = row2[1]['Original'].lower()
    geminiflashAAVE = row2[1]['Translated'].lower()

    gpt4oSE = row3[1]['Original'].lower()
    gpt4oAAVE = row3[1]['Translated'].lower()

    gpt3dot5SE = row4[1]['Original'].lower()
    gpt3dot5AAVE = row4[1]['Translated'].lower()

    geminiSE_result = row1[1]['SE Correct']
    geminiAAVE_result = row1[1]['AAVE Correct']

    geminiflashSE_result = row2[1]['SE Correct']
    geminiflashAAVE_result = row2[1]['AAVE Correct']

    gpt4oSE_result = row3[1]['SE Correct New']
    gpt4oAAVE_result = row3[1]['AAVE Correct New']

    gpt4SE_result = row[1]['SE Correct']
    gpt4AAVE_result = row[1]['AAVE Correct']

    gpt3dot5SE_result = row4[1]['SE Correct New']
    gpt3dot5AAVE_result = row4[1]['AAVE Correct New']

    if reference == geminiSE and hypothesis == geminiAAVE:
        # Append results to the list
        results.append({
            'Reference': reference,
            'Hypothesis': hypothesis,
            'BLEU_score': bleu_score.score,
            'Gemini Pro SE Correct': geminiSE_result,
            'Gemini Pro AAVE Correct': geminiAAVE_result,
            'Gemini Flash SE Correct': geminiflashSE_result,
            'Gemini Flash AAVE Correct': geminiflashAAVE_result,
            'GPT4o SE Correct': gpt4oSE_result,
            'GPT4o AAVE Correct': gpt4oAAVE_result,
            'GPT4 SE Correct': gpt4SE_result,
            'GPT4 AAVE Correct': gpt4AAVE_result,
            'GPT3.5 SE Correct': gpt3dot5SE_result,
            'GPT3.5 AAVE Correct': gpt3dot5AAVE_result
        })

# Create a DataFrame from the results list
results_df = pd.DataFrame(results)

# Save results to a new CSV file
output_results_csv = '/content/drive/MyDrive/Algoverse/BLEU Scores/evaluation_with_bleu_scores_sst2.csv'
results_df.to_csv(output_results_csv, index=False)

print(f"Results with BLEU scores saved to '{output_results_csv}'.")

