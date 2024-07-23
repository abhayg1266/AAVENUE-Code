#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install pandas')
import pandas as pd


# In[2]:


from google.colab import drive
drive.mount('/content/drive')


# In[12]:


import pandas as pd

def calculate_accuracy(csv_file_path):
    # Read the CSV file
    df = pd.read_csv(csv_file_path)

    # Calculate the total number of rows
    total_rows = len(df)

    # Calculate the number of correct and incorrect predictions for SE and AAVE
    se_correct_count = df['SE Correct'].sum()
    aave_correct_count = df['AAVE Correct'].sum()

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
csv_file_path = '/content/drive/MyDrive/Algoverse/AAVE Translations/GPT 4.0 Translations, Claude-Sonnet-3.5 Evaluation/SST-2/Filtered_Sentiment_Data.csv'
accuracy_results = calculate_accuracy(csv_file_path)


# In[11]:


import pandas as pd
import re

def extract_final_answer(text):
    if isinstance(text, str):
        match = re.search(r'\b0\b|\b1\b', text)
        if match:
            return match.group(0)
    return text

def calculate_wrong_questions_percentage_and_iou_sst2(file_path):
    df = pd.read_csv(file_path)

    # Ensure 'SE Correct' and 'AAVE Correct' are boolean
    df['SE Correct'] = df['SE Correct'].astype(bool)
    df['AAVE Correct'] = df['AAVE Correct'].astype(bool)

    # Calculate the number of rows
    total_rows = len(df)

    # Calculate the number of wrong answers in both SE and AAVE
    df['wrong_in_se'] = ~df['SE Correct']
    df['wrong_in_aave'] = ~df['AAVE Correct']

    both_wrong_df = df[df['wrong_in_se'] & df['wrong_in_aave']]
    both_wrong_count = len(both_wrong_df)

    # Calculate percentage of both wrong
    percentage_both_wrong = (both_wrong_count / total_rows) * 100

    # Calculate IoU for wrong answers
    wrong_questions_se = set(both_wrong_df['Original'].dropna().unique())
    wrong_questions_aave = set(both_wrong_df['Translated'].dropna().unique())

    intersection_wrong_questions = wrong_questions_se.intersection(wrong_questions_aave)
    union_wrong_questions = wrong_questions_se.union(wrong_questions_aave)
    iou_wrong_questions = len(intersection_wrong_questions) / len(union_wrong_questions) if union_wrong_questions else 0

    # Calculate IoU for passages
    wrong_passages_se = set(both_wrong_df['Original Sentiment'].dropna().unique())
    wrong_passages_aave = set(both_wrong_df['SE Sentiment'].dropna().unique())

    intersection_wrong_passages = wrong_passages_se.intersection(wrong_passages_aave)
    union_wrong_passages = wrong_passages_se.union(wrong_passages_aave)
    iou_wrong_passages = len(intersection_wrong_passages) / len(union_wrong_passages) if union_wrong_passages else 0

    print(f"Total rows: {total_rows}")
    print(f"Rows with both wrong answers: {both_wrong_count}")
    print(f"Percentage of both wrong: {percentage_both_wrong:.2f}%")
    print(f"Intersection over Union (IoU) for Questions: {iou_wrong_questions:.2f}, {iou_wrong_questions:.0%}")
    print(f"Intersection over Union (IoU) for Passages: {iou_wrong_passages:.2f}, {iou_wrong_passages:.0%}")

# Example usage with a given file path
file_path_example = '/content/drive/MyDrive/Algoverse/AAVE Translations/GPT 4.0 Translations, Claude-Sonnet-3.5 Evaluation/SST-2/Filtered_Sentiment_Data.csv'
calculate_wrong_questions_percentage_and_iou_sst2(file_path_example)


# In[ ]:


import pandas as pd
import re

def extract_final_answer(text):
    if isinstance(text, str):
        match = re.search(r'\b0\b|\b1\b', text)
        if match:
            return match.group(0)
    return text

def calculate_iou_copa(file_path):
    df = pd.read_csv(file_path)

    # Extract final answers from SE and AAVE response columns
    df['SE Response'] = df['SE Response'].apply(extract_final_answer)
    df['AAVE Response'] = df['AAVE Response'].apply(extract_final_answer)

    # Determine correctness based on the actual label
    df['SE Correct'] = df['SE Response'] == df['Actual Label']
    df['AAVE Correct'] = df['AAVE Response'] == df['Actual Label']

    # Calculate overall IoU for premises and choices
    premises_se = set(df['Premise'].dropna().unique())
    premises_aave = set(df['Translated Premise'].dropna().unique())

    choices1_se = set(df['Choice1'].dropna().unique())
    choices1_aave = set(df['Translated Choice1'].dropna().unique())

    choices2_se = set(df['Choice2'].dropna().unique())
    choices2_aave = set(df['Translated Choice2'].dropna().unique())

    intersection_premises = premises_se.intersection(premises_aave)
    union_premises = premises_se.union(premises_aave)
    iou_premises = len(intersection_premises) / len(union_premises) if union_premises else 0

    intersection_choices1 = choices1_se.intersection(choices1_aave)
    union_choices1 = choices1_se.union(choices1_aave)
    iou_choices1 = len(intersection_choices1) / len(union_choices1) if union_choices1 else 0

    intersection_choices2 = choices2_se.intersection(choices2_aave)
    union_choices2 = choices2_se.union(choices2_aave)
    iou_choices2 = len(intersection_choices2) / len(union_choices2) if union_choices2 else 0

    # Calculate wrong answers IoU
    df['wrong_in_se'] = ~df['SE Correct']
    df['wrong_in_aave'] = ~df['AAVE Correct']

    wrong_df = df[df['wrong_in_se'] & df['wrong_in_aave']]

    wrong_premises_se = set(wrong_df['Premise'].dropna().unique())
    wrong_premises_aave = set(wrong_df['Translated Premise'].dropna().unique())

    wrong_choices1_se = set(wrong_df['Choice1'].dropna().unique())
    wrong_choices1_aave = set(wrong_df['Translated Choice1'].dropna().unique())

    wrong_choices2_se = set(wrong_df['Choice2'].dropna().unique())
    wrong_choices2_aave = set(wrong_df['Translated Choice2'].dropna().unique())

    intersection_wrong_premises = wrong_premises_se.intersection(wrong_premises_aave)
    union_wrong_premises = wrong_premises_se.union(wrong_premises_aave)

    intersection_wrong_choices1 = wrong_choices1_se.intersection(wrong_choices1_aave)
    union_wrong_choices1 = wrong_choices1_se.union(wrong_choices1_aave)

    intersection_wrong_choices2 = wrong_choices2_se.intersection(wrong_choices2_aave)
    union_wrong_choices2 = wrong_choices2_se.union(wrong_choices2_aave)

    combined_intersection = len(intersection_wrong_premises) + len(intersection_wrong_choices1) + len(intersection_wrong_choices2)
    combined_union = len(union_wrong_premises) + len(union_wrong_choices1) + len(union_wrong_choices2)
    combined_iou = combined_intersection / combined_union if combined_union else 0

    print(f"Intersection over Union (IoU) for Premises: {iou_premises:.2f}, {iou_premises:.0%}")
    print(f"Intersection over Union (IoU) for Choice1: {iou_choices1:.2f}, {iou_choices1:.0%}")
    print(f"Intersection over Union (IoU) for Choice2: {iou_choices2:.2f}, {iou_choices2:.0%}")
    print(f"Intersection over Union (IoU) for wrong answers in SAE and AAVE: {combined_iou:.2f}, {combined_iou:.0%}")

# CHANGE THE FILE PATH EACH TIME IF YOU WANT TO TEST A DIFFERENT ONE OUT!!!
file_path = '/content/drive/MyDrive/Algoverse/AAVE Translations/GPT 4.0 Translations, Gemini-1.5-Flash Evaluation/COPA/COPA_superglue_500_gemini.csv'
calculate_iou_copa(file_path)


# In[ ]:


import pandas as pd
import re

def extract_final_answer(text):
    if isinstance(text, str):
        match = re.search(r'\b0\b|\b1\b', text)
        if match:
            return match.group(0)
    return text

def calculate_iou_copa(file_path):
    df = pd.read_csv(file_path)

    # Extract final answers from SE and AAVE response columns
    df['SE Response'] = df['SE Response'].apply(extract_final_answer)
    df['AAVE Response'] = df['AAVE Response'].apply(extract_final_answer)

    # Determine correctness based on the actual label
    df['SE Correct'] = df['SE Response'] == df['Actual Label']
    df['AAVE Correct'] = df['AAVE Response'] == df['Actual Label']

    # Calculate overall IoU for premises and choices
    premises_se = set(df['Premise'].dropna().unique())
    premises_aave = set(df['Translated Premise'].dropna().unique())

    choices1_se = set(df['Choice1'].dropna().unique())
    choices1_aave = set(df['Translated Choice1'].dropna().unique())

    choices2_se = set(df['Choice2'].dropna().unique())
    choices2_aave = set(df['Translated Choice2'].dropna().unique())

    intersection_premises = premises_se.intersection(premises_aave)
    union_premises = premises_se.union(premises_aave)
    iou_premises = len(intersection_premises) / len(union_premises) if union_premises else 0

    intersection_choices1 = choices1_se.intersection(choices1_aave)
    union_choices1 = choices1_se.union(choices1_aave)
    iou_choices1 = len(intersection_choices1) / len(union_choices1) if union_choices1 else 0

    intersection_choices2 = choices2_se.intersection(choices2_aave)
    union_choices2 = choices2_se.union(choices2_aave)
    iou_choices2 = len(intersection_choices2) / len(union_choices2) if union_choices2 else 0

    # Calculate wrong answers IoU
    df['wrong_in_se'] = ~df['SE Correct']
    df['wrong_in_aave'] = ~df['AAVE Correct']

    wrong_df = df[df['wrong_in_se'] & df['wrong_in_aave']]

    wrong_premises_se = set(wrong_df['Premise'].dropna().unique())
    wrong_premises_aave = set(wrong_df['Translated Premise'].dropna().unique())

    wrong_choices1_se = set(wrong_df['Choice1'].dropna().unique())
    wrong_choices1_aave = set(wrong_df['Translated Choice1'].dropna().unique())

    wrong_choices2_se = set(wrong_df['Choice2'].dropna().unique())
    wrong_choices2_aave = set(wrong_df['Translated Choice2'].dropna().unique())

    intersection_wrong_premises = wrong_premises_se.intersection(wrong_premises_aave)
    union_wrong_premises = wrong_premises_se.union(wrong_premises_aave)

    intersection_wrong_choices1 = wrong_choices1_se.intersection(wrong_choices1_aave)
    union_wrong_choices1 = wrong_choices1_se.union(wrong_choices1_aave)

    intersection_wrong_choices2 = wrong_choices2_se.intersection(wrong_choices2_aave)
    union_wrong_choices2 = wrong_choices2_se.union(wrong_choices2_aave)

    combined_intersection = len(intersection_wrong_premises) + len(intersection_wrong_choices1) + len(intersection_wrong_choices2)
    combined_union = len(union_wrong_premises) + len(union_wrong_choices1) + len(union_wrong_choices2)
    combined_iou = combined_intersection / combined_union if combined_union else 0

    print(f"Intersection over Union (IoU) for Premises: {iou_premises:.2f}, {iou_premises:.0%}")
    print(f"Intersection over Union (IoU) for Choice1: {iou_choices1:.2f}, {iou_choices1:.0%}")
    print(f"Intersection over Union (IoU) for Choice2: {iou_choices2:.2f}, {iou_choices2:.0%}")
    print(f"Intersection over Union (IoU) for wrong answers in SAE and AAVE: {combined_iou:.2f}, {combined_iou:.0%}")

# CHANGE THE FILE PATH EACH TIME IF YOU WANT TO TEST A DIFFERENT ONE OUT!!!
file_path = '/content/drive/MyDrive/Algoverse/AAVE Translations/GPT 4.0 Translations, Gemini-1.5-Flash Evaluation/COPA/COPA_superglue_500_gemini.csv'
calculate_iou_copa(file_path)


# In[4]:


import pandas as pd

def normalize_text(text):
    if isinstance(text, str):
        return text.strip().lower()
    return text

def calculate_iou_boolq(file_path, question_column_se, question_column_aave, passage_column_se, passage_column_aave, actual_answer_column, se_answer_column, aave_answer_column, se_correct_column, aave_correct_column):
    df = pd.read_csv(file_path)

    df[question_column_se] = df[question_column_se].apply(normalize_text)
    df[question_column_aave] = df[question_column_aave].apply(normalize_text)
    df[passage_column_se] = df[passage_column_se].apply(normalize_text)
    df[passage_column_aave] = df[passage_column_aave].apply(normalize_text)

    questions_se = set(df[question_column_se].dropna().unique())
    questions_aave = set(df[question_column_aave].dropna().unique())

    passages_se = set(df[passage_column_se].dropna().unique())
    passages_aave = set(df[passage_column_aave].dropna().unique())

    intersection_questions = questions_se.intersection(questions_aave)
    union_questions = questions_se.union(questions_aave)

    iou_questions = len(intersection_questions) / len(union_questions)

    intersection_passages = passages_se.intersection(passages_aave)
    union_passages = passages_se.union(passages_aave)

    iou_passages = len(intersection_passages) / len(union_passages)

    print(f"Intersection over Union (IoU) for Questions: {iou_questions}, {iou_questions:.0%}")
    print(f"Intersection over Union (IoU) for Passages: {iou_passages}, {iou_passages:.0%}")

    # Calculate wrong answers
    df['wrong_in_se'] = df[se_correct_column] == 0
    df['wrong_in_aave'] = df[aave_correct_column] == 0

    # Filter the DataFrame to only include rows where both SE and AAVE got the wrong answer
    wrong_df = df[df['wrong_in_se'] & df['wrong_in_aave']]

    # Calculate the proportion of wrong answers
    total_rows = len(df)
    wrong_both_count = len(wrong_df)

    wrong_both_proportion = wrong_both_count / total_rows

    print(f"Proportion of questions where both SE and AAVE answers are wrong: {wrong_both_proportion}, {wrong_both_proportion:.0%}")

# CHANGE THE FILE PATH EACH TIME IF YOU WANT TO TEST A DIFFERENT ONE OUT!!!
file_path = '/content/drive/MyDrive/Algoverse/AAVE Translations/GPT 4.0 Translations, Claude-Sonnet-3.5 Evaluation/BoolQ/Filtered_BoolQA_Data_Final.csv'
question_column_se = 'se question'
question_column_aave = 'aave question'
passage_column_se = 'se passage'
passage_column_aave = 'aave passage'
actual_answer_column = 'actual answer'
se_answer_column = 'se answer'
aave_answer_column = 'aave answer'
se_correct_column = 'se correct'
aave_correct_column = 'aave correct'

calculate_iou_boolq(file_path, question_column_se, question_column_aave, passage_column_se, passage_column_aave, actual_answer_column, se_answer_column, aave_answer_column, se_correct_column, aave_correct_column)


# In[13]:


import pandas as pd

def normalize_text(text):
    if isinstance(text, str):
        return text.strip().lower()
    return text

def calculate_iou_multirc(file_path, paragraph_column_se, paragraph_column_aave, question_column_se, question_column_aave, answer_column_se, answer_column_aave, correct_column, se_response_column, aave_response_column):
    df = pd.read_csv(file_path)

    df[paragraph_column_se] = df[paragraph_column_se].apply(normalize_text)
    df[paragraph_column_aave] = df[paragraph_column_aave].apply(normalize_text)
    df[question_column_se] = df[question_column_se].apply(normalize_text)
    df[question_column_aave] = df[question_column_aave].apply(normalize_text)
    df[answer_column_se] = df[answer_column_se].apply(normalize_text)
    df[answer_column_aave] = df[answer_column_aave].apply(normalize_text)

    paragraphs_se = set(df[paragraph_column_se].dropna().unique())
    paragraphs_aave = set(df[paragraph_column_aave].dropna().unique())

    questions_se = set(df[question_column_se].dropna().unique())
    questions_aave = set(df[question_column_aave].dropna().unique())

    answers_se = set(df[answer_column_se].dropna().unique())
    answers_aave = set(df[answer_column_aave].dropna().unique())

    intersection_paragraphs = paragraphs_se.intersection(paragraphs_aave)
    union_paragraphs = paragraphs_se.union(paragraphs_aave)
    iou_paragraphs = len(intersection_paragraphs) / len(union_paragraphs) if union_paragraphs else 0

    intersection_questions = questions_se.intersection(questions_aave)
    union_questions = questions_se.union(questions_aave)
    iou_questions = len(intersection_questions) / len(union_questions) if union_questions else 0

    intersection_answers = answers_se.intersection(answers_aave)
    union_answers = answers_se.union(answers_aave)
    iou_answers = len(intersection_answers) / len(union_answers) if union_answers else 0

    print(f"Intersection over Union (IoU) for Paragraphs: {iou_paragraphs:.2f}, {iou_paragraphs:.0%}")
    print(f"Intersection over Union (IoU) for Questions: {iou_questions:.2f}, {iou_questions:.0%}")
    print(f"Intersection over Union (IoU) for Answers: {iou_answers:.2f}, {iou_answers:.0%}")

    # Determine correctness based on the actual label
    df['SE Correct'] = df.apply(lambda row: str(row[se_response_column]).strip() == str(row[correct_column]).strip(), axis=1)
    df['AAVE Correct'] = df.apply(lambda row: str(row[aave_response_column]).strip() == str(row[correct_column]).strip(), axis=1)

    # Calculate wrong answers
    df['wrong_in_se'] = df['SE Correct'] == False
    df['wrong_in_aave'] = df['AAVE Correct'] == False

    # Filter the DataFrame to only include rows where both SE and AAVE got the wrong answer
    wrong_df = df[df['wrong_in_se'] & df['wrong_in_aave']]

    # Calculate the proportion of wrong answers
    total_rows = len(df)
    wrong_both_count = len(wrong_df)

    wrong_both_proportion = wrong_both_count / total_rows if total_rows else 0

    print(f"Proportion of questions where both SE and AAVE answers are wrong: {wrong_both_proportion:.2f}, {wrong_both_proportion:.0%}")

    # Print out a sample of the comparisons
    sample_comparisons = df[['Actual Label', se_response_column, 'SE Correct', aave_response_column, 'AAVE Correct']].sample(10)
    print("Sample comparisons:\n", sample_comparisons)

# Example usage for MultiRC
file_path = '/content/drive/MyDrive/Algoverse/AAVE Translations/GPT 4.0 Translations, Gemini-1.5-Pro Evaluation/Multi-RC/MultiRC_1000_gemini.csv'
paragraph_column_se = 'Paragraph'
paragraph_column_aave = 'Translated Paragraph'
question_column_se = 'Question'
question_column_aave = 'Translated Question'
answer_column_se = 'Answer'
answer_column_aave = 'Translated Answer'
correct_column = 'Actual Label'
se_response_column = 'SE Response'
aave_response_column = 'AAVE Response'

calculate_iou_multirc(file_path, paragraph_column_se, paragraph_column_aave, question_column_se, question_column_aave, answer_column_se, answer_column_aave, correct_column, se_response_column, aave_response_column)

