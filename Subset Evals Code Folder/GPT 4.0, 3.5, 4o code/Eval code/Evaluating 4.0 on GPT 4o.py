#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install datasets')
get_ipython().system('pip install openai==0.28')
get_ipython().system('pip install pandas')
get_ipython().system('pip install google-colab')


# In[ ]:


import openai
import pandas as pd
from datasets import load_dataset
from google.colab import drive

# Set up your OpenAI API key
openai.api_key = "YOUR_API_KEY_HERE"

# Set up OpenAI organization key
openai.organization = "YOUR_ORG_KEY_HERE"


# In[ ]:


from google.colab import drive
drive.mount('/content/drive')


# In[ ]:


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

# Function to determine sentiment
def determine_sentiment(passage):
    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "Assess the sentiment of the passage and respond with 'Positive' or 'Negative'. Ensure the response contains only these terms without any additional characters or explanations."},
            {"role": "user", "content": f"Analyze the sentiment of the passage:\n\"{passage}\""}
        ],
        max_tokens=50
    )
    return response.choices[0].message['content'].strip() if response.choices else "No translation found."

# Function to check correctness of sentiment
def check_correctness(predicted, actual):
    return 1 if predicted.lower() == actual.lower() else 0

# Function to determine the answer for MultiRC
def determine_answer(paragraph, question, answer):
    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You can only say 2 things, '0' & '1'. Determine if the answer to the question based on the given paragraph is true or false. Only respond with '0' for false or '1' for true."},
            {"role": "user", "content": f"Paragraph: \"{paragraph}\"\nQuestion: \"{question}\"\nAnswer: \"{answer}\""}
        ],
        max_tokens=25
    )
    return response.choices[0].message['content'].strip() if response.choices else "No response found."

# Function to predict cause-effect for COPA
def predict_cause_effect(premise, choices):
    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You must determine if the presented choice is the cause or the effect of the premise given. Only respond with '0' if the first choice is correct or '1' if the second choice is correct."},
            {"role": "user", "content": f"Premise: \"{premise}\"\nChoice1: \"{choices[0]}\"\nChoice2: \"{choices[1]}\""}
        ],
        max_tokens=25
    )
    return response.choices[0].message['content'].strip() if response.choices else "No response found."

# Function to query GPT for BoolQ
def query_gpt(passage, question):
    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are reading a passage, interpreting the information, and answering the question as 'True' or 'False' only. No other characters should be added in your response."},
            {"role": "user", "content": f"Passage: \"{passage}\"\nQuestion: \"{question}\""}
        ],
        max_tokens=50
    )
    return response.choices[0].message['content'].strip() if response.choices else "No translation found."

# Function to convert label to True/False
def convert(answer):
    return ["True", "True.", "true", "true."] if answer == 1 else ["False", "False.", "false", "false."]

# Function to check correctness for BoolQ
def check_correctness_boolq(predicted, actual):
    return 1 if predicted in actual else 0

# Evaluation for BoolQ
def evaluate_boolq():
    csv_file_path = '/content/drive/MyDrive/Algoverse/AAVE Translations/GPT 4.0 Translations/BoolQ/BoolQA_superglue_1000_4-0.csv'
    translations_df = pd.read_csv(csv_file_path)
    results = []

    for index, row in translations_df.iterrows():
        question = row['se question']
        passage = row['se passage']
        answer = row['actual answer']
        actual_answer = convert(answer)

        aave_question = row['aave qustion']
        aave_passage = row['aave passage']

        se_answer_new = query_gpt(passage, question)
        aave_answer = query_gpt(aave_passage, aave_question)

        se_correct = check_correctness_boolq(se_answer_new, actual_answer)
        aave_correct = check_correctness_boolq(aave_answer, actual_answer)

        results.append({
            'se question': question,
            'aave question': aave_question,
            'actual answer': actual_answer[0],
            'se passage': passage,
            'se answer': se_answer_new,
            'aave passage': aave_passage,
            'aave answer': aave_answer,
            'se correct': se_correct,
            'aave correct': aave_correct
        })

        print(f"Processed row {index + 1}")

    results_df = pd.DataFrame(results)
    output_file_path = '/content/drive/MyDrive/Algoverse/AAVE Translations/GPT 4.0 Translations, 4o Evaluation/BoolQ/BoolQA_superglue_1000_4o.csv'
    results_df.to_csv(output_file_path, index=False)
    print("BoolQ evaluation results saved to CSV file.")

# Evaluation for SST-2
def evaluate_sst2():
    csv_file_path = '/content/drive/MyDrive/Algoverse/AAVE Translations/GPT 4.0 Translations/SST-2/4.0/SST-2_super-glue_1000_4.0.csv'
    translations_df = pd.read_csv(csv_file_path)
    results = []

    for index, row in translations_df.iterrows():
        sentence = row['Original']
        translated_sentence = row['Translated']
        actual_sentiment = row['Original Sentiment']

        se_sentiment = determine_sentiment(sentence)
        aave_sentiment = determine_sentiment(translated_sentence)

        se_correct = check_correctness(se_sentiment, actual_sentiment)
        aave_correct = check_correctness(aave_sentiment, actual_sentiment)

        results.append({
            'Original': sentence,
            'Translated': translated_sentence,
            'Original Sentiment': actual_sentiment,
            'SE Sentiment': se_sentiment,
            'AAVE Sentiment': aave_sentiment,
            'SE Correct': se_correct,
            'AAVE Correct': aave_correct
        })

        print(f"Processed row {index + 1}")

    results_df = pd.DataFrame(results)
    output_file_path = '/content/drive/MyDrive/Algoverse/AAVE Translations/GPT 4.0 Translations, 4o Evaluation/SST-2/sst-2_super-glue_1000_4o.csv'
    results_df.to_csv(output_file_path, index=False)
    print("SST-2 evaluation results saved to CSV file.")

# Evaluation for MultiRC
def evaluate_multirc():
    csv_file_path = '/content/drive/MyDrive/Algoverse/AAVE Translations/GPT 4.0 Translations/Multi-RC/4.0/MultiRC_1000_4.0.csv'
    translations_df = pd.read_csv(csv_file_path)
    results = []

    for index, row in translations_df.iterrows():
        paragraph = row['Paragraph']
        translated_paragraph = row['Translated Paragraph']
        question = row['Question']
        translated_question = row['Translated Question']
        answer = row['Answer']
        translated_answer = row['Translated Answer']
        actual_label = row['Actual Label']

        se_response = determine_answer(paragraph, question, answer)
        aave_response = determine_answer(translated_paragraph, translated_question, translated_answer)

        results.append({
            'Paragraph': paragraph,
            'Translated Paragraph': translated_paragraph,
            'Question': question,
            'Translated Question':```python
        translated_question,
            'Answer': answer,
            'Translated Answer': translated_answer,
            'Actual Label': str(actual_label),
            'SE Response': se_response,
            'AAVE Response': aave_response,
            'SE Correct': check_correctness(se_response, str(actual_label)),
            'AAVE Correct': check_correctness(aave_response, str(actual_label))
        })

        print(f"Processed row {index + 1}")

    results_df = pd.DataFrame(results)
    output_file_path = '/content/drive/MyDrive/Algoverse/AAVE Translations/GPT 4.0 Translation, 4o evaluation/MultiRC/MultiRC_1000_4o.csv'
    results_df.to_csv(output_file_path, index=False)
    print("MultiRC evaluation results saved to CSV file.")

# Evaluation for COPA
def evaluate_copa():
    csv_file_path = '/content/drive/MyDrive/Algoverse/AAVE Translations/GPT 4.0 Translations/Copa_superglue_500_4.0.csv'
    translations_df = pd.read_csv(csv_file_path)
    results = []

    for index, row in translations_df.iterrows():
        premise = row['Premise']
        translated_premise = row['Translated Premise']
        choice1 = row['Choice1']
        translated_choice1 = row['Translated Choice1']
        choice2 = row['Choice2']
        translated_choice2 = row['Translated Choice2']
        actual_label = row['Actual Label']

        se_response = predict_cause_effect(premise, [choice1, choice2])
        aave_response = predict_cause_effect(translated_premise, [translated_choice1, translated_choice2])

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

        print(f"Processed row {index + 1}")

    results_df = pd.DataFrame(results)
    output_file_path = '/content/drive/MyDrive/Algoverse/AAVE Translations/GPT 4.0 Translation, 4o evaluation/Copa_superglue_500_4o.csv'
    results_df.to_csv(output_file_path, index=False)
    print("COPA evaluation results saved to CSV file.")

# Evaluation for WSC
def evaluate_wsc():
    csv_file_path = '/content/drive/MyDrive/Algoverse/AAVE Translations/GPT 4.0 Translations/WSC/WSC_630_4.0.csv'
    translations_df = pd.read_csv(csv_file_path)
    results = []

    for index, row in translations_df.iterrows():
        passage = row['Original Passage']
        translated_passage = row['Translated Passage']
        pronoun = row['Pronoun']
        span1 = row['Span1']
        translated_span1 = row['Translated Span1']
        span2 = row['Span2']
        translated_span2 = row['Translated Span2']
        actual_reference = str(row['Actual Reference'])

        se_reference = determine_pronoun_reference(passage, pronoun, span1, span2)
        aave_reference = determine_pronoun_reference(translated_passage, pronoun, translated_span1, translated_span2)

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

        print(f"Processed row {index + 1}")

    results_df = pd.DataFrame(results)
    output_file_path = '/content/drive/MyDrive/Algoverse/AAVE Translations/GPT 4.0 Translation, 4o evaluation/WSC/WSC_630_4o.csv'
    results_df.to_csv(output_file_path, index=False)
    print("WSC evaluation results saved to CSV file.")

# Run evaluations for all datasets
evaluate_boolq()
evaluate_sst2()
evaluate_multirc()
evaluate_copa()
evaluate_wsc()

print("All evaluations completed")

