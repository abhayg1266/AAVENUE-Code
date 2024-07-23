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

# Function to format text as Markdown
def to_markdown(text):
    text = text.replace('•', '  *')
    return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))

# Set up your Google API key
GOOGLE_API_KEY = "API KEY"
genai.configure(api_key=GOOGLE_API_KEY)


# In[ ]:


model = genai.GenerativeModel('gemini-1.5-pro')

drive.mount('/content/drive')


# In[ ]:


# Define evaluation functions
def predict_cause_effect(premise, choices):
    try:
        response = model.generate_content(f"Premise: \"{premise}\"\nChoice1: \"{choices[0]}\"\nChoice2: \"{choices[1]}\". Only respond with '0' if the first choice is correct or '1' if the second choice is correct.")
        return response.candidates[0].content.parts[0].text.strip()
    except Exception as e:
        print(f"Error predicting cause-effect: {e}")
        return "Prediction failed"

def query_gpt_boolq(passage, question):
    try:
        response = model.generate_content(f"Passage: \"{passage}\"\nQuestion: \"{question}\". Answer as 'True' or 'False' only.")
        return response.candidates[0].content.parts[0].text.strip()
    except Exception as e:
        print(f"Error querying GPT for BoolQ: {e}")
        return "Query failed"

def determine_sentiment(passage):
    try:
        response = model.generate_content(f"Analyze the sentiment of the passage: \"{passage}\". Respond with 'Positive' or 'Negative' only.")
        return response.candidates[0].content.parts[0].text.strip()
    except Exception as e:
        print(f"Error determining sentiment: {e}")
        return "Sentiment determination failed"

def determine_answer(paragraph, question, answer):
    try:
        response = model.generate_content(f"Paragraph: \"{paragraph}\"\nQuestion: \"{question}\"\nAnswer: \"{answer}\". Respond with '0' for false or '1' for true.")
        return response.candidates[0].content.parts[0].text.strip()
    except Exception as e:
        print(f"Error determining answer: {e}")
        return "Answer determination failed"

def determine_pronoun_reference(passage, pronoun, span1, span2):
    try:
        response = model.generate_content(f"Passage: \"{passage}\"\nPronoun: \"{pronoun}\"\nSpan1: \"{span1}\"\nSpan2: \"{span2}\". Respond with '0' if the pronoun refers to Span1 or '1' if it refers to Span2.")
        return response.candidates[0].content.parts[0].text.strip()
    except Exception as e:
        print(f"Error determining pronoun reference: {e}")
        return "Pronoun reference determination failed"

# Evaluation functions for each dataset
def evaluate_boolq():
    csv_file_path = '/content/drive/MyDrive/Algoverse/AAVE Translations/GPT 4.0 Translations, 4.0 Evaluations/BoolQ/BoolQA_superglue_1000_4-0.csv'
    translations_df = pd.read_csv(csv_file_path)
    results = []

    for index, row in translations_df.iterrows():
        question = row['se question']
        passage = row['se passage']
        answer = row['actual answer']
        actual_answer = ["True", "True.", "true", "true."] if answer == 1 else ["False", "False.", "false", "false."]

        aave_question = row['aave qustion']
        aave_passage = row['aave passage']

        se_answer_new = query_gpt_boolq(passage, question)
        aave_answer = query_gpt_boolq(aave_passage, aave_question)

        se_correct = 1 if se_answer_new in actual_answer else 0
        aave_correct = 1 if aave_answer in actual_answer else 0

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
    output_file_path = '/content/drive/MyDrive/Algoverse/AAVE Translations/GPT 4.0 Translations, Gemini-1.5-Pro Evaluation/BoolQ/BoolQA_superglue_1000_gemini.csv'
    results_df.to_csv(output_file_path, index=False)
    print("BoolQ evaluation results saved to CSV file.")

def evaluate_sst2():
    csv_file_path = '/content/drive/MyDrive/Algoverse/AAVE Translations/GPT 4.0 Translations, 4.0 Evaluations/SST-2/SST-2_super-glue_1000_4.0.csv'
    translations_df = pd.read_csv(csv_file_path)
    results = []

    for index, row in translations_df.iterrows():
        sentence = row['Original']
        translated_sentence = row['Translated']
        actual_sentiment = "Positive" if row['Original Sentiment'] == 1 else "Negative"

        se_sentiment = determine_sentiment(sentence)
        aave_sentiment = determine_sentiment(translated_sentence)

        se_correct = 1 if se_sentiment.lower() == actual_sentiment.lower() else 0
        aave_correct = 1 if aave_sentiment.lower() == actual_sentiment.lower() else 0

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
    output_file_path = '/content/drive/MyDrive/Algoverse/AAVE Translations/GPT 4.0 Translations, Gemini-1.5-Pro Evaluation/SST-2/sst-2_super-glue_1000_gemini.csv'
    results_df.to_csv(output_file_path, index=False)
    print("SST-2 evaluation results saved to CSV file.")

def evaluate_multirc():
    csv_file_path = '/content/drive/MyDrive/Algoverse/AAVE Translations/GPT 4.0 Translations, 4.0 Evaluations/Multi-RC/MultiRC_1000_4.0.csv'
    translations_df = pd.read_csv(csv_file_path)
    results = []

    for index, row in translations_df.iterrows():
        paragraph = row['Paragraph']
        translated_paragraph = row['Translated Paragraph']
        question = row['Question']
        translated_question = row['Translated Question']
        answer = row['Answer']
        translated_answer = row['Translated Answer']
        actual_label = str(row['Actual Label'])

        se_response = determine_answer(paragraph, question, answer)
        aave_response = determine_answer(translated_paragraph, translated_question, translated_answer)

        results.append({
            'Paragraph': paragraph,
            'Translated Paragraph': translated_paragraph,
            'Question': question,
            'Translated Question': translated_question,
            'Answer': answer,
            'Translated Answer': translated_answer,
            'Actual Label': actual_label,
            'SE Response': se_response,
            'AAVE Response': aave_response
        })

        print(f"Processed row {index + 1}")

    results_df = pd.DataFrame(results)
    output_file_path = '/content/drive/MyDrive/Algoverse/AAVE Translations/GPT 4.0 Translations, Gemini-1.5-Pro Evaluation/Multi-RC/MultiRC_1000_gemini.csv'
    results_df.to_csv(output_file_path, index=False)
    print("MultiRC evaluation results saved to CSV file.")

def evaluate_wsc():
    csv_file_path = '/content/drive/MyDrive/Algoverse/AAVE Translations/GPT 4.0 Translations, 4.0 Evaluations/WSC/WSC_630_4.0.csv'
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
    output_file_path = '/content/drive/MyDrive/Algoverse/AAVE Translations/GPT 4.0 Translations, Gemini-1.5-Pro Evaluation/WSC/WSC_630_gemini.csv'
    results_df.to_csv(output_file_path, index=False)
    print("WSC evaluation results saved to CSV file.")

# Run evaluations for selected datasets
evaluate_boolq()
evaluate_sst2()
evaluate_multirc()
evaluate_wsc()

print("Selected evaluations completed")

