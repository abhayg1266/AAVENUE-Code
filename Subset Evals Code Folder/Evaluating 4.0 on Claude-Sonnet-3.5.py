#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install datasets')
get_ipython().system('pip install pandas')
get_ipython().system('pip install google-colab')


# In[ ]:


from google.colab import drive
drive.mount('/content/drive')


# code below has functions multirc, copa, wsc:
# boolq and sst-2 are done

# In[ ]:


import pathlib
import textwrap
import pandas as pd
from IPython.display import display, Markdown
from google.colab import drive
import requests
import time
import os

# Claude API configuration
API_KEY = "API KEY"
HEADERS = {
    'x-api-key': API_KEY,
    'Content-Type': 'application/json',
    'anthropic-version': '2023-06-01'
}

MAX_REQUESTS_PER_MINUTE = 50

def call_claude(prompt):
    attempt = 0
    request_count = 0
    while True:
        if request_count >= MAX_REQUESTS_PER_MINUTE:
            print("Rate limit reached. Waiting for 45 seconds...")
            time.sleep(45)
            request_count = 0  # Reset the request count after waiting

        try:
            payload = {
                "model": "claude-3-5-sonnet-20240620",
                "system": "You are a helpful assistant.",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 50
            }
            response = requests.post('https://api.anthropic.com/v1/messages', headers=HEADERS, json=payload)
            request_count += 1
            response_data = response.json()
            if 'content' in response_data:
                message_content = response_data['content'][0]['text']
                return message_content.strip(), request_count
            else:
                print("Error: 'content' key not found in response_data")
                print("Response Data:", response_data)
                return "Request failed", request_count
        except requests.exceptions.RequestException as e:
            print(f"Request exception: {e}")
            return "Request failed", request_count
        except Exception as e:
            if 'rate_limit_error' in str(e):
                print("Rate limit error encountered. Waiting for 20 seconds before retrying...")
                time.sleep(20)
                request_count = 0  # Reset the request count after waiting
            else:
                print(f"Error: {e}")
                return "Request failed", request_count

# Function to format text as Markdown
def to_markdown(text):
    text = text.replace('•', '  *')
    return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))

# Function to check if file exists
def check_file_exists(file_path):
    if not os.path.isfile(file_path):
        print(f"Error: File not found at {file_path}")
        return False
    return True

# Function to ensure directory exists
def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


# In[ ]:


def evaluate_multirc():
    csv_file_path = '/content/MultiRC_1000_4.0.csv'
    if not check_file_exists(csv_file_path):
        return

    translations_df = pd.read_csv(csv_file_path)
    results = []
    request_count = 0

    for index, row in translations_df.iterrows():
        paragraph = row['Paragraph']
        translated_paragraph = row['Translated Paragraph']
        question = row['Question']
        translated_question = row['Translated Question']
        answer = row['Answer']
        translated_answer = row['Translated Answer']
        actual_label = str(row['Actual Label'])

        se_response, request_count = call_claude(f"Paragraph: \"{paragraph}\"\nQuestion: \"{question}\"\nAnswer: \"{answer}\". Respond with '0' for false or '1' for true.")
        aave_response, request_count = call_claude(f"Paragraph: \"{translated_paragraph}\"\nQuestion: \"{translated_question}\"\nAnswer: \"{translated_answer}\". Respond with '0' for false or '1' for true.")

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

    output_directory = '/content/drive/MyDrive/Algoverse Comparison/Sonnet Evals'
    ensure_directory_exists(output_directory)
    output_file_path = f'{output_directory}/MultiRC_1000_claude.csv'
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file_path, index=False)
    print("MultiRC evaluation results saved to CSV file.")

# Run MultiRC evaluation
evaluate_multirc()


# In[ ]:


def evaluate_wsc():
    csv_file_path = '/content/WSC_630_4.0.csv'
    if not check_file_exists(csv_file_path):
        return

    translations_df = pd.read_csv(csv_file_path)
    results = []
    request_count = 0

    for index, row in translations_df.iterrows():
        passage = row['Original Passage']
        translated_passage = row['Translated Passage']
        pronoun = row['Pronoun']
        span1 = row['Span1']
        translated_span1 = row['Translated Span1']
        span2 = row['Span2']
        translated_span2 = row['Translated Span2']
        actual_reference = str(row['Actual Reference'])

        se_reference, request_count = call_claude(f"Passage: \"{passage}\"\nPronoun: \"{pronoun}\"\nSpan1: \"{span1}\"\nSpan2: \"{span2}\". Respond with '0' if the pronoun refers to Span1 or '1' if it refers to Span2.", request_count)
        aave_reference, request_count = call_claude(f"Passage: \"{translated_passage}\"\nPronoun: \"{pronoun}\"\nSpan1: \"{translated_span1}\"\nSpan2: \"{translated_span2}\". Respond with '0' if the pronoun refers to Span1 or '1' if it refers to Span2.", request_count)

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

    output_directory = '/content/drive/MyDrive/Algoverse Comparison/Sonnet Evals'
    ensure_directory_exists(output_directory)
    output_file_path = f'{output_directory}/WSC_630_claude.csv'
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file_path, index=False)
    print("WSC evaluation results saved to CSV file.")

# Run WSC evaluation
evaluate_wsc()


# In[ ]:


def evaluate_copa():
    csv_file_path = '/content/Copa_superglue_500_4.0.csv'
    if not check_file_exists(csv_file_path):
        return

    translations_df = pd.read_csv(csv_file_path)
    results = []
    request_count = 0

    for index, row in translations_df.iterrows():
        premise = row['Premise']
        translated_premise = row['Translated Premise']
        choice1 = row['Choice1']
        translated_choice1 = row['Translated Choice1']
        choice2 = row['Choice2']
        translated_choice2 = row['Translated Choice2']
        actual_label = row['Actual Label']

        se_response, request_count = call_claude(f"Premise: \"{premise}\"\nChoice1: \"{choice1}\"\nChoice2: \"{choice2}\". Respond with '0' if the first choice is correct or '1' if the second choice is correct.")
        aave_response, request_count = call_claude(f"Premise: \"{translated_premise}\"\nChoice1: \"{translated_choice1}\"\nChoice2: \"{translated_choice2}\". Respond with '0' if the first choice is correct or '1' if the second choice is correct.")

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

    output_directory = '/content/drive/MyDrive/Algoverse Comparison/Sonnet Evals'
    ensure_directory_exists(output_directory)
    output_file_path = f'{output_directory}/Copa_superglue_500_claude.csv'
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file_path, index=False)
    print("COPA evaluation results saved to CSV file.")

# Run COPA evaluation
evaluate_copa()


# ****
# 
# > Add blockquote
# 
# 

# In[ ]:


import pathlib
import textwrap
import pandas as pd
from IPython.display import display, Markdown
from google.colab import drive
import requests
import time
import os

# Claude API configuration
API_KEY = 'API KEY'
HEADERS = {
    'x-api-key': API_KEY,
    'Content-Type': 'application/json',
    'anthropic-version': '2023-06-01'
}

MAX_REQUESTS_PER_MINUTE = 50

def call_claude(prompt):
    attempt = 0
    request_count = 0
    while True:
        if request_count >= MAX_REQUESTS_PER_MINUTE:
            print("Rate limit reached. Waiting for 45 seconds...")
            time.sleep(45)
            request_count = 0  # Reset the request count after waiting

        try:
            payload = {
                "model": "claude-3-5-sonnet-20240620",
                "system": "You are a helpful assistant.",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 50
            }
            response = requests.post('https://api.anthropic.com/v1/messages', headers=HEADERS, json=payload)
            request_count += 1
            response_data = response.json()
            if 'content' in response_data:
                message_content = response_data['content'][0]['text']
                return message_content.strip(), request_count
            else:
                print("Error: 'content' key not found in response_data")
                print("Response Data:", response_data)
                return "Request failed", request_count
        except requests.exceptions.RequestException as e:
            print(f"Request exception: {e}")
            return "Request failed", request_count
        except Exception as e:
            if 'rate_limit_error' in str(e):
                print("Rate limit error encountered. Waiting for 20 seconds before retrying...")
                time.sleep(20)
                request_count = 0  # Reset the request count after waiting
            else:
                print(f"Error: {e}")
                return "Request failed", request_count

# Function to format text as Markdown
def to_markdown(text):
    text = text.replace('•', '  *')
    return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))

# Function to check if file exists
def check_file_exists(file_path):
    if not os.path.isfile(file_path):
        print(f"Error: File not found at {file_path}")
        return False
    return True

# Evaluation functions for each dataset
def evaluate_multirc():
    csv_file_path = '/content/MultiRC_1000_4.0.csv'
    if not check_file_exists(csv_file_path):
        return

    translations_df = pd.read_csv(csv_file_path)
    results = []
    request_count = 0

    for index, row in translations_df.iterrows():
        paragraph = row['Paragraph']
        translated_paragraph = row['Translated Paragraph']
        question = row['Question']
        translated_question = row['Translated Question']
        answer = row['Answer']
        translated_answer = row['Translated Answer']
        actual_label = str(row['Actual Label'])

        se_response, request_count = call_claude(f"Paragraph: \"{paragraph}\"\nQuestion: \"{question}\"\nAnswer: \"{answer}\". Respond with '0' for false or '1' for true.")
        aave_response, request_count = call_claude(f"Paragraph: \"{translated_paragraph}\"\nQuestion: \"{translated_question}\"\nAnswer: \"{translated_answer}\". Respond with '0' for false or '1' for true.")

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
    output_file_path = '/content/drive/MyDrive/Algoverse Comparison/Sonnet Evals'
    results_df.to_csv(output_file_path, index=False)
    print("MultiRC evaluation results saved to CSV file.")

def evaluate_wsc():
    csv_file_path = '/content/WSC_630_4.0.csv'
    if not check_file_exists(csv_file_path):
        return

    translations_df = pd.read_csv(csv_file_path)
    results = []
    request_count = 0

    for index, row in translations_df.iterrows():
        passage = row['Original Passage']
        translated_passage = row['Translated Passage']
        pronoun = row['Pronoun']
        span1 = row['Span1']
        translated_span1 = row['Translated Span1']
        span2 = row['Span2']
        translated_span2 = row['Translated Span2']
        actual_reference = str(row['Actual Reference'])

        se_reference, request_count = call_claude(f"Passage: \"{passage}\"\nPronoun: \"{pronoun}\"\nSpan1: \"{span1}\"\nSpan2: \"{span2}\". Respond with '0' if the pronoun refers to Span1 or '1' if it refers to Span2.", request_count)
        aave_reference, request_count = call_claude(f"Passage: \"{translated_passage}\"\nPronoun: \"{pronoun}\"\nSpan1: \"{translated_span1}\"\nSpan2: \"{translated_span2}\". Respond with '0' if the pronoun refers to Span1 or '1' if it refers to Span2.", request_count)

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
    output_file_path = '/content/drive/MyDrive/Algoverse Comparison/Sonnet Evals'
    results_df.to_csv(output_file_path, index=False)
    print("WSC evaluation results saved to CSV file.")

# Evaluation for COPA
def evaluate_copa():
    csv_file_path = '/content/Copa_superglue_500_4.0.csv'
    if not check_file_exists(csv_file_path):
        return

    translations_df = pd.read_csv(csv_file_path)
    results = []
    request_count = 0

    for index, row in translations_df.iterrows():
        premise = row['Premise']
        translated_premise = row['Translated Premise']
        choice1 = row['Choice1']
        translated_choice1 = row['Translated Choice1']
        choice2 = row['Choice2']
        translated_choice2 = row['Translated Choice2']
        actual_label = row['Actual Label']

        se_response, request_count = call_claude(f"Premise: \"{premise}\"\nChoice1: \"{choice1}\"\nChoice2: \"{choice2}\". Respond with '0' if the first choice is correct or '1' if the second choice is correct.")
        aave_response, request_count = call_claude(f"Premise: \"{translated_premise}\"\nChoice1: \"{translated_choice1}\"\nChoice2: \"{translated_choice2}\". Respond with '0' if the first choice is correct or '1' if the second choice is correct.")

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
    output_file_path = '/content/drive/MyDrive/Algoverse/AAVE Translations/GPT 4.0 Translation, Claude-Sonnet-3.5 Evaluation/Copa_superglue_500_claude.csv'
    results_df.to_csv(output_file_path, index=False)
    print("COPA evaluation results saved to CSV file.")

# Run evaluations for selected datasets
evaluate_multirc()
evaluate_wsc()
evaluate_copa()

print("Selected evaluations completed")


# code below has all five functions

# In[ ]:


import pathlib
import textwrap
import pandas as pd
from IPython.display import display, Markdown
from google.colab import drive
import requests
import time

# Claude API configuration
API_KEY = API_KEY
HEADERS = {
    'x-api-key': API_KEY,
    'Content-Type': 'application/json',
    'anthropic-version': '2023-06-01'
}

MAX_REQUESTS_PER_MINUTE = 50

def call_claude(prompt):
    attempt = 0
    request_count = 0
    while True:
        if request_count >= MAX_REQUESTS_PER_MINUTE:
            print("Rate limit reached. Waiting for 45 seconds...")
            time.sleep(45)
            request_count = 0  # Reset the request count after waiting

        try:
            payload = {
                "model": "claude-3-5-sonnet-20240620",
                "system": "You are a helpful assistant.",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 50
            }
            response = requests.post('https://api.anthropic.com/v1/messages', headers=HEADERS, json=payload)
            request_count += 1
            response_data = response.json()
            if 'content' in response_data:
                message_content = response_data['content'][0]['text']
                return message_content.strip(), request_count
            else:
                print("Error: 'content' key not found in response_data")
                print("Response Data:", response_data)
                return "Request failed", request_count
        except requests.exceptions.RequestException as e:
            print(f"Request exception: {e}")
            return "Request failed", request_count
        except Exception as e:
            if 'rate_limit_error' in str(e):
                print("Rate limit error encountered. Waiting for 20 seconds before retrying...")
                time.sleep(20)
                request_count = 0  # Reset the request count after waiting
            else:
                print(f"Error: {e}")
                return "Request failed", request_count

# Function to format text as Markdown
def to_markdown(text):
    text = text.replace('•', '  *')
    return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))

# Evaluation functions for each dataset
def evaluate_boolq():
    csv_file_path = '/content/BoolQA_superglue_1000_4-0 (1).csv'
    translations_df = pd.read_csv(csv_file_path)
    results = []
    request_count = 0

    for index, row in translations_df.iterrows():
        question = row['se question']
        passage = row['se passage']
        answer = row['actual answer']
        actual_answer = ["True", "True.", "true", "true."] if answer == 1 else ["False", "False.", "false", "false."]

        aave_question = row['aave qustion']
        aave_passage = row['aave passage']

        se_answer_new, request_count = call_claude(f"Passage: \"{passage}\"\nQuestion: \"{question}\". Answer as 'True' or 'False' only.")
        aave_answer, request_count = call_claude(f"Passage: \"{aave_passage}\"\nQuestion: \"{aave_question}\". Answer as 'True' or 'False' only.")

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
    output_file_path = '/content/drive/MyDrive/Algoverse/AAVE Translations/GPT 4.0 Translations, Claude-Sonnet-3.5 Evaluation/BoolQA_superglue_1000_claude.csv'
    results_df.to_csv(output_file_path, index=False)
    print("BoolQ evaluation results saved to CSV file.")

def evaluate_sst2():
    csv_file_path = '/content/SST-2_super-glue_1000_4.0.csv'
    translations_df = pd.read_csv(csv_file_path)
    results = []
    request_count = 0

    for index, row in translations_df.iterrows():
        sentence = row['Original']
        translated_sentence = row['Translated']
        actual_sentiment = "Positive" if row['Original Sentiment'] == 1 else "Negative"

        se_sentiment, request_count = call_claude(f"Analyze the sentiment of the passage: \"{sentence}\". Respond with 'Positive' or 'Negative' only.")
        aave_sentiment, request_count = call_claude(f"Analyze the sentiment of the passage: \"{translated_sentence}\". Respond with 'Positive' or 'Negative' only.")

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
    output_file_path = '/content/drive/MyDrive/Algoverse/AAVE Translations/GPT 4.0 Translations, Claude-Sonnet-3.5 Evaluation/sst-2_super-glue_1000_claude.csv'
    results_df.to_csv(output_file_path, index=False)
    print("SST-2 evaluation results saved to CSV file.")

def evaluate_multirc():
    csv_file_path = '/content/MultiRC_1000_4.0.csv'
    translations_df = pd.read_csv(csv_file_path)
    results = []
    request_count = 0

    for index, row in translations_df.iterrows():
        paragraph = row['Paragraph']
        translated_paragraph = row['Translated Paragraph']
        question = row['Question']
        translated_question = row['Translated Question']
        answer = row['Answer']
        translated_answer = row['Translated Answer']
        actual_label = str(row['Actual Label'])

        se_response, request_count = call_claude(f"Paragraph: \"{paragraph}\"\nQuestion: \"{question}\"\nAnswer: \"{answer}\". Respond with '0' for false or '1' for true.")
        aave_response, request_count = call_claude(f"Paragraph: \"{translated_paragraph}\"\nQuestion: \"{translated_question}\"\nAnswer: \"{translated_answer}\". Respond with '0' for false or '1' for true.")

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
    output_file_path = '/content/drive/MyDrive/Algoverse/AAVE Translations/GPT 4.0 Translations, Claude-Sonnet-3.5 Evaluation/MultiRC_1000_claude.csv'
    results_df.to_csv(output_file_path, index=False)
    print("MultiRC evaluation results saved to CSV file.")

def evaluate_wsc():
    csv_file_path = '/content/WSC_630_4.0.csv'
    translations_df = pd.read_csv(csv_file_path)
    results = []
    request_count = 0

    for index, row in translations_df.iterrows():
        passage = row['Original Passage']
        translated_passage = row['Translated Passage']
        pronoun = row['Pronoun']
        span1 = row['Span1']
        translated_span1 = row['Translated Span1']
        span2 = row['Span2']
        translated_span2 = row['Translated Span2']
        actual_reference = str(row['Actual Reference'])

        se_reference, request_count = call_claude(f"Passage: \"{passage}\"\nPronoun: \"{pronoun}\"\nSpan1: \"{span1}\"\nSpan2: \"{span2}\". Respond with '0' if the pronoun refers to Span1 or '1' if it refers to Span2.", request_count)
        aave_reference, request_count = call_claude(f"Passage: \"{translated_passage}\"\nPronoun: \"{pronoun}\"\nSpan1: \"{translated_span1}\"\nSpan2: \"{translated_span2}\". Respond with '0' if the pronoun refers to Span1 or '1' if it refers to Span2.", request_count)

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
    output_file_path = '/content/drive/MyDrive/Algoverse/AAVE Translations/GPT 4.0 Translations, Claude-Sonnet-3.5 Evaluation/WSC_630_claude.csv'
    results_df.to_csv(output_file_path, index=False)
    print("WSC evaluation results saved to CSV file.")

# Evaluation for COPA
def evaluate_copa():
    csv_file_path = '/content/Copa_superglue_500_4.0.csv'
    translations_df = pd.read_csv(csv_file_path)
    results = []
    request_count = 0

    for index, row in translations_df.iterrows():
        premise = row['Premise']
        translated_premise = row['Translated Premise']
        choice1 = row['Choice1']
        translated_choice1 = row['Translated Choice1']
        choice2 = row['Choice2']
        translated_choice2 = row['Translated Choice2']
        actual_label = row['Actual Label']

        se_response, request_count = call_claude(f"Premise: \"{premise}\"\nChoice1: \"{choice1}\"\nChoice2: \"{choice2}\". Respond with '0' if the first choice is correct or '1' if the second choice is correct.")
        aave_response, request_count = call_claude(f"Premise: \"{translated_premise}\"\nChoice1: \"{translated_choice1}\"\nChoice2: \"{translated_choice2}\". Respond with '0' if the first choice is correct or '1' if the second choice is correct.")

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
    output_file_path = '/content/drive/MyDrive/Algoverse/AAVE Translations/GPT 4.0 Translation, Claude-Sonnet-3.5 Evaluation/Copa_superglue_500_claude.csv'
    results_df.to_csv(output_file_path, index=False)
    print("COPA evaluation results saved to CSV file.")

# Run evaluations for selected datasets
evaluate_boolq()
evaluate_sst2()
evaluate_multirc()
evaluate_wsc()
evaluate_copa()

print("Selected evaluations completed")

