#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install datasets')
get_ipython().system('pip install openai==0.28')
get_ipython().system('pip install pandas')
get_ipython().system('pip install google-colab')


# In[ ]:


from google.colab import drive
drive.mount('/content/drive')
import openai
import pandas as pd


# In[ ]:


# Set up your OpenAI API key
openai.api_key = "API KEY"

# Set up OpenAI organization key
openai.organization = "Organization"


# In[ ]:


import pandas as pd
import os

def evaluate_fluency(texts):
    fluency_scores = []
    for i, text in enumerate(texts):
        try:
            prompt = f"Rate the fluency of the following text on a scale of 1 to 10 and return only the number. Not 1 extra character other than a number: {text}"
            response = openai.ChatCompletion.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=3,
                temperature=0
            )
            score = response['choices'][0]['message']['content'].strip()
            fluency_scores.append(score)
            print(f"Processed {i + 1}/{len(texts)} entries. Score: {score}")

            # Print every 5 scores
            if (i + 1) % 5 == 0:
                print(f"Processed {i + 1} entries. Scores so far: {fluency_scores[-5:]}")

        except openai.error.RateLimitError:
            print("Rate limit exceeded, waiting 10 seconds...")
            continue  # retry the current item
        except Exception as e:
            print(f"An error occurred: {e}")
            continue
    return fluency_scores

# Function to calculate average fluency score
def calculate_average_fluency(fluency_scores):
    scores = []
    for score in fluency_scores:
        try:
            # Extract numerical score from the response
            score_value = float(score)  # Assuming the score is the only token
            scores.append(score_value)
        except ValueError:
            print(f"Skipping non-numerical score: {score}")  # Debugging information
            continue  # Ignore non-numerical scores
    return sum(scores) / len(scores) if scores else None

def save_fluency_scores(test_name, category, scores):
    directory = f'/content/drive/MyDrive/Algoverse/Validation Scores/GPT 4o/Fluency Scores/{test_name}'
    os.makedirs(directory, exist_ok=True)

    filepath = f'{directory}/Fluency_Scores_{test_name}.txt'
    with open(filepath, 'a') as f:  # Use 'a' to append to the file
        average_fluency = calculate_average_fluency(scores)
        f.write(f"Average Fluency Score for GPT Translated {category}:\n")
        f.write(f'{average_fluency}\n\n')

    print(f"Fluency evaluation for {category} complete. Scores saved to '{filepath}'.")

# Load your datasets
# Ensure these paths are correct
translations_BoolQ_df = pd.read_csv('/content/drive/MyDrive/Algoverse/Values Translations/GPT 4o/BoolQ_filtered.csv')
translations_Copa_df = pd.read_csv('/content/drive/MyDrive/Algoverse/Values Translations/GPT 4o/Copa_Filtered.csv')
translations_SST2_df = pd.read_csv('/content/drive/MyDrive/Algoverse/Values Translations/GPT 4o/SST-2_filtered.csv')
translations_Wsc_df = pd.read_csv('/content/drive/MyDrive/Algoverse/Values Translations/GPT 4o/WSC_filtered.csv')

# Extract columns for fluency evaluation
sst2_texts = translations_SST2_df['Translated'].tolist()
copa_premise_texts = translations_Copa_df['Translated Premise'].tolist()
copa_choice1_texts = translations_Copa_df['Translated Choice1'].tolist()
copa_choice2_texts = translations_Copa_df['Translated Choice2'].tolist()
boolq_passage_texts = translations_BoolQ_df['Translated Passage'].tolist()
boolq_question_texts = translations_BoolQ_df['Translated Question'].tolist()
wsc_texts = translations_Wsc_df['Translated Passage'].tolist()

# Evaluate fluency scores for each subset
sst2_fluency_scores = evaluate_fluency(sst2_texts)
copa_premise_fluency_scores = evaluate_fluency(copa_premise_texts)
copa_choice1_fluency_scores = evaluate_fluency(copa_choice1_texts)
copa_choice2_fluency_scores = evaluate_fluency(copa_choice2_texts)
boolq_passage_fluency_scores = evaluate_fluency(boolq_passage_texts)
boolq_question_fluency_scores = evaluate_fluency(boolq_question_texts)
wsc_fluency_scores = evaluate_fluency(wsc_texts)

# Save fluency scores to respective files
save_fluency_scores('SST2', 'Translated', sst2_fluency_scores)
save_fluency_scores('Copa', 'Translated Premise', copa_premise_fluency_scores)
save_fluency_scores('Copa', 'Translated Choice1', copa_choice1_fluency_scores)
save_fluency_scores('Copa', 'Translated Choice2', copa_choice2_fluency_scores)
save_fluency_scores('BoolQ', 'Translated Passage', boolq_passage_fluency_scores)
save_fluency_scores('BoolQ', 'Translated Question', boolq_question_fluency_scores)
save_fluency_scores('Wsc', 'Translated Passage', wsc_fluency_scores)


# In[ ]:




