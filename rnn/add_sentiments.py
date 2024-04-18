from datasets import load_dataset
import pandas as pd
# dataset = load_dataset("iwslt2017.py", "iwslt2017-en-zh")
dataset = load_dataset("ngxingyu/iwslt17_google_trans_scores")

#%%
# sentiment analysis model
task='sentiment'
MODEL = f"cardiffnlp/twitter-xlm-roberta-base-{task}-multilingual"
from transformers import AutoModelForSequenceClassification
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer

sentiment_tokenizer = AutoTokenizer.from_pretrained(MODEL)
sentiment_model = AutoModelForSequenceClassification.from_pretrained(MODEL).to('cuda')
sentiment_tokenizer.save_pretrained(MODEL)
sentiment_model.save_pretrained(MODEL)

# %%
from itertools import chain
def add_sentiments(x):
    inputs = []
    x["en_sentiment"] = []
    x["zh_sentiment"] = []
    inputs = x["en"] + x["zh"]
    encoded_inputs = sentiment_tokenizer(
        inputs, return_tensors='pt', padding=True).to('cuda')
    sentiments = sentiment_model(**encoded_inputs)[0].detach().cpu().numpy()
    x["en_sentiment"] = sentiments[:len(x["en"])]
    x["zh_sentiment"] = sentiments[len(x["en"]):]
    return x

dataset_with_sentiments = dataset.map(add_sentiments, batched=True, batch_size=32)
dataset_with_sentiments.push_to_hub(
    repo_id="ngxingyu/iwslt17_google_trans_scores_sentiments", private=True)

# %%
