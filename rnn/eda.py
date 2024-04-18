"""
Here are some of the steps taken to analyze and preprocess the raw IWSLT17 dataset
"""

# %%
from datasets import load_dataset
import pandas as pd

dataset = load_dataset("ngxingyu/iwslt17_google_trans_scores")
print(f"Number of training examples: {len(dataset['train'])}")
print(f"Number of validation examples: {len(dataset['validation'])}")
print(f"Number of test examples: {len(dataset['test'])}")

train_df = pd.DataFrame(dataset["train"])
val_df = pd.DataFrame(dataset["validation"])
test_df = pd.DataFrame(dataset["test"])

# %%
# Retrain sentencepiece processors
from data import IWSLTSentimentDataModule

dm = IWSLTSentimentDataModule(
    vocab_size=8000, retrain=False, dataset="ngxingyu/iwslt17_google_trans"
)
dm.setup()

en_sp = dm.get_sentencepiece_processor("en", 8000)
zh_sp = dm.get_sentencepiece_processor("zh", 8000)
en_zh_sp = dm.get_sentencepiece_processor("en_zh", 8000)

# %%
en_sentences = [x["en"] for x in dataset["train"]]
zh_sentences = [x["zh"] for x in dataset["train"]]

en_lengths = [len(en_sp.encode_as_pieces(x)) for x in en_sentences]
zh_lengths = [len(zh_sp.encode_as_pieces(x)) for x in zh_sentences]

en_shared_lengths = [len(en_zh_sp.encode_as_pieces(x)) for x in en_sentences]
zh_shared_lengths = [len(en_zh_sp.encode_as_pieces(x)) for x in zh_sentences]

import matplotlib.pyplot as plt

plt.hist(zh_lengths, bins=50, label="zh")
plt.hist(en_lengths, bins=50, label="en")
plt.legend()
plt.show()

plt.hist(zh_shared_lengths, bins=50, label="zh")
plt.hist(en_shared_lengths, bins=50, label="en")
plt.legend()
plt.show()
import pandas as pd

pd.DataFrame(
    {
        "en": en_lengths,
        "zh": zh_lengths,
        "en_shared": en_shared_lengths,
        "zh_shared": zh_shared_lengths,
    }
).describe()

# %%
# example tokenization of english sentence with shared embedding
print(zh_sp.encode_as_pieces("最大的山顶采矿是head of Massey Coal"))
print(en_zh_sp.encode_as_pieces("最大的山顶采矿是head of Massey Coal"))

print(zh_sp.encode_as_pieces("旁白：Repower America 。是时候梦想成真了！"))
print(en_zh_sp.encode_as_pieces("旁白：Repower America 。是时候梦想成真了！"))
print(
    zh_sp.encode_as_pieces(
        "我不理解。他们说，要回到没有汽车、 没有twitter或《美国偶像》的年代。"
    )
)
print(
    en_zh_sp.encode_as_pieces(
        "我不理解。他们说，要回到没有汽车、 没有twitter或《美国偶像》的年代。"
    )
)

# %%
# Find long sentences
train_df["en_length"] = [len(en_sp.encode_as_pieces(x)) for x in train_df["en"]]
train_df["zh_length"] = [len(zh_sp.encode_as_pieces(x)) for x in train_df["zh"]]
import matplotlib.pyplot as plt

plt.hist(train_df["en_length"], bins=50)
plt.hist(train_df["zh_length"], bins=50)
plt.show()

plt.scatter(train_df["en_length"], train_df["zh_length"])
plt.show()

import pandas as pd
import numpy as np

dfs = {"train": train_df, "val": val_df}
"""
Finds sentences with more than 128 tokens, splits them into smaller sentences based on various heuristics e.g. splitting at same punctuation marks

"""
for split, df in dfs.items():
    df["en_length"] = [len(en_sp.encode_as_pieces(x)) for x in df["en"]]
    df["zh_length"] = [len(zh_sp.encode_as_pieces(x)) for x in df["zh"]]

    long_sentences = df[np.logical_or(df["en_length"] > 128, df["zh_length"] > 128)]

    import re

    processed = []
    unprocessed = []
    unprocessed = [
        {"en": r[1]["en"], "zh": r[1]["zh"]} for r in long_sentences.iterrows()
    ]

    new_rows = []
    for sentence in unprocessed:
        splits = [(sentence["en"], sentence["zh"])]

        for pattern in [
            r"\[?[^?!?？!!]+[?？!!]*\"?[\]】]?",
            r"\[?[^\]】]+[\]】]?",
            r"\[?[^.。]+[.。]?",
            r"\[?[^!! ]+[!! ]?",
            r"\[?[^,，、]+[,，、]?",
            r"\[?♪?[^♪]*♪?",
        ]:
            new_splits = []
            for pair in splits:
                proposed_en_splits = re.findall(pattern, pair[0])
                proposed_zh_splits = re.findall(pattern, pair[1])
                if len(proposed_en_splits) == len(proposed_zh_splits):
                    proposed_new_splits = list(
                        zip(proposed_en_splits, proposed_zh_splits)
                    )
                    i = 0
                    while i < len(proposed_new_splits) - 1:
                        en_len = len(en_sp.encode_as_pieces(proposed_new_splits[i][0]))
                        zh_len = len(zh_sp.encode_as_pieces(proposed_new_splits[i][1]))
                        next_en_len = len(
                            en_sp.encode_as_pieces(proposed_new_splits[i + 1][0])
                        )
                        next_zh_len = len(
                            zh_sp.encode_as_pieces(proposed_new_splits[i + 1][1])
                        )
                        if (en_len + next_en_len < 128) and (
                            zh_len + next_zh_len < 128
                        ):
                            proposed_new_splits[i] = (
                                proposed_new_splits[i][0]
                                + proposed_new_splits[i + 1][0],
                                proposed_new_splits[i][1]
                                + proposed_new_splits[i + 1][1],
                            )
                            proposed_new_splits.pop(i + 1)
                            i += 1
                        else:
                            i += 1
                    new_splits.extend(proposed_new_splits)
                else:
                    new_splits.append(pair)
            splits = new_splits
        new_rows.extend(splits)

    valid_df = df[
        np.logical_and(
            np.logical_and(df["en_length"] <= 128, df["zh_length"] <= 128),
            np.logical_and(df["en_length"] > 0, df["zh_length"] > 0),
        )
    ]

    from eval import compute_bleurt_score, compute_comet_score
    from deep_translator import GoogleTranslator

    translator = GoogleTranslator(source="en", target="zh-CN")

    for row in new_rows:
        google_zh = translator.translate(row[0])
        valid_df.loc[valid_df.index[-1] + 1] = [
            0,
            0,
            row[0],
            google_zh,
            row[1],
            len(en_sp.encode_as_pieces(row[0])),
            len(zh_sp.encode_as_pieces(row[1])),
        ]

    invalid_rows = valid_df[valid_df["bleurt_score"] == 0]
    bleurt_scores = compute_bleurt_score(invalid_rows["zh"], invalid_rows["en"])
    valid_df.loc[valid_df["bleurt_score"] == 0, "bleurt_score"] = bleurt_scores
    invalid_rows = valid_df[valid_df["comet_score"] == 0]
    comet_scores = compute_comet_score(
        invalid_rows["en"], invalid_rows["zh"], invalid_rows["google_zh"]
    )
    valid_df.loc[valid_df["comet_score"] == 0, "comet_score"] = comet_scores
    valid_df.drop_duplicates(["en", "zh"], inplace=True)
    dfs[split] = valid_df[
        np.logical_and(
            np.logical_and(valid_df["en_length"] <= 128, valid_df["zh_length"] <= 128),
            np.logical_and(valid_df["en_length"] > 0, valid_df["zh_length"] > 0),
        )
    ]


from datasets import DatasetDict, Dataset

splits = {
    split: Dataset.from_dict(
        {
            "translation": dfs[split]
            .drop(columns=["en_length", "zh_length"])
            .to_dict(orient="records")
        }
    )
    for split in ["train", "validation"]
}
splits["test"] = dataset["test"]
new_dataset = DatasetDict(splits)

new_dataset.push_to_hub(repo_id="ngxingyu/iwslt17_google_trans_scores", private=False)

# %%
[
    x
    for x in new_rows
    if len(en_sp.encode_as_pieces(x[0])) > 128
    or len(zh_sp.encode_as_pieces(x[1])) > 128
]
long_sentences[
    long_sentences["en"].map(lambda x: x.count("."))
    == long_sentences["zh"].map(lambda x: x.count("。"))
]


# %%
import nltk.tokenize.punkt
import pickle
import codecs

zh_tokenizer = nltk.tokenize.punkt.PunktSentenceTokenizer()
text = codecs.open("iwslt_train_zh.csv", "r", "utf8").read()
zh_tokenizer.train(text)
out = open("iwslt_zh_plain.pk", "wb")
pickle.dump(zh_tokenizer, out)
out.close()

# %%
import nltk.tokenize.punkt
import pickle
import codecs

zh_tokenizer = pickle.load(open("iwslt_zh_plain.pk", "rb"))
zh_tokenizer.sentences_from_text("我是一句话。")
# %%

from nltk.tokenize import sent_tokenize

# lengths after splitting by sentences
max_en_chunked = [
    (
        max([len(en_sp.encode_as_pieces(part)) for part in sent_tokenize(x)]),
        sent_tokenize(x),
    )
    for x in en_long
]
max_zh_chunked = [
    (
        max(
            [
                len(zh_sp.encode_as_pieces(part))
                for part in zh_tokenizer.sentences_from_text(x)
            ]
        ),
        zh_tokenizer.sentences_from_text(x),
    )
    for x in zh_long
]

# plot new histograms

plt.hist([x[0] for x in max_en_chunked], bins=50)
plt.hist([x[0] for x in max_zh_chunked], bins=50)
plt.show()

print(pd.DataFrame({"en": [x[0] for x in max_en_chunked]}).describe())
print(pd.DataFrame({"zh": [x[0] for x in max_zh_chunked]}).describe())

# How to handle?
# https://arxiv.org/pdf/2312.05172.pdf
# %%

from transformers import AutoModelForSequenceClassification
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer
import numpy as np
from scipy.special import softmax
import csv
import urllib.request


# Preprocess text (username and link placeholders)
def preprocess(text):
    new_text = []

    for t in text.split(" "):
        t = "@user" if t.startswith("@") and len(t) > 1 else t
        t = "http" if t.startswith("http") else t
        new_text.append(t)
    return " ".join(new_text)


task = "sentiment"
MODEL = f"cardiffnlp/twitter-xlm-roberta-base-{task}-multilingual"

tokenizer = AutoTokenizer.from_pretrained(MODEL)

# download label mapping
labels = []
mapping_link = f"https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/{task}/mapping.txt"
with urllib.request.urlopen(mapping_link) as f:
    html = f.read().decode("utf-8").split("\n")
    csvreader = csv.reader(html, delimiter="\t")
labels = [row[1] for row in csvreader if len(row) > 1]

# PT
model = AutoModelForSequenceClassification.from_pretrained(MODEL)
tokenizer.save_pretrained(MODEL)
model.save_pretrained(MODEL)

# %%

en_sentiments = []
for i in dataset["train"]["translation"][:200]:
    text = i["en"]
    text = preprocess(text)
    encoded_input = tokenizer(text, return_tensors="pt")
    output = model(**encoded_input)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    en_sentiments.append(scores)

zh_sentiments = []
for i in dataset["train"]["translation"][:200]:
    text = i["zh"]
    text = preprocess(text)
    encoded_input = tokenizer(text, return_tensors="pt")
    output = model(**encoded_input)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    zh_sentiments.append(scores)

translated_sentiments = []
translated_sentences = {}
# %%
# pip install -U deep-translator
import json
from deep_translator import GoogleTranslator

translator = GoogleTranslator(source="en", target="zh-CN")

google_translation = json.load(open("tmp_translations.json", "r"))

# %%

from multiprocessing import Process, Manager

manager = Manager()
d = manager.dict()
d.update(google_translation)


# %%
def process_list(sentences, google_translation):
    for text in sentences:
        if text in google_translation.keys():
            continue

        translated = translator.translate(text)
        google_translation[text] = translated
        if len(google_translation) % 100 == 0:
            print(len(google_translation))
        continue


processes = []
n_proc = 2
train_en = [x["en"] for x in dataset["train"]["translation"]]
for i in range(n_proc):
    p = Process(target=process_list, args=(train_en[i::n_proc], d))
    p.start()
    processes.append(p)
for p in processes:
    p.join()
