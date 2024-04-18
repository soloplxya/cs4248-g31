"""
Removes rows with duplicate en/zh ground truth translations
Identifies consecutive rows with identical zh translations, combines rows by concatenating en values. Fixes issue where english and chinese translations are misaligned for traning/validation set
> shown to improve BLEURT and COMET scores for affected instances
BLEURT average: 0.266 -> 0.350
COMET average: 0.609 -> 0.695
"""
# %%
from datasets import load_dataset
import json
train_dataset_with_scores = json.load(open("train_with_scores.json", "r"))

# %%
from eval import compute_bleurt_score, compute_comet_score
from collections import defaultdict

scores = defaultdict(list)


# %%
def compute_scores(data, split):
    global scores
    if split in data:
        data_split = data[split]["translation"]
    else:
        return None

    english_sentences = [x["en"] for x in data_split]
    references = [x["zh"] for x in data_split]
    candidates = [x["google_zh"] for x in data_split]
    bleurt_scores = compute_bleurt_score(references, candidates)
    comet_scores = compute_comet_score(english_sentences, references, candidates)

    scores[split] = []
    for i, x in enumerate(data_split):
        scores[split].append(x)
        # scores[split]['translation'][i] = x
        scores[split][i]["bleurt_score"] = bleurt_scores[i]
        scores[split][i]["comet_score"] = comet_scores[i]
    return data


dataset = load_dataset("ngxingyu/iwslt17_google_trans")
train_dataset_with_scores = compute_scores(dataset, "train")
val_dataset_with_scores = compute_scores(dataset, "validation")
test_dataset_with_scores = compute_scores(dataset, "test")

import json

with open("train_val_test_with_scores.json", "w") as f:
    json.dump(scores, f)
dataset_with_scores = json.load(open("train_val_test_with_scores.json", "r"))
scores = {}
for split in ["train", "validation", "test"]:
    for data in dataset_with_scores[split]:
        scores[data["en"]] = {
            "bleurt_score": data["bleurt_score"],
            "comet_score": data["comet_score"],
        }


def compute_scores_map(data):
    global scores
    for sample in data:
        if sample["en"] in scores:
            sample["bleurt_score"] = scores[sample["en"]]["bleurt_score"]
            sample["comet_score"] = scores[sample["en"]]["comet_score"]
        else:
            print(f"{sample['en']} not found")

    return data


dataset = load_dataset("ngxingyu/iwslt17_google_trans")
dataset_with_scores = dataset.map(compute_scores_map, batched=True, batch_size=1000)
dataset_with_scores.push_to_hub(
    repo_id="ngxingyu/iwslt17_google_trans_scores", private=True
)
# %%
import pandas as pd

dataset_with_scores = load_dataset("ngxingyu/iwslt17_google_trans_scores")
val_df = pd.DataFrame(dataset_with_scores["validation"]["translation"])
val_df.drop_duplicates(subset=['en','zh'], inplace=True)

test_df = pd.DataFrame(dataset_with_scores["test"]["translation"])
test_df.drop_duplicates(subset=['en','zh'], inplace=True)

train_df = pd.DataFrame(dataset_with_scores["train"]["translation"])
train_df.drop_duplicates(subset=['en','zh'], inplace=True)

#%%
dfs = {
    "train": train_df,
    "validation": val_df,
    # "test": test_df
}
from deep_translator import GoogleTranslator
translator= GoogleTranslator(source='en', target='zh-CN')
for split, df in dfs.items():
    df.drop_duplicates(subset=['en','zh'], inplace=True)
    split_a = df[df.duplicated(['zh'])]
    split_b = df[~df.duplicated(['zh'])]
    rows_to_combine = set()
    for i in range(len(split_a) - 1):
        if split_a.iloc[i]['zh'] == split_a.iloc[i + 1]['zh'] and split_a.iloc[i].name == split_a.iloc[i + 1].name - 1:
            rows_to_combine = rows_to_combine.union({split_a.iloc[i].name, split_a.iloc[i + 1].name})
    if len(rows_to_combine) == 0:
        print("No rows to combine")
        continue
    to_combine = (split_a.loc[list(rows_to_combine)]
                .reset_index()
                .sort_values(['index'])
                .groupby("zh")
                .aggregate({
                "index": "min",
                "bleurt_score": "max",
                "comet_score": "max",
                "zh": "first",
                "en": ' '.join,
                "google_zh": ' '.join,
            })
            .set_index("index")
            .rename_axis(None, axis=0)
            .sort_index(ascending=True)
    )

    for i, idx in enumerate(to_combine.index):
        split_a.at[idx, 'google_zh'] = translator.translate(to_combine.at[idx, 'en'])
    new_comet_scores = compute_comet_score(
        list(to_combine['en'].values), list(to_combine['zh'].values), list(to_combine['google_zh'].values))
    new_bleurt_scores = compute_bleurt_score(
        list(to_combine['zh'].values), list(to_combine['google_zh'].values))
    to_combine['comet_score'] = new_comet_scores
    to_combine['bleurt_score'] = new_bleurt_scores
    print(split_a.loc[list(rows_to_combine)]["bleurt_score"].mean(), to_combine["bleurt_score"].mean())
    print(split_a.loc[list(rows_to_combine)]["comet_score"].mean(), to_combine["comet_score"].mean())
    split_a.drop(list(rows_to_combine), inplace=True)
    dfs[split] = pd.concat([split_a, to_combine, split_b]).sort_index(ascending=True)

#%%
from datasets import DatasetDict, Dataset
splits = {
    split: Dataset.from_dict({"translation": dfs[split].to_dict(orient="records")}) for split in ["train", "validation"]
}
splits["test"] = dataset_with_scores["test"]
new_dataset = DatasetDict(splits)
new_dataset.push_to_hub(
    repo_id="ngxingyu/iwslt17_google_trans_scores", private=False
)
