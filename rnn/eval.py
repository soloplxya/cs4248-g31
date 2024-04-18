"""
This module contains the evaluation functions for the model.
Various dependencies are required to run this module.

pip install git+https://github.com/lucadiliello/bleurt-pytorch.git
pip install unbabel-comet
pip install sacrebleu
pip install rouge-score
"""
import torch
from bleurt_pytorch import BleurtConfig, BleurtForSequenceClassification, BleurtTokenizer

from comet import download_model, load_from_checkpoint
bleurt_checkpoint = "lucadiliello/BLEURT-20-D12"
bleurt_config = BleurtConfig.from_pretrained(bleurt_checkpoint)
bleurt_model = BleurtForSequenceClassification.from_pretrained(bleurt_checkpoint).to("cuda" if torch.cuda.is_available() else "cpu")
bleurt_tokenizer = BleurtTokenizer.from_pretrained(bleurt_checkpoint)
comet_checkpoint = "Unbabel/wmt22-comet-da"
comet_model_path = download_model(comet_checkpoint)
comet_model = load_from_checkpoint(comet_model_path)

from sacrebleu.metrics import BLEU, CHRF, TER
bleu = BLEU()
chrf = CHRF()
ter = TER()

from rouge_score import rouge_scorer
rouge = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
def compute_rouge_score(references, candidates):
    global rouge
    scores = []
    for src, tgt in zip(references, candidates):
        scores.append(rouge.score(src, tgt)['rougeL'].fmeasure)
    return scores

def compute_bleu_score(references, candidates):
    global bleu
    try:
        score = bleu.corpus_score([str(x) for x in candidates], [[str(x)] for x in references])
        print(score)
        return score
    except:
        return 0

def compute_chrf_score(references, candidates):
    global chrf
    return chrf.corpus_score([x for x in candidates], [[x] for x in references])

def compute_ter_score(references, candidates):
    global ter
    return ter.corpus_score([x for x in candidates], [[x] for x in references])
def compute_bleurt_score(references, candidates, checkpoint="lucadiliello/BLEURT-20-D12"):
    global bleurt_checkpoint, bleurt_model, bleurt_tokenizer, bleurt_config
    if checkpoint != bleurt_checkpoint:
        bleurt_checkpoint = checkpoint
        bleurt_config = BleurtConfig.from_pretrained(bleurt_checkpoint)
        bleurt_model = BleurtForSequenceClassification.from_pretrained(bleurt_checkpoint).to("cuda" if torch.cuda.is_available() else "cpu")
        bleurt_tokenizer = BleurtTokenizer.from_pretrained(bleurt_checkpoint)
    scores = []
    for ref, cand in zip(references, candidates):
        inputs = bleurt_tokenizer(ref, cand, return_tensors="pt", padding=True, truncation=True).to("cuda" if torch.cuda.is_available() else "cpu")
        outputs = bleurt_model(**inputs).logits.flatten().tolist()
        scores.append(outputs[0])
    return scores

def compute_comet_score(en, references, candidates, checkpoint="Unbabel/wmt22-comet-da"):
    global comet_checkpoint, comet_model, comet_model_path
    if checkpoint != comet_checkpoint:
        comet_checkpoint = checkpoint
        comet_model_path = download_model(comet_checkpoint)
        comet_model = load_from_checkpoint(comet_model_path)
    data = [
        {"src": src, "mt": mt, "ref": ref}
        for src, mt, ref in zip(en, candidates, references)
    ]

    model_output = comet_model.predict(data, batch_size=64, gpus=1)
    return model_output["scores"]

if __name__ == "__main__":
    """
    This is a sample code for usage of the evaluation functions.
    """
    from datasets import load_dataset
    dataset = load_dataset("ngxingyu/iwslt17_google_trans")
    english = [x["en"] for x in dataset["validation"]]
    references = [x["zh"] for x in dataset["validation"]]
    candidates = [x["google_zh"] for x in dataset["validation"]]
    bleurt_scores = compute_bleurt_score(references, candidates)
    comet_scores = compute_comet_score(english, references, candidates)
    bleu_scores = compute_bleu_score(references, candidates)

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    results = pd.DataFrame({"bleurt_scores": bleurt_scores, "comet_scores": comet_scores, "english": english, "references": references, "candidates": candidates}).sort_values("comet_scores", ascending=True)

    print(results)

    print(f"BLEURT: {np.mean(bleurt_scores)}")
    print(f"COMET: {np.mean(comet_scores)}")
    print(f"BLEU: {bleu_scores}")
    plt.scatter(bleurt_scores, comet_scores)


    # import pandas as pd
    # import numpy as np
    # predictions = pd.read_csv("predictions_bilstm_8000_0.7_256_1layer.csv")
    # predictions["en_pred"]=predictions["en_pred"].map(lambda x: x[2:-2])
    # predictions["zh_pred"]=predictions["zh_pred"].map(lambda x: x[2:-2])


    # zh_bleurt_scores = compute_bleurt_score(predictions["zh"], predictions["zh_pred"])
    # en_bleurt_scores = compute_bleurt_score(predictions["en"], predictions["en_pred"])
    # zh_comet_scores = compute_comet_score(predictions["en"], predictions["zh"], predictions["zh_pred"])
    # en_comet_scores = compute_comet_score(predictions["en"], predictions["en"], predictions["en_pred"])
    # print(np.mean(zh_bleurt_scores), np.mean(zh_comet_scores))
    # print(np.mean(en_bleurt_scores), np.mean(en_comet_scores))

    # results = pd.DataFrame({"zh_bleurt_scores": zh_bleurt_scores, "zh_comet_scores": zh_comet_scores, "en_bleurt_scores": en_bleurt_scores, "en_comet_scores": en_comet_scores, "en": predictions["en"], "zh": predictions["zh"], "en_pred": predictions["en_pred"], "zh_pred": predictions["zh_pred"]}).sort_values("zh_comet_scores", ascending=True)
