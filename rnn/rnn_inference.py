# Import necessary libraries and modules
from onmt.translate.beam_search import BeamSearch
from onmt.translate.greedy_search import GreedySearch
from onmt.translate import GNMTGlobalScorer
from translate_rnn import RNNSeq2Seq
import re
from data import IWSLTSentimentDataModule
import torch
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Define functions

def get_language(text):
    """Detect the language of the input text."""
    if re.match(r"[\u4e00-\u9fff]+", text):
        return "zh"
    return "en"


# Define beam search and greedy search strategies
dm = IWSLTSentimentDataModule(
    tokenizer="sentencepiece",
    use_google_zh=True,
    shared_tokenizer=False,
    vocab_size=8000,
    model_type="unigram"
)

beam_search_strategy = BeamSearch(
    beam_size=4,
    batch_size=1,
    pad=dm.pad_id,
    bos=dm.bos_id,
    eos=dm.eos_id,
    unk=dm.unk_id,
    start=dm.bos_id,
    n_best=1,
    global_scorer=GNMTGlobalScorer(
        alpha=0, beta=0,
        length_penalty="none",
        coverage_penalty="none"
    ),
    min_length=1,
    max_length=128,
    ratio=0.0,
    ban_unk_token=False,
    block_ngram_repeat=0,
    exclusion_tokens=set(),
    return_attention=False,
    stepwise_penalty=False
)

greedy_strategy = GreedySearch(
    pad=dm.pad_id,
    bos=dm.bos_id,
    eos=dm.eos_id,
    unk=dm.unk_id,
    start=dm.bos_id,
    n_best=1,
    batch_size=1,
    global_scorer=GNMTGlobalScorer(
        alpha=0.6, beta=0.2,
        length_penalty="wu",
        coverage_penalty="none"
    ),
    min_length=1,
    max_length=128,
    sampling_temp=1.0,
    keep_topk=1,
    keep_topp=0.0,
    beam_size=1,
    ban_unk_token=False,
    block_ngram_repeat=2,
    exclusion_tokens=set(),
    return_attention=False
)

def translate_sentence_with_strategy(batch, model, strategy=beam_search_strategy):
    """
    Translate a sentence using the given strategy.
    """
    with torch.no_grad():
        if isinstance(batch, str):
            batch = [batch]
        if len(batch) == 0:
            return []
        source = get_language(batch[0])

        if source == "en":
            encoder = model.en_encoder
            decoder = model.zh_decoder
            encode = dm.encode_en
            source_decode = dm.decode_en
            decode = dm.decode_zh
        else:
            encoder = model.zh_encoder
            decoder = model.en_decoder
            encode = dm.encode_zh
            source_decode = dm.decode_zh
            decode = dm.decode_en
        
        translations = []
        attns = []
        input_tokens=[]
        output_tokens=[]

        for sentence in batch:
            _attns = []
            tokens = torch.tensor(encode(sentence)).to(model.device).unsqueeze(0)
            if tokens.shape[1] == 0:
                translations.append("")
                continue

            outputs, hidden = encoder(tokens)
            src_len = torch.tensor([x.shape for x in tokens]).to(model.device)
            (fn_map_state, src, src_map) = strategy.initialize(outputs, src_len)
            for step in range(strategy.max_length):
                decoder_input = strategy.current_predictions
                decoder_input = decoder_input.unsqueeze(1)
                decoder_outputs = []
                decoder_attns = []
                for i in range(len(decoder_input)):
                    decoder_output, hidden, attn = decoder._step(decoder_input[i:i+1].to(model.device), hidden, outputs)
                    log_probs = torch.log_softmax(decoder_output, dim=-1)
                    decoder_outputs.append(log_probs)
                    decoder_attns.append(attn)
                strategy.advance(torch.stack(decoder_outputs).detach().squeeze(1), torch.stack(decoder_attns).detach().squeeze(1))
                any_finished = any(
                    [any(sublist) for sublist in strategy.is_finished_list]
                )
                _attns.append(decoder_attns)
                if any_finished:
                    strategy.update_finished()
                    if strategy.done:
                        break
                select_indices = strategy.select_indices

                if any_finished:
                    if isinstance(outputs, tuple):
                        enc_out = tuple(x[select_indices] for x in outputs)
                    else:
                        enc_out = outputs[select_indices]
            translations.append(decode(list(int(i) for i in strategy.alive_seq[0].cpu().numpy())))
            attns.append(_attns)
            input_tokens.append([source_decode(int(i)) for i in tokens[0].cpu().numpy()])
            output_tokens.append([decode(int(i)) for i in strategy.alive_seq[0].cpu().numpy()])
        return translations, attns, input_tokens, output_tokens

# Initialize data module and model

iwslt_dm = IWSLTSentimentDataModule(tokenizer="sentencepiece",
                                    use_google_zh=True,
                                    shared_tokenizer=False,
                                    vocab_size=8000,
                                    model_type="unigram")
iwslt_dm.setup()

ckpt_name = "lstmlstm2layer_google_8000_512_256_32len_costf_itl_uni_14_-1045.09"
checkpoint = torch.load(f"checkpoints/{ckpt_name}.ckpt")
dm = checkpoint["hyper_parameters"]['dm']
rnn_model = RNNSeq2Seq(
    dm,
    hidden_dim=checkpoint["hyper_parameters"]["hidden_dim"],
    num_layers=checkpoint["hyper_parameters"]["num_layers"],
    bidirectional=checkpoint["hyper_parameters"]["bidirectional"],
    embedding_dim=checkpoint["hyper_parameters"]["embedding_dim"],
    encoder_type=checkpoint["hyper_parameters"]["encoder_type"],
    decoder_type=checkpoint["hyper_parameters"]["decoder_type"],
)
rnn_model.load_state_dict(checkpoint["state_dict"])
rnn_model.cuda()
rnn_model.eval()

# Perform translation

results = []
test_dl = dm.test_dataloader(batch_size=32)
for text in tqdm(iter(test_dl)):
    zh_preds_greedy = translate_sentence_with_strategy(
        text["en"],
        rnn_model,
        strategy=greedy_strategy,
    )
    en_preds_greedy = translate_sentence_with_strategy(
        text["zh"],
        rnn_model,
        strategy=greedy_strategy,
    )
    results.extend(list(zip(
        text["en"], text["zh"], zh_preds_greedy, en_preds_greedy
    )))

# Generate predictions and save to CSV

preds = []
test_dl = dm.test_dataloader(batch_size=32)
for batch in tqdm(iter(test_dl)):
    for i in zip(batch["en"], batch["zh"]):
        en_pred = rnn_model.predict_text(i[0])
        zh_pred = rnn_model.predict_text(i[1])
        preds.append((i[0], i[1], en_pred[0], zh_pred[0]))

pd.DataFrame(preds, columns=["en", "zh", "en_pred", "zh_pred"]).to_csv(f"predictions_{ckpt_name}.csv")



# Plot attention maps for a few examples
from matplotlib import font_manager
fontP = font_manager.FontProperties()
fontP.set_family('SimHei')
fontP.set_size('large')

texts = ["I am happy", "I am sad", "I am angry", "I am excited"]
result = translate_sentence_with_strategy(texts, rnn_model, strategy=greedy_strategy)

for i in range(len(texts)):
    attn = torch.stack([x[0][0] for x in result[1][i]])
    output_tokens = result[3][i][-attn.shape[0]:]
    input_tokens = result[2][i][:attn.shape[1]]
    output_mask = [0 if x=="" else 1 for x in output_tokens]
    attn = attn[np.argwhere(output_mask)].flatten(1)
    output_tokens = [x for x, y in zip(output_tokens, output_mask) if y]

    plt.figure(figsize=(10,10))
    df = pd.DataFrame(attn.cpu().numpy(), index=output_tokens, columns=input_tokens)
    plt.imshow(attn.cpu().numpy(), cmap="hot")
    plt.yticks(ticks=range(len(output_tokens)), labels=output_tokens, fontproperties=fontP)
    plt.xticks(ticks=range(len(input_tokens)), labels=input_tokens, rotation=45)
