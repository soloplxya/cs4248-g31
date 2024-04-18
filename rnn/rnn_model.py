#!/usr/bin/env python3
"""
This script is used to train the various RNN model for en-zh translation
"""
import os
import re
import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from types import SimpleNamespace
from torch.nn import functional as F
from torch import optim
from torch import nn

from eval import compute_bleurt_score
from rnn import Encoder, Decoder


class In_trust_Loss(nn.Module):
    """
    In-trust loss as described in the paper
    Source: https://github.com/WENGSYX/Low-resource-text-translation/blob/main/code/main.py
    """
    def __init__(self, alpha=1, beta=0.8, delta=0.5, num_classes=8000):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes
        self.delta = delta
        self.cross_entropy = torch.nn.CrossEntropyLoss()

    def forward(self, logits, labels):
        ce = self.cross_entropy(logits, labels)
        # Loss In_trust
        active_logits = logits.view(-1, self.num_classes)
        active_labels = labels.view(-1)

        pred = F.softmax(active_logits, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = torch.nn.functional.one_hot(
            active_labels, self.num_classes
        ).float()
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        dce = -1 * torch.sum(
            pred * torch.log(pred * self.delta + label_one_hot * (1 - self.delta)),
            dim=1,
        )

        # Loss

        loss = self.alpha * ce - self.beta * dce.mean()
        return loss


class RNNSeq2Seq(L.LightningModule):
    def __init__(
        self,
        dm,
        embedding_dim=512,
        hidden_dim=256,
        num_layers=2,
        seq_length=128,
        encoder_type="lstm",  # or "lstm" or "gru"
        decoder_type="lstm",  # or "lstm" or "gru"
        bidirectional=True,
        teacher_forcing_prob=0.9,
        teacher_forcing_schedule=None,
        loss="crossentropy",  # or intrust
    ):
        super().__init__()
        # set the hparams

        self.save_hyperparameters()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.train_epoch = 0
        self.dm: IWSLTSentimentDataModule = dm

        # RNN encoder
        self.en_encoder = Encoder(
            SimpleNamespace(
                **{
                    "vocab_size_encoder": self.dm.vocab_size,  # size of the vocabulary
                    "embed_size": embedding_dim,  # size of the word embeddings
                    "rnn_cell": encoder_type,  # in practice GRU or LSTM will always outperform RNN
                    "rnn_hidden_size": hidden_dim,  # size of the hidden state
                    "rnn_num_layers": num_layers,  # 1 or 2 layers are most common; more rarely sees any benefit
                    "rnn_dropout": 0.3,  # only relevant if rnn_num_layers > 1
                    "rnn_encoder_bidirectional": bidirectional,  # The encoder can be bidirectional; the decoder can not
                    "device": device,
                    "special_token_pad": self.dm.pad_id,
                    "seq_length": seq_length,
                }
            )
        ).to(device)
        self.zh_encoder = Encoder(
            SimpleNamespace(
                **{
                    "vocab_size_encoder": dm.vocab_size,
                    "embed_size": embedding_dim,
                    "rnn_cell": encoder_type,
                    "rnn_hidden_size": hidden_dim,
                    "rnn_num_layers": num_layers,
                    "rnn_dropout": 0.3,
                    "rnn_encoder_bidirectional": bidirectional,
                    "device": device,
                    "special_token_pad": self.dm.pad_id,
                    "seq_length": seq_length,
                }
            )
        ).to(device)

        criterion = (
            nn.CrossEntropyLoss()
            if loss == "crossentropy"
            else In_trust_Loss(num_classes=self.dm.vocab_size)
        )

        self.en_decoder = Decoder(
            SimpleNamespace(
                **{
                    "vocab_size_decoder": self.dm.vocab_size,
                    "embed_size": embedding_dim,
                    "rnn_cell": decoder_type,
                    "rnn_hidden_size": hidden_dim,
                    "rnn_num_layers": num_layers,
                    "rnn_dropout": 0.2,
                    "rnn_encoder_bidirectional": bidirectional,
                    "linear_dropout": 0.2,
                    "attention": "dot",  # "dot" if dot attention
                    "linear_hidden_sizes": [],  # list of sizes of subsequent hidden layers; can be [] (empty); only relevant for the decoder
                    "device": device,
                    "special_token_sos": self.dm.bos_id,
                    "special_token_eos": self.dm.eos_id,
                    "special_token_unk": self.dm.unk_id,
                    "special_token_pad": self.dm.pad_id,
                    "teacher_forcing_prob": teacher_forcing_prob,
                    "teacher_forcing_schedule": teacher_forcing_schedule,
                    "seq_length": seq_length,
                    "batch_size": 32,
                }
            ),
            criterion,
        ).to(device)

        self.zh_decoder = Decoder(
            SimpleNamespace(
                **{
                    "vocab_size_decoder": self.dm.vocab_size,
                    "embed_size": embedding_dim,
                    "rnn_cell": decoder_type,
                    "rnn_hidden_size": hidden_dim,
                    "rnn_num_layers": num_layers,
                    "rnn_dropout": 0.2,
                    "rnn_encoder_bidirectional": bidirectional,
                    "linear_dropout": 0.2,
                    "attention": "dot",
                    "linear_hidden_sizes": [],
                    "device": device,
                    "special_token_sos": self.dm.bos_id,
                    "special_token_eos": self.dm.eos_id,
                    "special_token_unk": self.dm.unk_id,
                    "special_token_pad": self.dm.pad_id,
                    "teacher_forcing_prob": teacher_forcing_prob,
                    "teacher_forcing_schedule": teacher_forcing_schedule,
                    "seq_length": seq_length,
                    "batch_size": 32,
                }
            ),
            criterion,
        )
        self.en_comet_score = 1
        self.zh_comet_score = 1

        # translation
        self.seq_length = seq_length

    def on_train_start(self) -> None:
        self.train_epoch = 0
        return super().on_train_start()

    def on_train_epoch_end(self) -> None:
        self.train_epoch += 1
        return super().on_train_epoch_end()

    def training_step(self, batch, batch_idx):
        en, zh = batch["en_tokens"], batch["zh_tokens"]

        en_outputs, en_hidden = self.en_encoder(en)
        zh_outputs, zh_hidden = self.zh_encoder(zh)
        en2zh_loss = self.zh_decoder(zh, en_hidden, en_outputs, self.train_epoch)
        zh2en_loss = self.en_decoder(en, zh_hidden, zh_outputs, self.train_epoch)
        loss = en2zh_loss + zh2en_loss
        self.log(
            "train_loss",
            loss,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=batch["en_tokens"].shape[0],
        )
        return loss

    def validation_step(self, batch, batch_idx):

        en, zh = batch["en_tokens"], batch["zh_tokens"]

        en_outputs, en_hidden = self.en_encoder(en)
        zh_outputs, zh_hidden = self.zh_encoder(zh)
        en2zh_loss = self.zh_decoder(zh, en_hidden, en_outputs, self.train_epoch)
        zh2en_loss = self.en_decoder(en, zh_hidden, zh_outputs, self.train_epoch)
        loss = en2zh_loss + zh2en_loss
        self.log(
            "val_loss",
            loss,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=batch["en_tokens"].shape[0],
        )
        if batch_idx == 0:
            en_indices, en_attentions = self.en_decoder.generate(
                zh_hidden, zh_outputs, self.seq_length
            )
            zh_indices, zh_attentions = self.zh_decoder.generate(
                en_hidden, en_outputs, self.seq_length
            )
            en_string = self.dm.decode_en(en_indices)
            zh_string = self.dm.decode_zh(zh_indices)
            en_bleurt_score = compute_bleurt_score(batch["en"], en_string)
            zh_bleurt_score = compute_bleurt_score(batch["zh"], zh_string)
            self.log(
                "val_en_bleurt_score",
                en_bleurt_score[0],
                on_step=True,
                prog_bar=True,
                logger=True,
                batch_size=batch["en_tokens"].shape[0],
            )
            self.log(
                "val_zh_bleurt_score",
                zh_bleurt_score[0],
                on_step=True,
                prog_bar=True,
                logger=True,
                batch_size=batch["en_tokens"].shape[0],
            )
            for i in range(min(5, len(batch["en"]))):
                print(f"sample en translation: {batch['en'][i]} -> [{en_string[i]}]")
                print(f"sample zh translation: {batch['zh'][i]} -> [{zh_string[i]}]")
        return loss

    def test_step(self, batch, batch_idx):
        global latest_batch, latest_batch_idx
        latest_batch = batch
        latest_batch_idx = batch_idx

        en, zh = batch["en_tokens"], batch["zh_tokens"]

        en_outputs, en_hidden = self.en_encoder(en)
        zh_outputs, zh_hidden = self.zh_encoder(zh)

        en_indices, en_attentions = self.en_decoder.generate(
            zh_hidden, zh_outputs, self.seq_length
        )
        zh_indices, zh_attentions = self.zh_decoder.generate(
            en_hidden, en_outputs, self.seq_length
        )

        # compute BLEURT score
        en_string = self.dm.decode_en(en_indices)
        zh_string = self.dm.decode_zh(zh_indices)

        zh_bleurt_score = compute_bleurt_score(batch["zh"], [zh_string])
        en_bleurt_score = compute_bleurt_score(batch["en"], [en_string])
        self.log(
            "zh_bleurt_score",
            zh_bleurt_score[0],
            on_step=True,
            prog_bar=True,
            logger=True,
            batch_size=batch["en_tokens"].shape[0],
        )
        self.log(
            "en_bleurt_score",
            en_bleurt_score[0],
            on_step=True,
            prog_bar=True,
            logger=True,
            batch_size=batch["en_tokens"].shape[0],
        )

        print(f"sample translation: {batch['en'][0]} -> {zh_string[0]}")

        return (zh_bleurt_score, en_bleurt_score)

    @classmethod
    def get_language(cls, text):
        if re.search("[\u4e00-\u9fff]", text):
            return "zh"
        return "en"

    def predict_text(self, text):
        language = self.get_language(text)
        if language == "zh":
            encode = self.dm.encode_zh
            decode = self.dm.decode_en
            encoder = self.zh_encoder.to(self.device)
            decoder = self.en_decoder.to(self.device)
        else:
            encode = self.dm.encode_en
            decode = self.dm.decode_zh
            encoder = self.en_encoder.to(self.device)
            decoder = self.zh_decoder.to(self.device)

        sentence = text
        tokens = encode(sentence)
        tokens = torch.tensor(tokens).unsqueeze(0).to(self.device)
        outputs, hidden = encoder(tokens)
        indices, attentions = decoder.generate(hidden, outputs, self.seq_length)
        string = decode(indices)
        return string

    def configure_optimizers(self, learning_rate=1e-3):
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        return optimizer


if __name__ == "__main__":
    # argparse
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--embedding_dim", type=int, default=512)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--seq_length", type=int, default=128)
    parser.add_argument("--encoder_type", type=str, default="lstm")
    parser.add_argument("--decoder_type", type=str, default="lstm")
    parser.add_argument("--bidirectional", type=bool, default=True)
    parser.add_argument("--teacher_forcing_prob", type=float, default=0.9)
    parser.add_argument("--teacher_forcing_schedule", type=str, default=None)
    parser.add_argument("--loss", type=str, default="intrust", help="crossentropy or intrust") # or intrust
    parser.add_argument("--vocab_size", type=int, default=8000)
    parser.add_argument("--train_batch_size", type=int, default=64)
    parser.add_argument("--checkpoint_prefix", type=str, default="rnn_seq2seq")
    parser.add_argument("--max_epochs", type=int, default=300)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    args = parser.parse_args()

    vocab_size = args.vocab_size
    embedding_dim = args.embedding_dim

    # create a model
    from data import IWSLTSentimentDataModule

    iwslt_dm = IWSLTSentimentDataModule(
        tokenizer="sentencepiece",
        use_google_zh=True,
        shared_tokenizer=False,
        vocab_size=vocab_size,
    )
    iwslt_dm.setup()  # Initialize datamodule, train tokenizer if necessary
    train_dm = iwslt_dm.train_dataloader(batch_size=args.train_batch_size)
    val_dm = iwslt_dm.val_dataloader(batch_size=128)
    test_dm = iwslt_dm.test_dataloader(batch_size=128)

    from math import cos

    exponential_schedule = lambda epoch: 0.1 + 0.9 * 0.9 ** (epoch / 2)
    # custom cyclic teacher forcing probability schedule
    cosine_annealing_schedule = (
        lambda epoch: 0.4 + 0.27 * (cos(epoch) + 1) * 0.98**epoch
    )
    model = RNNSeq2Seq(
        dm=iwslt_dm,
        teacher_forcing_prob=args.teacher_forcing_prob,
        teacher_forcing_schedule=None if args.teacher_forcing_schedule is None else cosine_annealing_schedule,
        loss=args.loss,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        seq_length=args.seq_length,
        encoder_type=args.encoder_type,
        decoder_type=args.decoder_type,
        bidirectional=args.bidirectional,
    ).cuda()

    model.configure_optimizers(learning_rate=args.learning_rate)

    os.makedirs("checkpoints", exist_ok=True) # create checkpoints folder
    os.makedirs("predictions", exist_ok=True) # create predictions folder
    cp_name = args.checkpoint_prefix

    # model.load_from_checkpoint("checkpoints/rnn_seq2seq-v2.ckpt").cuda()
    # create a trainer
    from lightning.pytorch.callbacks import GradientAccumulationScheduler

    trainer = L.Trainer(
        max_epochs=args.max_epochs,
        log_every_n_steps=5,
        precision=16,
        accelerator="cuda",
        gradient_clip_val=0.5,
        callbacks=[
            ModelCheckpoint(
                monitor="val_loss",
                save_top_k=3,
                mode="min",
                dirpath="checkpoints",
                filename=cp_name + "_{epoch}_{val_loss:.2f}",
                every_n_epochs=1,
                auto_insert_metric_name=False,
            ),
            EarlyStopping(monitor="val_loss", patience=15, mode="min"),
            GradientAccumulationScheduler(scheduling={0: 8, 4: 4, 8: 1}),
        ],
    )

    # train the model
    trainer.fit(model, train_dm, val_dm)
    # trainer.fit(model, train_dm, val_dm, ckpt_path="checkpoints/lstmlstm2layer_google_8000_512_256_32len_costf_itl-last.ckpt")

    # save the model
    trainer.save_checkpoint(f"checkpoints/{cp_name}-last.ckpt")
    checkpoint = torch.load(f"checkpoints/{cp_name}-last.ckpt")
    model.load_from_checkpoint(f"checkpoints/{cp_name}-last.ckpt").cuda()
    model.load_state_dict(checkpoint["state_dict"])
    # evaluate
    result = trainer.test(model, test_dm)
    model.cuda()
    model.eval()

    preds = []
    from tqdm import tqdm

    for batch in tqdm(iter(test_dm)):
        for i in zip(batch["en"], batch["zh"]):
            en_pred = model.predict_text(i[0])
            zh_pred = model.predict_text(i[1])
            preds.append((i[0], i[1], en_pred[0], zh_pred[0]))
            print(f"en: {i[0]} -> {en_pred}")

    import pandas as pd
    pd.DataFrame(preds, columns=["en", "zh", "en_pred", "zh_pred"]).to_csv(
        f"predictions/predictions_{cp_name}.csv"
    )
