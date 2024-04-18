"""
This module contains the data processing pipeline for the IWSLT 2017 dataset.

"""
import torch.utils.data
import torch.nn.functional as F
import sentencepiece as spm
from datasets import load_dataset
import os, io
from transformers import AutoTokenizer
import pytorch_lightning as pl
import sentencepiece as spm


class DatasetIterator:
    def __init__(self, dataset, field="en"):
        self.dataset = dataset
        self.field = field
        self.index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.field == "en_zh":
            if self.index < len(self.dataset) * 2:
                result = self.dataset[self.index // 2]
                language = "zh" if self.index % 2 == 0 else "en"
                self.index += 1
                return result[language]
            else:
                raise StopIteration
        else:
            if self.index < len(self.dataset):
                result = self.dataset[self.index]
                self.index += 1
                if self.field == "zh":
                    return result["zh"] + ("" if result["google_zh"] is None else result["google_zh"])
                else:
                    return result[self.field]
            else:
                raise StopIteration


class IWSLTSentimentDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=16, retrain=False, tokenizer="sentencepiece",
                 use_google_zh=False, vocab_size=16000,
                 shared_tokenizer=False, # use en_zh_sentencepiece if true
                 dataset="ngxingyu/iwslt17_google_trans_scores_sentiments",
                 model_type="unigram",
                 device=None):
        """
        Args:
            batch_size: batch size for dataloader
            retrain: retrain sentencepiece model
            tokenizer: sentencepiece or xlm-roberta
        """
        super().__init__()
        self.retrain=retrain
        self.batch_size = batch_size
        self.device = device
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = tokenizer
        # tokenizers will be loaded in setup
        self.en_tokenizer = None
        self.zh_tokenizer = None
        self.seq_len = 128
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.vocab_size = vocab_size
        self.shared_tokenizer = shared_tokenizer
        self.dataset = dataset
        self.model_type = model_type

        self.use_google_zh = use_google_zh

    @staticmethod
    def get_sentencepiece_processor(language="en", vocab_size=16000, model_type="unigram") -> spm.SentencePieceProcessor:
        if not os.path.isfile(f"{language}_sentencepiece.model.{vocab_size}.{model_type}"):
            raise FileNotFoundError(f"{language}_sentencepiece.model.{vocab_size}.{model_type} not found")
        processor = spm.SentencePieceProcessor(
            model_file=f"{language}_sentencepiece.model.{vocab_size}.{model_type}"
        )
        processor.SetEncodeExtraOptions("bos:eos")

        return processor

    @property
    def num_tokens(self):
        if self.tokenizer == "sentencepiece":
            return self.en_tokenizer.get_piece_size()
        return self.en_tokenizer.vocab_size
    
    @property
    def bos_id(self):
        if self.tokenizer == "sentencepiece":
            return self.en_tokenizer.bos_id()
        return self.en_tokenizer.bos_token_id
    
    @property
    def eos_id(self):
        if self.tokenizer == "sentencepiece":
            return self.en_tokenizer.eos_id()
        return self.en_tokenizer.eos_token_id
    
    @property
    def pad_id(self):
        if self.tokenizer == "sentencepiece":
            return self.en_tokenizer.pad_id()
        return self.en_tokenizer.pad_token_id
    
    @property
    def unk_id(self):
        if self.tokenizer == "sentencepiece":
            return self.en_tokenizer.unk_id()
        return self.en_tokenizer.unk_token_id

    def setup(self):
        self.train_dataset = load_dataset(
            self.dataset,
            split="train",
        ).with_format("torch")
        if self.tokenizer == "sentencepiece":
            if not os.path.isfile(f"en_sentencepiece.model.{self.vocab_size}.{self.model_type}") or self.retrain:
                self.train_dataset = load_dataset(
                    self.dataset,
                    split="train",
                ).with_format("torch")
                model = io.BytesIO()

                train_en_iter = DatasetIterator(self.train_dataset, field="en")
                print("Training sentencepiece model for English")
                spm.SentencePieceTrainer.train(
                    sentence_iterator=train_en_iter, model_writer=model, vocab_size=self.vocab_size,
                    model_type=self.model_type,
                    split_digits=True,
                    hard_vocab_limit=False,
                    pad_id=3
                )
                with open(f"en_sentencepiece.model.{self.vocab_size}.{self.model_type}", "wb") as f:
                    f.write(model.getvalue())
            if not os.path.isfile(f"zh_sentencepiece.model.{self.vocab_size}.{self.model_type}") or self.retrain:
                train_zh_iter = DatasetIterator(self.train_dataset, field="zh")
                model = io.BytesIO()
                print("Training sentencepiece model for Chinese")
                spm.SentencePieceTrainer.train(
                    sentence_iterator=train_zh_iter, model_writer=model, vocab_size=self.vocab_size,
                    model_type=self.model_type,
                    split_digits=True,
                    hard_vocab_limit=False,
                    pad_id=3
                )
                with open(f"zh_sentencepiece.model.{self.vocab_size}.{self.model_type}", "wb") as f:
                    f.write(model.getvalue())
            if not os.path.isfile(f"zh_sentencepiece.model.{self.vocab_size}.{self.model_type}") or self.retrain:
                train_zh_iter = DatasetIterator(self.train_dataset, field="zh")
                model = io.BytesIO()
                print("Training sentencepiece model for Chinese")
                spm.SentencePieceTrainer.train(
                    sentence_iterator=train_zh_iter, model_writer=model, vocab_size=self.vocab_size,
                    model_type=self.model_type,
                    split_digits=True,
                    hard_vocab_limit=False,
                    pad_id=3
                )
                with open(f"zh_sentencepiece.model.{self.vocab_size}.{self.model_type}", "wb") as f:
                    f.write(model.getvalue())
            if not os.path.isfile(f"en_zh_sentencepiece.model.{self.vocab_size}.{self.model_type}") or self.retrain:
                train_en_zh_iter = DatasetIterator(self.train_dataset, field="en_zh")
                model = io.BytesIO()
                print("Training shared sentencepiece model for English+Chinese")
                spm.SentencePieceTrainer.train(
                    sentence_iterator=train_en_zh_iter, model_writer=model, vocab_size=self.vocab_size,
                    model_type=self.model_type,
                    split_digits=True,
                    hard_vocab_limit=False,
                    pad_id=3
                )
                with open(f"en_zh_sentencepiece.model.{self.vocab_size}.{self.model_type}", "wb") as f:
                    f.write(model.getvalue())
            self.en_tokenizer = self.get_sentencepiece_processor("en", self.vocab_size, self.model_type)
            self.zh_tokenizer = self.get_sentencepiece_processor("zh", self.vocab_size, self.model_type)
            self.en_zh_tokenizer = self.get_sentencepiece_processor("en_zh", self.vocab_size, self.model_type)
            self.padding = [self.en_tokenizer.pad_id()] * self.seq_len
        else:
            self.en_tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual")
            self.zh_tokenizer = self.en_tokenizer
            self.en_zh_tokenizer = self.en_tokenizer
            self.padding = [self.en_tokenizer.pad_token_id] * self.seq_len

    def pad_sequence(self, sequence):
        if isinstance(sequence, torch.Tensor):
            return F.pad(sequence, (0, self.seq_len - sequence.size(1)), value=self.en_tokenizer.pad_token_id)
        else:
            return (sequence + self.padding)[: self.seq_len]

    def process_batch(self, split="train"):
        def _process_batch(batch):
            en_inputs = batch["en"]
            zh_inputs = batch["zh"] if (not self.use_google_zh or split=="test") else batch["google_zh"]
            zh_inputs = [x if x is not None else "" for x in zh_inputs]
            en_sentiments = batch["en_sentiment"]
            zh_sentiments = batch["zh_sentiment"]
            if self.tokenizer == "sentencepiece":
                en_encoded_inputs = torch.tensor([self.pad_sequence(x) for x in self.en_tokenizer.tokenize(
                    en_inputs
                )]).to(self.device)
                zh_encoded_inputs = torch.tensor([self.pad_sequence(x) for x in self.zh_tokenizer.tokenize(
                    zh_inputs
                )]).to(self.device)
            else:
                en_encoded_inputs = self.pad_sequence(self.en_tokenizer(
                    en_inputs, return_tensors="pt", padding=True
                ).to(self.device)['input_ids'])
                zh_encoded_inputs = self.pad_sequence(self.zh_tokenizer(
                    zh_inputs, return_tensors="pt", padding=True
                ).to(self.device)['input_ids'])
            en_sentiments.to(self.device)
            zh_sentiments.to(self.device)
            return {
                "en_tokens": en_encoded_inputs,
                "zh_tokens": zh_encoded_inputs,
                "en_sentiments": en_sentiments,
                "zh_sentiments": zh_sentiments
            }
        return _process_batch

    
    def collate_fn(self, batch):
        if not isinstance(batch, list):
            raise ValueError("batch must be a list")
    
        result = {
            "en": [],
            "zh": [],
            "en_tokens": torch.empty((0, self.seq_len), dtype=torch.int64),
            "zh_tokens": torch.empty((0, self.seq_len), dtype=torch.int64),
            "en_sentiment": torch.empty((0, 3), dtype=torch.float32),
            "zh_sentiment": torch.empty((0, 3), dtype=torch.float32),
        }
        for i in range(len(batch)):
            if batch is None or len(batch[i]["en_tokens"]) == 0:
                continue
            result["en_tokens"] = torch.cat((result["en_tokens"],
                                                batch[i]["en_tokens"].reshape(-1, self.seq_len)))
            result["zh_tokens"] = torch.cat((result["zh_tokens"],
                                                batch[i]["zh_tokens"].reshape(-1, self.seq_len)))
            result["en_sentiment"] = torch.cat((result["en_sentiment"],
                                                batch[i]["en_sentiments"].reshape(-1, 3)))
            result["zh_sentiment"] = torch.cat((result["zh_sentiment"],
                                                batch[i]["zh_sentiments"].reshape(-1, 3)))
            result["en"].extend([batch[i]["en"]] if isinstance(batch[i]["en"], str) else batch[i]["en"])
            result["zh"].extend([batch[i]["zh"]] if isinstance(batch[i]["zh"], str) else batch[i]["zh"])
        return result


    def train_dataloader(self, batch_size=32):
        if self.train_dataset is None:
            self.setup()
            self.train_dataset = load_dataset(
                self.dataset,
                streaming=False,
                split="train",
            ).with_format("torch")
        self.train_dataset = self.train_dataset.map(
            self.process_batch("train"),
            batched=True,
            batch_size=32,
        )
        return torch.utils.data.DataLoader(
            self.train_dataset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=self.collate_fn
        )

    def val_dataloader(self, batch_size=32, shuffle=False):
        if self.val_dataset is None:
            self.setup()
            self.val_dataset = load_dataset(
                self.dataset,
                streaming=False,
                split="validation",
            ).with_format("torch")
        self.val_dataset = self.val_dataset.map(
            self.process_batch("val"),
            batched=True,
            batch_size=32)
        return torch.utils.data.DataLoader(
            self.val_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4, collate_fn=self.collate_fn
        )

    def test_dataloader(self, batch_size=32, shuffle=False):
        if self.test_dataset is None:
            self.setup()
            self.test_dataset = load_dataset(
                self.dataset,
                split="test",
            ).with_format("torch")
        self.test_dataset = self.test_dataset.map(
            self.process_batch("test"),
            batched=True,
            batch_size=32)
        return torch.utils.data.DataLoader(
            self.test_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4, collate_fn=self.collate_fn
        )

    def encode_en(self, text):
        if self.tokenizer == "sentencepiece":
            if self.shared_tokenizer:
                return self.en_zh_tokenizer.encode(text)
            return self.en_tokenizer.encode(text)
        return self.en_tokenizer.batch_encode(text, return_tensors="pt")

    def decode_en(self, tokens):
        if self.tokenizer == "sentencepiece":
            if self.shared_tokenizer:
                return self.en_zh_tokenizer.decode(tokens)
            return self.en_tokenizer.decode(tokens)
        
        else:
            if isinstance(tokens, list):
                return self.en_tokenizer.batch_decode(tokens, skip_special_tokens=True)
            else:
                return self.en_tokenizer.decode(tokens, skip_special_tokens=True)

    def encode_zh(self, text):
        if self.tokenizer == "sentencepiece":
            if self.shared_tokenizer:
                return self.en_zh_tokenizer.encode(text)
            return self.zh_tokenizer.encode(text)
        return self.zh_tokenizer.encode(text, return_tensors="pt")

    def decode_zh(self, tokens):
        if self.tokenizer == "sentencepiece":
            if self.shared_tokenizer:
                return self.en_zh_tokenizer.decode(tokens)
            return self.zh_tokenizer.decode(tokens)
        else:
            if isinstance(tokens, list):
                return self.zh_tokenizer.batch_decode(tokens, skip_special_tokens=True)
            else:
                return self.zh_tokenizer.decode(tokens, skip_special_tokens=True)


if __name__ == "__main__":

    """%% https://github.com/google/sentencepiece/blob/master/python/README.md"""

    idm = IWSLTSentimentDataModule(
        batch_size=32,
        tokenizer="sentencepiece", # currently only sentencepiece is supported
        use_google_zh=True,
        vocab_size=8000,
        model_type="unigram", # unigram, bpe
    )
    idm.setup()
    train_dl = idm.train_dataloader()
    print("Number of tokens in vocabulary: ", idm.num_tokens)
    print("BOS ID: ", idm.bos_id)
    print("EOS ID: ", idm.eos_id)
    print("PAD ID: ", idm.pad_id)
    print("UNK ID: ", idm.unk_id)
    idm.decode_en([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    print("First batch: ", next(iter(train_dl)))
