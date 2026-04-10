
import os
from tokenizers import Tokenizer, models, trainers, pre_tokenizers


class BpeTokenizer:
    def __init__(self, vocab_file="bpe_tokenizer.json"):
        self.vocab_file = vocab_file

        if os.path.exists(vocab_file):
            self.tokenizer = Tokenizer.from_file(vocab_file)
        else:
            self.tokenizer = None

    @property
    def vocab_size(self):
        if self.tokenizer is None:
            return 16000
        return len(self.tokenizer.get_vocab())

    def train(self, texts, vocab_size=16000):
        self.tokenizer = Tokenizer(models.BPE(unk_token="<UNK>"))
        self.tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()

        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=["<PAD>", "<UNK>", "<BOS>", "<EOS>"]
        )

        tmp_file = "_tmp_bpe_data.txt"

        with open(tmp_file, "w", encoding="utf-8") as f:
            for line in texts:
                f.write(line + "\n")

        # ✅ FIX: korrekt HuggingFace API
        self.tokenizer.train(files=[tmp_file], trainer=trainer)

        os.remove(tmp_file)
        self.save_vocab()

    def encode(self, text):
        return self.tokenizer.encode(text).ids

    def decode(self, ids):
        return self.tokenizer.decode(ids)

    def save_vocab(self, path=None):
        self.tokenizer.save(path or self.vocab_file)

    def load_vocab(self, path=None):
        self.tokenizer = Tokenizer.from_file(path or self.vocab_file)

    @property
    def word2idx(self):
        return self.tokenizer.get_vocab() if self.tokenizer else {}