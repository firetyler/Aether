
import os
from tokenizers import Tokenizer, models, trainers, pre_tokenizers

class BpeTokenizer:
    def __init__(self, vocab_file="bpe_tokenizer.json"):
        self.vocab_file = vocab_file
        if os.path.exists(vocab_file):
            self.load_vocab(vocab_file)
        else:
            self.tokenizer = None

    def train(self, texts, vocab_size=16000):
        self.tokenizer = Tokenizer(models.BPE(unk_token="<UNK>"))
        self.tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
        trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=["<PAD>", "<UNK>", "<BOS>", "<EOS>"])

        # Spara texter till temporär fil och träna
        with open("_tmp_bpe_data.txt", "w", encoding="utf-8") as f:
            for line in texts:
                f.write(line + "\n")

        self.tokenizer.train(["_tmp_bpe_data.txt"], trainer)
        os.remove("_tmp_bpe_data.txt")
        self.save_vocab(self.vocab_file)

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
        return self.tokenizer.get_vocab()