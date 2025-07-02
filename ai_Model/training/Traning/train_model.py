# trainer.py
import sys
import os
import json
import torch
import torch.nn as nn
from ai_Model.tokenizer.SimpleTokenizer import StackedTransformer, SimpleTokenizer, AdvancedSentenceTransformer
#from SimpleTokenizer import StackedTransformer, SimpleTokenizer, AdvancedSentenceTransformer
from ai_Model.chat_dataset import ChatDataset
#from chatDataset import ChatDataset
from ai_Model.utils.logTrainer import get_logger
#from logTrainer import get_logger
from ai_Model.utils.mask_utils import generate_square_subsequent_mask
#from mask_utils import generate_square_subsequent_mask
from torch.utils.data import DataLoader

sys.stdout.reconfigure(encoding='utf-8')
logger = get_logger("training")

class Trainer:
    def __init__(self, tokenizer=None, device=None, embed_size=256):
        self.tokenizer = tokenizer or SimpleTokenizer()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embed_size = embed_size
        self.model = None

    def load_config(self, path):
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Could not read the config file: {e}")
            return {}

    def save_checkpoint(self, epoch, optimizer, filename):
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "tokenizer": self.tokenizer.word2idx
        }
        torch.save(checkpoint, filename)

    def load_checkpoint(self, filename, optimizer, config):
        if not os.path.isfile(filename):
            logger.warning(f"No checkpoint file found: {filename}")
            return 0

        checkpoint = torch.load(filename, map_location=self.device)

        self.tokenizer.load_vocab(config.get("tokenizer_vocab_path", "tokenizer_vocab.json"))
        vocab_size = self.tokenizer.vocab_size

        self.model = StackedTransformer(
            vocab_size=vocab_size,
            embed_size=self.embed_size,
            num_layers=config.get("num_layers", 6),
            num_heads=config.get("heads", 8),
            forward_expansion=config.get("forward_expansion", 4),
            dropout=config.get("dropout", 0.1),
            pooling=config.get("pooling", "cls")
        ).to(self.device)

        model_dict = self.model.state_dict()
        pretrained_dict = {k: v for k, v in checkpoint["model_state_dict"].items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.model.load_state_dict(model_dict)

        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        return checkpoint.get("epoch", 0) + 1

    def collate_fn(self, batch):
        inputs, targets = zip(*batch)
        max_len = max(max(len(seq) for seq in inputs), max(len(seq) for seq in targets))

        def pad(seqs):
            return torch.stack([
                torch.cat([seq, torch.zeros(max_len - len(seq), dtype=torch.long)])
                for seq in seqs
            ])

        return pad(inputs), pad(targets)

    def train_model(self, config_path="ai_Model/config.json"):
        config = self.load_config(config_path)
        epochs = config.get("epochs", 10)
        lr = config.get("learning_rate", 0.001)
        batch_size = config.get("batch_size", 32)
        data_paths = config.get("train_data_paths", [])

        training_data = []
        for path in data_paths:
            if not os.path.exists(path):
                logger.warning(f"Missing training file: {path}")
                continue
            with open(path, "r", encoding="utf-8") as f:
                raw = json.load(f)
                records = raw if isinstance(raw, list) else raw.get("data", [])
                pairs = [(d["input"], d["output"]) for d in records if "input" in d and "output" in d]
                training_data.extend(pairs)

        if not training_data:
            logger.error("No training data found.")
            return

        texts = [x for pair in training_data for x in pair]
        self.tokenizer.build_vocab(texts)
        self.tokenizer.save_vocab("tokenizer_vocab.json")

        dataset = ChatDataset(training_data, self.tokenizer)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=self.collate_fn)

        vocab_size = len(self.tokenizer.word2idx)
        self.model = StackedTransformer(
            vocab_size=vocab_size,
            embed_size=config.get("embedding_dim", 256),
            num_layers=config.get("num_layers", 6),
            num_heads=config.get("heads", 8),
            forward_expansion=config.get("forward_expansion", 4),
            dropout=config.get("dropout", 0.1),
            pooling=config.get("pooling", "cls")
        ).to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss(ignore_index=0)

        checkpoint_path = "checkpoint_latest.pth"
        start_epoch = self.load_checkpoint(checkpoint_path, optimizer, config) if os.path.exists(checkpoint_path) else 0

        for epoch in range(start_epoch, epochs):
            self.model.train()
            total_loss = 0
            for x, y in dataloader:
                x, y = x.to(self.device), y.to(self.device)

                optimizer.zero_grad()
                out = self.model(x)

                loss = criterion(out.view(-1, out.size(-1)), y.view(-1))
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(dataloader)
            logger.info(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}")
            self.save_checkpoint(epoch + 1, optimizer, checkpoint_path)

        self.model.eval()
        torch.save(self.model.state_dict(), config.get("model_path", "aether_model.pth"))
        logger.info("Training complete and model saved.")
