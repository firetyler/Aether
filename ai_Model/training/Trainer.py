import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ai_Model.chat_dataset.chatDataset import ChatDataset
from ai_Model.utils.mask_utils import generate_square_subsequent_mask
from ai_Model.utils.logger_setup import get_logger

logger = get_logger("trainer")


class Trainer:
    def __init__(self, model, tokenizer, device=None):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_config(self, path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def collate_fn(self, batch):
        inputs, targets = zip(*batch)

        max_len = max(max(len(i) for i in inputs), max(len(t) for t in targets))

        def pad(seqs):
            return torch.stack([
                torch.cat([seq, torch.zeros(max_len - len(seq), dtype=torch.long)])
                for seq in seqs
            ])

        return pad(inputs), pad(targets)

    def train(self, config_path="ai_Model/config.json"):
        config = self.load_config(config_path)

        batch_size = config.get("batch_size", 32)
        epochs = config.get("epochs", 10)
        lr = config.get("learning_rate", 0.001)
        data_paths = config.get("train_data_paths", [])

        # ---------------- DATA ----------------
        training_data = []

        for path in data_paths:
            if not os.path.exists(path):
                logger.warning(f"Missing file: {path}")
                continue

            with open(path, "r", encoding="utf-8") as f:
                raw = json.load(f)
                records = raw if isinstance(raw, list) else raw.get("data", [])

                training_data.extend(
                    [(d["input"], d["output"]) for d in records if "input" in d and "output" in d]
                )

        if not training_data:
            logger.error("No training data found")
            return

        # ---------------- TOKENIZER ----------------
        texts = [x for pair in training_data for x in pair]
        self.tokenizer.train(texts)

        vocab_size = len(self.tokenizer.word2idx)
        logger.info(f"Tokenizer trained. Vocab size: {vocab_size}")

        # ---------------- DATASET ----------------
        dataset = ChatDataset(training_data, self.tokenizer)

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=self.collate_fn
        )

        # ---------------- MODEL ----------------
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss(ignore_index=0)

        # ---------------- TRAIN ----------------
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0

            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)

                optimizer.zero_grad()

                mask = generate_square_subsequent_mask(x.size(1)).to(self.device)

                out = self.model(x, mask)

                loss = criterion(out.view(-1, out.size(-1)), y.view(-1))

                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            logger.info(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(loader):.4f}")

        torch.save(self.model.state_dict(), config.get("model_path", "aether_model.pth"))
        logger.info("Training complete")