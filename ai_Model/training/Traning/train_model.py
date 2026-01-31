# trainer.py
import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from ai_Model.chat_dataset.chatDataset import ChatDataset
from ai_Model.utils.mask_utils import generate_square_subsequent_mask
from ai_Model.tokenizer.SimpleTokenizer import StackedTransformer
from ai_Model.utils.logger_setup import get_logger

logger = get_logger("trainer")


class Trainer:
    def __init__(self, model, tokenizer, device=None, embed_size=256, tokenizer_path="bpe_tokenizer.json"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embed_size = embed_size
        self.tokenizer_path = tokenizer_path

    def load_config(self, path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Could not load config: {e}")
            return {}

    def collate_fn(self, batch):
        inputs, targets = zip(*batch)
        max_len = max(max(len(seq) for seq in inputs), max(len(seq) for seq in targets))
        def pad(seqs):
            return torch.stack([torch.cat([seq, torch.zeros(max_len - len(seq), dtype=torch.long)]) for seq in seqs])
        return pad(inputs), pad(targets)

    def save_checkpoint(self, epoch, optimizer, filename="checkpoint_latest.pth"):
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "tokenizer": self.tokenizer.word2idx
        }
        torch.save(checkpoint, filename)

    def load_checkpoint(self, filename, optimizer):
        checkpoint = torch.load(filename, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        return checkpoint.get("epoch", 0) + 1

    def train(self, config_path="ai_Model/config.json"):
        config = self.load_config(config_path)
        batch_size = config.get("batch_size", 32)
        epochs = config.get("epochs", 10)
        lr = config.get("learning_rate", 0.001)
        data_paths = config.get("train_data_paths", [])

        # 🔹 Läs träningsdata
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
            logger.error("No training data found. Aborting training.")
            return

        # 🔹 Bygg vokabulär
        texts = [x for pair in training_data for x in pair]
        self.tokenizer.train(texts)
        self.tokenizer.save_vocab(self.tokenizer_path)
        vocab_size = len(self.tokenizer.word2idx)
        logger.info(f"Tokenizer trained/updated. Vocab size: {vocab_size}")

        # 🔹 Dataset och DataLoader
        dataset = ChatDataset(training_data, self.tokenizer)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=self.collate_fn)

        # 🔹 Optimizer och loss
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss(ignore_index=0)

        # 🔹 Checkpoint
        checkpoint_path = "checkpoint_latest.pth"
        start_epoch = 0
        if os.path.exists(checkpoint_path):
            start_epoch = self.load_checkpoint(checkpoint_path, optimizer)
            logger.info(f"Resuming training from epoch {start_epoch}")

        # 🔹 Träningsloop med early stopping
        best_loss = float("inf")
        patience_counter = 0
        early_stopping_patience = config.get("early_stopping_patience", 3)

        for epoch in range(start_epoch, epochs):
            self.model.train()
            total_loss = 0
            for x, y in dataloader:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                mask = generate_square_subsequent_mask(x.size(1)).to(self.device)
                out = self.model(x, mask)
                loss = criterion(out.view(-1, out.size(-1)), y.view(-1))
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(dataloader)
            logger.info(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")
            self.save_checkpoint(epoch+1, optimizer, checkpoint_path)

            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    logger.info(f"No improvement for {early_stopping_patience} epochs. Early stopping at epoch {epoch+1}.")
                    break

        self.model.eval()
        torch.save(self.model.state_dict(), config.get("model_path", "aether_model.pth"))
        logger.info("Training complete. Model saved successfully.")
