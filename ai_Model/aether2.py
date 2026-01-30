import os
import re
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from ai_Model.tokenizer.Tokenizer import BpeTokenizer
from ai_Model.tokenizer.SimpleTokenizer import AdvancedSentenceTransformer, StackedTransformer
from ai_Model.momory.AetherMemory import AetherMemory
from ai_Model.utils.mask_utils import generate_square_subsequent_mask
from ai_Model.database.DatabaseConnector import DatabaseConnector
from ai_Model.utils.logger_setup import get_logger
from ai_Model.codeEx.code_executor import CodeExecutor

logger = get_logger("aether2")


# ===== CHAT DATASET =====
class ChatDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.pad_id = tokenizer.tokenizer.token_to_id("<PAD>")

    def __len__(self):
        return len(self.data)

    def pad(self, ids):
        ids = ids[:self.max_len]
        return ids + [self.pad_id] * (self.max_len - len(ids))

    def __getitem__(self, idx):
        inp, out = self.data[idx]
        x = self.pad(self.tokenizer.encode(inp))
        y = self.pad(self.tokenizer.encode(out))
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


# ===== AETHER AGENT =====
class AetherAgent:
    def __init__(self, db_connector):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.db_connector = db_connector
        self.name = "Aether"
        self.embed_size = 256

        # ✅ BPE tokenizer
        self.tokenizer = BpeTokenizer("bpe_tokenizer.json")
        if self.tokenizer.tokenizer is None:
            raise RuntimeError(
                "BPE-tokenizer är inte tränad. Kör tokenizer-träningen separat först."
            )

        self.vocab_size = self.tokenizer.tokenizer.get_vocab_size()
        logger.info(f"BPE vocab size: {self.vocab_size}")

        self.token_embedding = nn.Embedding(
            self.vocab_size,
            self.embed_size,
            padding_idx=self.tokenizer.tokenizer.token_to_id("<PAD>")
        )

        self.embed_model = AdvancedSentenceTransformer(
            vocab_size=self.vocab_size
        ).to(self.device)

        self.memory = AetherMemory(self.embed_model, self.tokenizer, self.device)
        self.model = None
        self.code_executor = CodeExecutor()
        self.ChatDataset = ChatDataset

    # ===== TEXT GENERATION =====
    def generate_text(self, prompt):
        if self.model is None:
            return json.dumps({"error": "Model not initialized"})

        tokens = self.tokenizer.encode(prompt)
        input_ids = torch.tensor([tokens], dtype=torch.long).to(self.device)
        mask = generate_square_subsequent_mask(input_ids.size(1)).to(self.device)

        with torch.no_grad():
            out = self.model(input_ids, mask)

        pred_ids = out.argmax(dim=-1).squeeze().tolist()
        text = self.tokenizer.decode(pred_ids)

        return json.dumps({"response": text})

    # ===== TRAINING =====
    def train_model(self, config_path="ai_Model/config.json"):
        config = self.load_config(config_path)
        batch_size = config.get("batch_size", 32)
        epochs = config.get("epochs", 10)
        lr = config.get("learning_rate", 0.001)

        # Läs träningsdata
        training_data = []
        filenames = config.get("train_data_paths", [])
        for filename in filenames:
            if os.path.exists(filename):
                with open(filename, encoding="utf-8") as f:
                    data = json.load(f)
                    valid_data = [(d["input"], d["output"]) for d in data if "input" in d and "output" in d]
                    training_data.extend(valid_data)

        if not training_data:
            logger.error("No valid training data found. Aborting training.")
            return

        dataset = self.ChatDataset(training_data, self.tokenizer)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

        # Initiera modell
        self.model = StackedTransformer(
            embed_size=config.get("embedding_dim", 256),
            vocab_size=self.vocab_size,
            num_layers=config.get("num_layers", 4),
            heads=config.get("heads", 8),
            forward_expansion=config.get("forward_expansion", 4),
            dropout=config.get("dropout", 0.1)
        ).to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        self.model.train()

        for epoch in range(epochs):
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

            logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")

    # ===== CONFIG & MODEL HELPERS =====
    def load_config(self, path):
        try:
            with open(path, "r") as f:
                return json.load(f)
        except:
            logger.warning(f"Could not read config at {path}")
            return {}

    def save_model(self, filename="aether_model.pth"):
        if self.model is not None:
            torch.save(self.model.state_dict(), filename)
            logger.info(f"Model saved to {filename}")

    def load_model(self, filename="aether_model.pth", config_path="ai_Model/config.json"):
        config = self.load_config(config_path)
        self.model = StackedTransformer(
            embed_size=config.get("embedding_dim", 256),
            vocab_size=self.vocab_size,
            num_layers=config.get("num_layers", 4),
            heads=config.get("heads", 8),
            forward_expansion=config.get("forward_expansion", 4),
            dropout=config.get("dropout", 0.1)
        ).to(self.device)

        if os.path.exists(filename):
            state_dict = torch.load(filename, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.eval()
            logger.info(f"Model loaded from {filename}")
        else:
            logger.warning(f"Model file {filename} not found, please train first.")

    # ===== RUN USER INPUT =====
    def run(self, user_input):
        lowered = user_input.lower().strip()
        if lowered.startswith("run "):
            parts = user_input.split(" ", 2)
            if len(parts) < 3:
                return "Use: run <language> <code>"
            language, code = parts[1], parts[2]
            result = self.code_executor.run_code(code, language)
            return f"Result ({language}): {result}"

        if lowered == "train model":
            self.train_model()
            self.save_model()
            return "Model trained and saved."

        return self.generate_text(user_input)


# ===== MAIN =====
if __name__ == "__main__":
    db = DatabaseConnector()
    agent = AetherAgent(db)
    agent.load_model(filename="aether_model.pth")

    print("Aether ready. Type 'exit' to quit.")
    while True:
        inp = input("> ")
        if inp.lower() in ["exit", "quit"]:
            break
        out = agent.run(inp)
        print(f"Aether: {out}")
