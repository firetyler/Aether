import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from ai_Model.tokenizer.Tokenizer import BpeTokenizer
from ai_Model.tokenizer.SimpleTokenizer import StackedTransformer
from ai_Model.utils.mask_utils import generate_square_subsequent_mask
from ai_Model.momory.AetherMemory import AetherMemory
from ai_Model.database.DatabaseConnector import DatabaseConnector
from ai_Model.codeEx.code_executor import CodeExecutor
from ai_Model.utils.logger_setup import get_logger
from ai_Model.training.Traning.train_model import Trainer 

logger = get_logger("aether2")

class AetherAgent:
    def __init__(self, db_connector, tokenizer_path="bpe_tokenizer.json", config_path="ai_Model/config.json"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.db_connector = db_connector
        self.name = "Aether"
        self.embed_size = 256
        self.tokenizer_path = tokenizer_path
        self.config_path = config_path
        self.code_executor = CodeExecutor()
       
        self.model = None
        self.token_embedding = None
        self.memory = None

        # 🔹 Initiera tokenizer
        self._init_tokenizer()

        # 🔹 Initiera modell och embedding
        self._init_model()

        self.trainer = Trainer(model=self.model, tokenizer=self.tokenizer, device=self.device)
        
        # 🔹 Initiera minne
        self.memory = AetherMemory(self.model, self.tokenizer, self.device)

    # =====================
    # Tokenizer-initiering
    # =====================
    def _init_tokenizer(self):
        if os.path.exists(self.tokenizer_path):
            self.tokenizer = BpeTokenizer(self.tokenizer_path)
            logger.info(f"Loaded BPE tokenizer from {self.tokenizer_path}, vocab size: {len(self.tokenizer.word2idx)}")
        else:
            logger.warning(f"BPE tokenizer file '{self.tokenizer_path}' not found. Training tokenizer...")
            self.tokenizer = BpeTokenizer()
            self._train_tokenizer_from_training_data()
            logger.info(f"Tokenizer trained and saved to {self.tokenizer_path}")

    def _train_tokenizer_from_training_data(self):
        config = self.load_config(self.config_path)
        data_paths = config.get("train_data_paths", [])
        training_texts = []

        for path in data_paths:
            if not os.path.exists(path):
                logger.warning(f"Training file missing: {path}")
                continue
            with open(path, "r", encoding="utf-8") as f:
                raw = json.load(f)
                records = raw if isinstance(raw, list) else raw.get("data", [])
                for rec in records:
                    if "input" in rec and "output" in rec:
                        training_texts.append(rec["input"])
                        training_texts.append(rec["output"])

        if not training_texts:
            logger.error("No training data found for tokenizer. Using fallback words.")
            training_texts = ["hello", "world", "question", "answer", "sweden"]

        self.tokenizer.train(training_texts)
        self.tokenizer.save_vocab(self.tokenizer_path)

    # =====================
    # Modell-initiering
    # =====================
    def _init_model(self):
        vocab_size = len(self.tokenizer.word2idx)
        self.token_embedding = nn.Embedding(vocab_size, self.embed_size, padding_idx=0)
        self.model = StackedTransformer(
            embed_size=self.embed_size,
            vocab_size=vocab_size,
            num_layers=4,
            heads=8,
            forward_expansion=4,
            dropout=0.1
        ).to(self.device)
        self.model.token_embedding = self.model.embedding
        logger.info(f"Model initialized with vocab size: {vocab_size}")

    # =====================
    # Konfigurationsläsning
    # =====================
    def load_config(self, path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Could not load config: {e}")
            return {}

    # =====================
    # Checkpoint-hantering
    # =====================
    def save_checkpoint(self, epoch, optimizer, filename):
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

    # =====================
    # Collate fn för DataLoader
    # =====================
    def collate_fn(self, batch):
        inputs, targets = zip(*batch)
        max_len = max(max(len(seq) for seq in inputs), max(len(seq) for seq in targets))
        def pad(seqs):
            return torch.stack([torch.cat([seq, torch.zeros(max_len - len(seq), dtype=torch.long)]) for seq in seqs])
        return pad(inputs), pad(targets)

    # =====================
    # Enkel textgenerering
    # =====================
    def generate_text(self, prompt):
        tokens = self.tokenizer.encode(prompt)
        input_ids = torch.tensor([tokens], dtype=torch.long).to(self.device)
        mask = generate_square_subsequent_mask(input_ids.size(1)).to(self.device)
        with torch.no_grad():
            out = self.model(input_ids, mask)
        return self.tokenizer.decode(out.argmax(dim=-1).squeeze().tolist())
    
    def train_model(self, config_path="ai_Model/config.json"):
        """Kör träning via Trainer."""
        self.trainer.train(config_path=config_path)

    def load_model(self, filename="aether_model.pth"):
        if os.path.exists(filename):
            self.model.load_state_dict(torch.load(filename, map_location=self.device))
            self.model.to(self.device)
            print(f"Model loaded from {filename}")
        else:
            print(f"Model file '{filename}' not found. You need to train a new model first.")

    def save_model(self, filename=None):
        """Spara modell till fil via Trainer."""
        filename = filename or "aether_model.pth"
        torch.save(self.model.state_dict(), filename)
    # =====================
    # Kör agenten
    # =====================
    def run(self, user_input):
        self.memory.add(user_input)
        return self.generate_text(user_input)


# =====================
# Main
# =====================
if __name__ == "__main__":
    # 🔹 Initiera databas och agent
    db = DatabaseConnector()
    agent = AetherAgent(db)

    # 🔹 Skapa Trainer-instans kopplad till agentens modell och tokenizer
    agent.trainer = Trainer(
        model=agent.model,
        tokenizer=agent.tokenizer,
        device=agent.device,
        embed_size=agent.embed_size,
        tokenizer_path=agent.tokenizer_path
    )

    # 🔹 Försök ladda modell, annars träna ny modell
    model_file = "aether_model.pth"
    try:
        agent.load_model(filename=model_file)
        print(f"Loaded existing model from {model_file}")
    except FileNotFoundError:
        print(f"Model file '{model_file}' not found. Training new model...")
        agent.trainer.train(config_path=agent.config_path)
        agent.save_model(filename=model_file)
        print(f"Training complete. Model saved to {model_file}")

    # 🔹 Kör agenten interaktivt
    print("\nAether ready! Type 'exit' to quit.")
    print("Type 'train model' to retrain or continue training the model.\n")

    while True:
        try:
            user_input = input("> ")

            if user_input.lower() in ["exit", "quit"]:
                print("Exiting Aether...")
                break

            elif user_input.lower() == "train model":
                print("Training model... This may take a while.")
                agent.trainer.train(config_path=agent.config_path)
                agent.save_model(filename=model_file)
                print("Training complete!")
                continue

            # 🔹 Kör agenten för vanlig konversation
            response = agent.run(user_input)
            print(f"Aether: {response}")

        except KeyboardInterrupt:
            print("\nInterrupted by user. Exiting...")
            break
        except Exception as e:
            print(f"Error: {e}")
