import os
import json
import torch
from ai_Model.tokenizer.BpeTokenizer import BpeTokenizer
from ai_Model.transformer.stacktTransformer.stacked_transformer import AdvancedStackedTransformer
from ai_Model.momory.AetherMemory import AetherMemory
from ai_Model.database.DatabaseConnector import DatabaseConnector
from ai_Model.utils.logger_setup import get_logger
from ai_Model.training.Trainer import Trainer
from ai_Model.inference.inference import generate_text, beam_search_generate 

logger = get_logger("aether2")

class AetherAgent:
    def __init__(self, db_connector, tokenizer_path="bpe_tokenizer.json", config_path="ai_Model/config.json"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.db_connector = db_connector
        self.name = "Aether"
        self.embed_size = 256
        self.tokenizer_path = tokenizer_path
        self.config_path = config_path
        
        self.model = None
        self.memory = None

        # Initiera tokenizer
        self._init_tokenizer()

        # Initiera modell
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
        """Träna tokenizer från träningsdata"""
        config = self.trainer.load_config(self.config_path)
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
        self.model = AdvancedStackedTransformer(
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
    # Checkpoint-hantering (i Trainer.py) - ta bort dubbletter här
    # =====================

    # =====================
    # Collate fn för DataLoader (i Trainer.py) - ta bort dubblett här
    # =====================

    # =====================
    # Enkel textgenerering
    # =====================
    def generate_text(self, prompt, method="top_k", temperature=0.7, k=10, p=0.9, max_length=50):
        """Generera text med olika sampling-metoder.
        
        Args:
            prompt: Starttext
            method: "top_k" (default), "nucleus", "beam", eller "greedy"
            temperature: Kreativitet (högre = mer varierat)
            k: Antal top-k tokens
            p: Nucleus sampling parameter
            max_length: Max genererad längd
        """
        return generate_text(
            self.model, self.tokenizer, prompt,
            max_length=max_length,
            method=method,
            temperature=temperature,
            k=k, p=p
        )

    def generate_text_beam(self, prompt, beam_width=3, max_length=50):
        """Generera text med beam search"""
        return beam_search_generate(self.model, self.tokenizer, prompt, max_length, beam_width)
    
    def train_model(self, config_path="ai_Model/config.json", resume_from=None):
        """Kör träning via Trainer. Kan fortsätta från checkpoint om resume_from är specificerad."""
        self.trainer.train(config_path=config_path, resume_from=resume_from)

    def load_model(self, filename="aether_model.pth"):
        """Ladda modell för inference via Trainer"""
        if os.path.exists(filename):
            success = self.trainer.load_model_for_inference(filename)
            if success:
                logger.info(f"Model loaded from {filename}")
        else:
            logger.error(f"Model file '{filename}' not found.")

    def save_model(self, filename=None):
        """Spara modell"""
        filename = filename or "aether_model.pth"
        torch.save(self.model.state_dict(), filename)
        logger.info(f"Model saved to {filename}")
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
    # Initiera databas och agent
    db = DatabaseConnector()
    agent = AetherAgent(db)

    # Försök ladda modell, annars träna ny modell
    model_file = "aether_model.pth"
    try:
        agent.load_model(filename=model_file)
        logger.info(f"Loaded existing model from {model_file}")
    except FileNotFoundError:
        logger.warning(f"Model file '{model_file}' not found. Training new model...")
        agent.trainer.train(config_path=agent.config_path)
        agent.save_model(filename=model_file)

    logger.info("Aether ready!")

    while True:
        try:
            user_input = input("> ")

            if user_input.lower() in ["exit", "quit"]:
                logger.info("Exiting Aether...")
                break

            elif user_input.lower() == "train model":
                logger.info("Training model...")
                agent.trainer.train(config_path=agent.config_path)
                agent.save_model(filename=model_file)
                logger.info("Training complete!")
                continue

            # Kör agenten
            response = agent.run(user_input)
            logger.info(f"Response: {response}")

        except KeyboardInterrupt:
            logger.info("Interrupted by user.")
            break
        except Exception as e:
            logger.exception(f"Error: {e}")
