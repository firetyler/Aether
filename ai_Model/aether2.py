import json
import os
import sys
import torch
import torch.nn as nn
import math
import re
from torch.utils.data import Dataset,DataLoader
from ai_Model.tokenizer.Tokenizer import BpeTokenizer
#from ai_Model.tokenizer import BpeTokenizer
from ai_Model.tokenizer.SimpleTokenizer import AdvancedSentenceTransformer, StackedTransformer, SimpleTokenizer
from ai_Model.momory.AetherMemory import AetherMemory
#from SimpleTokenizer import AdvancedSentenceTransformer, StackedTransformer, SimpleTokenizer
from ai_Model.utils.mask_utils import generate_square_subsequent_mask
sys.stdout.reconfigure(encoding='utf-8')  # Forces UTF-8 output
from ai_Model.database.DatabaseConnector import DatabaseConnector
from ai_Model.utils.logger_setup import get_logger
#from loggeclr_setup import get_logger
from ai_Model.codeEx.code_executor import CodeExecutor


logger = get_logger("aether2")
class ChatDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=128):
        self.data = data  # list of (input_text, output_text) tuples
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_text, output_text = self.data[idx]
        
        # Tokenisera input och output och konvertera till tensorer
        input_tokens = torch.tensor(self.tokenizer.encode(input_text), dtype=torch.long)
        output_tokens = torch.tensor(self.tokenizer.encode(output_text), dtype=torch.long)
        
        return input_tokens, output_tokens


class AetherAgent:
    def __init__(self, db_connector):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.db_connector = db_connector
        self.name = "Aether"
        #self.tokenizer = BpeTokenizer()
        self.tokenizer = SimpleTokenizer()
        self.model = None
        self.code_executor = CodeExecutor()
        self.embed_model = None
        self.embed_size = 256  # Mock for embed_model.embed_size

        # ✅ Ladda vokabulären innan något annat
        self.tokenizer.load_vocab("tokenizer_vocab.json")
        self.token_embedding = torch.nn.Embedding(self.tokenizer.vocab_size, self.embed_size, padding_idx=0)
        logger.info(f"Vocab loaded with size: {self.tokenizer.vocab_size}")

        # ✅ Se till att vokabulären är tillräckligt stor
        if self.tokenizer.vocab_size < 10:
            logger.warning(f"Vocabulary size ({self.tokenizer.vocab_size}) is too small! Check tokenizer_vocab.json.")

        # ✅ Instansiera `token_embedding`
        self.token_embedding = torch.nn.Embedding(self.tokenizer.vocab_size, self.embed_size, padding_idx=0)
        logger.info(f"Token embedding initialized with vocab size: {self.tokenizer.vocab_size}")

        # ✅ Skapa `embed_model` med korrekt vokabulärstorlek
        self.embed_model = AdvancedSentenceTransformer(vocab_size=self.tokenizer.vocab_size).to(self.device)

        # ✅ Skapa minnessystemet
        self.memory = AetherMemory(self.embed_model, self.tokenizer, self.device)

        # ✅ Referera till träningsdataset
        self.ChatDataset = ChatDataset
    def generate_square_subsequent_mask(sz):
        if sz < 1:
            raise ValueError(f"Mask size {sz} is too small.")
        mask = torch.triu(torch.ones(sz, sz), diagonal=1).bool()
        return mask.float()
    def update_model_embedding(self):
        """Ensure model embedding layer matches tokenizer vocab size dynamically."""
        if self.tokenizer.vocab_size > self.token_embedding.num_embeddings:
            logger.warning(f"Updating embedding layer to match new vocab size ({self.tokenizer.vocab_size})")
            
            self.token_embedding = torch.nn.Embedding(self.tokenizer.vocab_size, self.embed_size, padding_idx=0)
            self.model = StackedTransformer(
                embed_size=self.embed_size,
                vocab_size=self.tokenizer.vocab_size,
                num_layers=4, heads=8, forward_expansion=4, dropout=0.1
            ).to(self.device)

            logger.info("Model successfully updated to match new vocab size.")



    def handle_code_question(self, language, code):
        return self.code_executor.run_code(code, language)
    def load_config(self, path):
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f" Could not read the config file.: {e}")
            return {}
    def save_checkpoint(self, epoch, optimizer, filename):
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "tokenizer": self.tokenizer.word2idx
        }
        try:
            torch.save(checkpoint, filename)
            logger.info(f"Checkpoint saved: {filename}")
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")

        
        

    def load_checkpoint(self, filename, optimizer, config):
        if not os.path.isfile(filename):
            logger.warning(f"No checkpoint file found: {filename}")
            return 0

        checkpoint = torch.load(filename, map_location=self.device)
         # ✅ Reload tokenizer vocabulary
        self.tokenizer.load_vocab(config.get("tokenizer_vocab_path", "tokenizer_vocab.json"))
        logger.info(f"Tokenizer vocab loaded from file with size {self.tokenizer.vocab_size}")
        
        new_vocab_size = self.tokenizer.vocab_size
        self.token_embedding = nn.Embedding(new_vocab_size, self.embed_size, padding_idx=0)
        logger.info(f"Token embedding updated with vocab size {new_vocab_size}")
         # ✅ Rebuild model with updated vocabulary size
        self.model = StackedTransformer(
            embed_size=self.embed_size,
            vocab_size=new_vocab_size,
            num_layers=config.get("num_layers", 4),
            heads=config.get("heads", 8),
            forward_expansion=config.get("forward_expansion", 4),
            dropout=config.get("dropout", 0.1)
        ).to(self.device)

        model_dict = self.model.state_dict()
        pretrained_dict = {k: v for k, v in checkpoint["model_state_dict"].items() if k in model_dict and model_dict[k].shape == v.shape}
        model_dict.update(pretrained_dict)
        self.model.load_state_dict(model_dict)

        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            logger.info("Optimizer state loaded from checkpoint.")

        start_epoch = checkpoint.get("epoch", 0) + 1
        logger.info(f"Checkpoint loaded from {filename}, starting from epoch {start_epoch}")
        vocab_path = config.get("tokenizer_vocab_path", "tokenizer_vocab.json")
        self.tokenizer.save_vocab(vocab_path)
        

        if not os.path.exists(vocab_path):
            logger.warning("Vocab file not found after saving! Double-check file permissions.")
        else:
            logger.info(f"Vocabulary successfully saved to {vocab_path}.")

        return start_epoch
       
    def preprocess_data(self, training_data, config):
        all_texts = [q for q, _ in training_data] + [a for _, a in training_data]
        self.tokenizer.build_vocab(all_texts)

        num_layers = config.get("num_layers", 4)
        heads = config.get("heads", 8)
        forward_expansion = config.get("forward_expansion", 4)
        dropout = config.get("dropout", 0.1)
        embed_size = config.get("embedding_dim", 256)

        self.model = StackedTransformer(
            embed_size=embed_size,
            vocab_size=self.tokenizer.vocab_size,
            num_layers=num_layers,
            heads=heads,
            forward_expansion=forward_expansion,
            dropout=dropout
        ).to(self.device)

        inputs = []
        targets = []

        for text_input, expected_output in training_data:
            x = self.tokenizer.encode(text_input)
            y = self.tokenizer.encode(expected_output)
            inputs.append(torch.tensor(x, dtype=torch.long))
            targets.append(torch.tensor(y, dtype=torch.long))

    # Hitta maxlängd för input och output
        max_len_x = max([len(seq) for seq in inputs])
        max_len_y = max([len(seq) for seq in targets])

    # Pad sequences
        def pad_sequence(seq, max_len):
            return torch.cat([seq, torch.zeros(max_len - len(seq), dtype=torch.long)])

        inputs_padded = [pad_sequence(seq, max_len_x) for seq in inputs]
        targets_padded = [pad_sequence(seq, max_len_y) for seq in targets]

    # Flytta till device och lägg ihop i batcher (listor av tuples)
        data = [(seq_x.unsqueeze(0).to(self.device), seq_y.unsqueeze(0).to(self.device)) for seq_x, seq_y in zip(inputs_padded, targets_padded)]

        return data
    def initialize_model(self, config):
        num_layers = config.get("num_layers", 4)
        heads = config.get("heads", 8)
        forward_expansion = config.get("forward_expansion", 4)
        dropout = config.get("dropout", 0.1)
        embed_size = config.get("embedding_dim", 256)

        self.model = StackedTransformer(
            embed_size=embed_size,
            vocab_size=self.tokenizer.vocab_size,
            num_layers=num_layers,
            heads=heads,
            forward_expansion=forward_expansion,
            dropout=dropout
        ).to(self.device)

    def train_model(self, config_path="ai_Model/config.json"):
        """Träna AI-modellen med data från flera JSON-filer."""

        # ✅ Ladda konfiguration
        config = self.load_config(config_path)
        epochs = config.get("epochs", 10)
        lr = config.get("learning_rate", 0.001)
        batch_size = config.get("batch_size", 32)
        filenames = config.get("train_data_paths", [])
        num_layers = config.get("num_layers", 4)

        logger.info(f"Training files listed in config: {filenames}")
        logger.info(f"Current working directory: {os.getcwd()}")

        training_data = []
        for filename in filenames:
            if not isinstance(filename, str):
                logger.error(f"Invalid filename format: {filename}. Expected a string.")
                continue

            if not os.path.exists(filename):
                logger.error(f"Training data file not found: {filename}")
                continue

            try:
                with open(filename, "r", encoding="utf-8") as f:
                    raw_data = json.load(f)
                    if isinstance(raw_data, dict):
                        raw_data = raw_data.get("data", [])

                    valid_data = [(d["input"], d["output"]) for d in raw_data if "input" in d and "output" in d]
                    training_data.extend(valid_data)
                    logger.info(f"Loaded {len(valid_data)} examples from {filename}")

            except Exception as e:
                logger.error(f"Failed to load training data from {filename}: {e}")

        if not training_data:
            logger.error("No valid training data loaded. Aborting training...")
            return

        # ✅ Bygg och spara vokabulär från träningsdata
        all_texts = [q for q, _ in training_data] + [a for _, a in training_data]
        self.tokenizer.build_vocab(all_texts)
        self.tokenizer.save_vocab("tokenizer_vocab.json")
        logger.info(f"Tokenizer updated and saved with {len(self.tokenizer.word2idx)} tokens.")

        dataset = self.ChatDataset(training_data, self.tokenizer)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=self.collate_fn)

        vocab_size = len(self.tokenizer.word2idx)
        logger.info(f"Using vocab_size = {vocab_size} for model initialization.")

        self.model = StackedTransformer(
            embed_size=config.get("embedding_dim", 256),
            vocab_size=vocab_size,
            num_layers=config.get("num_layers", 4),
            heads=config.get("heads", 8),
            forward_expansion=config.get("forward_expansion", 4),
            dropout=config.get("dropout", 0.1)
        ).to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        self.model.train()

        checkpoint_path = "checkpoint_latest.pth"
        start_epoch = 0
        if os.path.exists(checkpoint_path):
            logger.info(f"Found checkpoint: Loading from {checkpoint_path}...")
            start_epoch = self.load_checkpoint(checkpoint_path, optimizer, config)
            if start_epoch is None or not isinstance(start_epoch, int):
                logger.warning(f"Invalid start_epoch: {start_epoch}. Setting default to 0.")
                start_epoch = 0
        else:
            logger.warning("No checkpoint found – starting training from scratch...")

        if start_epoch >= epochs:
            logger.warning(f"start_epoch ({start_epoch}) is greater than total epochs ({epochs}). Skipping training.")
            return

        if epochs <= start_epoch:
            logger.warning(f"Epochs ({epochs}) are too low. Adjusting to {start_epoch + 10}.")
            epochs = start_epoch + 10

        logger.info(f"Starting model training for {epochs} epochs...")
        best_loss = float('inf')  
        patience_counter = 0      

        early_stopping_patience = config.get("early_stopping_patience", 3)
        for epoch in range(start_epoch, epochs):
            total_loss = 0.0
            for x, y in dataloader:
                x = x.to(self.device)
                y = y.to(self.device)

                optimizer.zero_grad()

                sz = x.size(1)
                if sz < 1:
                    logger.error(f"Invalid mask size: {sz}. Skipping batch.")
                    continue

                mask = generate_square_subsequent_mask(sz).to(self.device)
                out = self.model(x, mask)

                loss = criterion(out.view(-1, out.size(-1)), y.view(-1))
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(dataloader)
            logger.info(f"Epoch {epoch + 1}: Loss = {avg_loss:.4f}")
            #stänger av om ingen ändring sker
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
                self.save_checkpoint(epoch + 1, optimizer, checkpoint_path)
                logger.info(f"Epoch {epoch + 1}: Loss improved to {avg_loss:.4f}, checkpoint saved.")
            else:
                patience_counter += 1
                logger.info(f"Epoch {epoch + 1}: No improvement. Patience counter = {patience_counter}/{early_stopping_patience}")

            if patience_counter >= early_stopping_patience:
                logger.info(f"No improvement for {early_stopping_patience} epochs. Early stopping at epoch {epoch + 1}.")
                break
            
            self.save_checkpoint(epoch + 1, optimizer, checkpoint_path)

        logger.info("Training complete! Model saved successfully.")

        # === Ladda modellen och tokenizern efter träning ===
        logger.info("Reloading model and tokenizer after training...")

        # Ladda tokenizer vocab
        self.tokenizer.load_vocab("tokenizer_vocab.json")
        logger.info(f"Tokenizer vocab reloaded with {len(self.tokenizer.word2idx)} tokens.")

        # Ladda modell från checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()  # sätt modellen i eval-läge för inferens

        logger.info("Model reloaded and ready for inference.")

    def collate_fn(self, batch):
        inputs, targets = zip(*batch)
    
    # Hitta maxlängden bland både inputs och targets
        max_len = max(
        max(len(seq) for seq in inputs),
        max(len(seq) for seq in targets)
        )
    
        def pad_sequences(sequences, max_len):
            padded_seqs = []
            for seq in sequences:
                seq_tensor = seq.detach().clone()
                pad_size = max_len - len(seq)
                if pad_size > 0:
                    padding = torch.zeros(pad_size, dtype=torch.long)
                    seq_tensor = torch.cat([seq_tensor, padding])
                padded_seqs.append(seq_tensor)
            return torch.stack(padded_seqs)
    
        inputs_padded = pad_sequences(inputs, max_len)
        targets_padded = pad_sequences(targets, max_len)
    
        return inputs_padded, targets_padded
    
    def save_model(self, filename="aether_model.pth"):
        torch.save(self.model.state_dict(), filename)

    def load_model(self, filename="aether_model.pth"):
        # Ladda config och initiera modell
        config = self.load_config("ai_Model/config.json")
        self.initialize_model(config)

        if self.model is None:
            raise RuntimeError("Model not initialized. You must train or preprocess first.")

        try:
            state_dict = torch.load(filename, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.eval()
            logger.info(f"Model loaded successfully from {filename}.")

        except FileNotFoundError:
            logger.warning(f"Model file '{filename}' not found. Training new model...")
            self.train_model(config_path="ai_Model/config.json")
            self.save_model(filename)

        except RuntimeError as e:
            if "size mismatch" in str(e):
                logger.warning(f"Size mismatch error when loading model: {e}")
                logger.warning("Vocabulary or architecture changed — retraining model...")
                self.train_model(config_path="ai_Model/config.json")
                self.save_model(filename)
            else:
                raise


    def generate_text(self, prompt):
        if self.model is None:
            logger.warning("Model not initialized.")
            return json.dumps({"error": "Model not initialized."})

        # 🔹 Ladda config för korrekt instansiering
        config = self.load_config("ai_Model/config.json")

        # 🔹 Tokenisera först och kolla om några nya ord lagts till
        tokens = self.tokenizer.encode(prompt)
        logger.info(f"Tokenized input: {tokens}")

        vocab_size = self.tokenizer.vocab_size
        max_token_index = max(tokens) if tokens else -1

        # ✅ Uppdatera model och token_embedding FÖRE token används i modellen
        if vocab_size > self.token_embedding.num_embeddings or max_token_index >= self.token_embedding.num_embeddings:
            logger.warning(f"Updating model and embeddings to vocab size {vocab_size} (max token index: {max_token_index})")
            self.token_embedding = torch.nn.Embedding(vocab_size, self.embed_size, padding_idx=0)
            self.model = StackedTransformer(
                embed_size=config.get("embedding_dim", 256),
                vocab_size=vocab_size,
                num_layers=config.get("num_layers", 4),
                heads=config.get("heads", 8),
                forward_expansion=config.get("forward_expansion", 4),
                dropout=config.get("dropout", 0.1)
            ).to(self.device)
            logger.info("Model and embedding updated.")

        if not tokens or max_token_index >= self.token_embedding.num_embeddings:
            logger.error(f"Token index {max_token_index} exceeds embedding size ({self.token_embedding.num_embeddings})")
            return json.dumps({"error": "Generated token is out of range."})

        input_ids = torch.tensor([tokens], dtype=torch.long).to(self.device)
        mask = generate_square_subsequent_mask(input_ids.size(1)).to(self.device)

        with torch.no_grad():
            out = self.model(input_ids, mask)

        generated_text = self.tokenizer.decode(out.argmax(dim=-1).squeeze().tolist())
        return json.dumps({"response": generated_text})







    def run(self, user_input):
        logger.info(f"GOAL: {user_input}")
        
        # 🔹 Rensa och formattera input
        corrected_input = re.sub(r"\s+", " ", user_input.strip().lower())

        # 🔹 Identifiera namnrelaterade frågor
        if re.match(r"^(what('?s| is) your name\??)$", user_input.lower()):
            return "My name is Aether."

        lowered = user_input.lower()

        # 🔹 Hantera kod-exekvering (fixad variabelordning)
        if lowered.startswith("run "):
            try:
                parts = user_input.split(" ", 2)
                if len(parts) < 3:
                    logger.warning("Invalid format for the 'run' command.")
                    return "Use the format: run <language> <code>"
                
                language = parts[1]
                code = parts[2]

                logger.info(f"Running code in language: {language}")
                logger.debug(f"Code being executed:\n{code}")

                result = self.handle_code_question(language, code)
                logger.info(f"Code execution succeeded for language: {language}")
                return f"Result for {language}:\n{result}"
            
            except Exception as e:
                logger.exception(f"Error during code execution in language: {language}")
                return f"Something went wrong during code execution: {e}"
        
        # 🔹 Träna modellen
        if lowered == "train model":
            try:
                logger.info("Starting model training process...")
                self.train_model()
                self.save_model()
                logger.info("Training finished.")
                return "Model trained and saved."
            except Exception as e:
                logger.exception(f"Error during training: {e}")
                return f"Training failed: {e}"

        # 🔹 Kalkylator-verktyg (Fixad tom uttryckshantering)
        if "calculate" in lowered:
            expression = lowered.replace("calculate", "").strip()
            if not expression:
                return "Invalid calculation request. Please provide an expression."
            return self.memory.calculator_tool(expression)

        # 🔹 Wikipedia-sökning
        if "wikipedia" in lowered:
            query = lowered.replace("wikipedia", "").strip()
            if not query:
                return "Invalid Wikipedia request. Please provide a search term."
            return self.memory.wikipedia_tool(query)
        
        # 🔹 Hantera om användaren vill försöka igen
        if lowered in ["try again", "that's wrong", "incorrect", "answer again", "that doesn't sound right"]:
            if self.memory.memories and len(self.memory.memories) > 0:
                last_prompt = self.memory.memories[-1]
                logger.info(f"User requested retry for: {last_prompt}")
                regenerated = self.generate_text(last_prompt)
                self.db_connector.insert_conversation(name="Aether", input="", output=regenerated)
                return regenerated
            else:
                return "I don't have a previous message to retry."

        # 🔹 Generera nytt svar och lagra i minnet
        generated = self.generate_text(user_input)
        self.memory.add(user_input)
        self.db_connector.insert_conversation(name="User", input=user_input, output="")
        self.db_connector.insert_conversation(name="Aether", input="", output=generated)

        return generated

# --- Kör agenten ---
if __name__ == "__main__":
    logger.info("Starting Aether...")

    # ✅ Initialize database connector
    db = DatabaseConnector()

    # ✅ Create AI agent
    agent = AetherAgent(db)

    # ✅ Load configuration safely
    config = agent.load_config("ai_Model/config.json")

    # ✅ Build tokenizer vocabulary with multiple files
    try:
        filenames = config.get("train_data_paths", ["ai_Model/chat_training_data.json"])  # Ensure list format
        training_data = []

        for filename in filenames:
            if not isinstance(filename, str):
                logger.error(f"Invalid filename format: {filename}. Expected a string.")
                continue

            try:
                with open(filename, "r", encoding="utf-8") as f:
                    raw_data = json.load(f)
                    valid_data = [(d["input"], d["output"]) for d in raw_data if "input" in d and "output" in d]
                    training_data.extend(valid_data)
            except Exception as e:
                logger.error(f"Failed to load training data from {filename}: {e}")

        if training_data:
            all_texts = [q for q, _ in training_data] + [a for _, a in training_data]
            agent.tokenizer.build_vocab(all_texts)
            logger.info(f"Tokenizer vocabulary updated with {len(all_texts)} examples.")
        else:
            logger.warning("No valid training data loaded for vocabulary.")

    except Exception as e:
        logger.error(f"Failed to build tokenizer vocabulary: {e}")

    # ✅ Initialize and load the model safely
    agent.initialize_model(config)

    # ✅ Debugging info for file paths
    logger.info(f"Current working directory: {os.getcwd()}")
    logger.info(f"Files in current directory: {os.listdir()}")

    try:
        agent.load_model(filename=config["model_path"])
        logger.info("Model loaded successfully.")
    except Exception as e:
        logger.warning(f"Model not found – training new model... ({e})")
        agent.train_model(config_path="ai_Model/config.json")
        agent.save_model(filename=config["model_path"])

    logger.info("Aether is ready. Type something to chat!")

    # ✅ Terminal interaction loop
    while True:
        try:
            user_input = input("\n> ")  # 🔹 Gör att du kan skriva i terminalen
            if user_input.lower() in ["exit", "quit"]:
                logger.info("Exiting Aether.")
                break

            elif user_input.lower() == "clear":
                os.system("cls" if os.name == "nt" else "clear")  # 🔹 Rensar terminalen
                continue

            elif user_input.lower() == "help":
                print("\n🛠 Available Commands:")
                print("- Type any message to chat with Aether.")
                print("- `exit` or `quit` to leave the chat.")
                print("- `clear` to clear the terminal screen.")
                print("- `train model` to retrain Aether.")
                continue

            elif user_input.lower() == "train model":
                logger.info("Retraining model...")
                agent.train_model(config_path="ai_Model/config.json")
                agent.save_model(filename=config["model_path"])
                print("✅ Model retrained successfully!")
                continue

            output = agent.run(user_input)  # 🔹 Genererar svar från AI-agenten
            logger.info(f"Response: {output}")
            print(f"Aether: {output}")  # 🔹 Skriver ut svaret i terminalen

        except KeyboardInterrupt:
            logger.info("Interrupted by user. Exiting...")
            break

