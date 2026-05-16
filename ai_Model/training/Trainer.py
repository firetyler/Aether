import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from ai_Model.chat_dataset.chatDataset import ChatDataset
from ai_Model.utils.mask_utils import generate_square_subsequent_mask
from ai_Model.utils.logger_common import get_logger

logger = get_logger("trainer")


class Trainer:
    def __init__(self, model, tokenizer, device=None):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_config(self, path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def load_checkpoint(self, checkpoint_path):
        """Ladda checkpoint för att fortsätta träning"""
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"Model loaded from checkpoint: {checkpoint_path}")
            return checkpoint
        else:
            logger.warning(f"Checkpoint not found: {checkpoint_path}")
            return None

    def save_checkpoint(self, optimizer, epoch, loss, checkpoint_path):
        """Spara checkpoint"""
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'vocab_size': len(self.tokenizer.word2idx) if hasattr(self.tokenizer, 'word2idx') else 0
        }, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")

    def load_model_for_inference(self, model_path):
        """Ladda sparad modell för inference"""
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
            logger.info(f"Model loaded for inference: {model_path}")
            return True
        else:
            logger.error(f"Model file not found: {model_path}")
            return False

    def load_checkpoint_for_inference(self, checkpoint_path):
        """Ladda checkpoint för inference"""
        checkpoint = self.load_checkpoint(checkpoint_path)
        if checkpoint:
            self.model.eval()
            logger.info(f"Model in eval mode. Ready for inference.")
            return True
        return False

    def collate_fn(self, batch):
        inputs, targets = zip(*batch)

        max_len = max(max(len(i) for i in inputs), max(len(t) for t in targets))

        def pad(seqs):
            return torch.stack([
                torch.cat([seq, torch.zeros(max_len - len(seq), dtype=torch.long)])
                for seq in seqs
            ])

        return pad(inputs), pad(targets)

    def train(self, config_path="ai_Model/config.json", resume_from=None):
        config = self.load_config(config_path)

        batch_size = config.get("batch_size", 32)
        epochs = config.get("epochs", 50)
        lr = config.get("learning_rate", 0.0005)
        data_paths = config.get("train_data_paths", [])
        checkpoint_dir = config.get("checkpoint_dir", "checkpoints")
        checkpoint_interval = config.get("checkpoint_interval", 5)
        validation_split = config.get("validation_split", 0.1)
        max_grad_norm = config.get("max_grad_norm", 1.0)
        patience = config.get("patience", 10)
        warmup_steps = config.get("warmup_steps", 500)
        log_interval = config.get("log_interval", 10)

        os.makedirs(checkpoint_dir, exist_ok=True)

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

        # ---------------- DATASET & VALIDATION SPLIT ----------------
        dataset = ChatDataset(training_data, self.tokenizer)
        val_size = int(len(dataset) * validation_split)
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=self.collate_fn
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=self.collate_fn
        )

        # ---------------- OPTIMIZER & SCHEDULER ----------------
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss(ignore_index=0)
        
        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        # Warm-up
        def warmup_lr(step):
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            return 1.0
        
        warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_lr)

        # Ladda checkpoint om specificerad
        start_epoch = 0
        best_val_loss = float('inf')
        patience_counter = 0
        
        if resume_from:
            checkpoint = self.load_checkpoint(resume_from)
            if checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                start_epoch = checkpoint['epoch']
                best_val_loss = checkpoint.get('best_val_loss', float('inf'))
                logger.info(f"Resuming training from epoch {start_epoch}")

        # ---------------- TRAIN ----------------
        for epoch in range(start_epoch, epochs):
            # Training phase
            self.model.train()
            train_loss = 0
            num_batches = 0

            for batch_idx, (x, y) in enumerate(train_loader):
                x, y = x.to(self.device), y.to(self.device)

                optimizer.zero_grad()
                warmup_scheduler.step()

                mask = generate_square_subsequent_mask(x.size(1)).to(self.device)
                out = self.model(x, mask)
                loss = criterion(out.view(-1, out.size(-1)), y.view(-1))

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                optimizer.step()

                train_loss += loss.item()
                num_batches += 1
                
                if (batch_idx + 1) % log_interval == 0:
                    logger.info(f"Epoch {epoch+1} [{batch_idx+1}/{len(train_loader)}] Loss: {loss.item():.4f}")

            avg_train_loss = train_loss / num_batches
            
            # Validation phase
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(self.device), y.to(self.device)
                    mask = generate_square_subsequent_mask(x.size(1)).to(self.device)
                    out = self.model(x, mask)
                    loss = criterion(out.view(-1, out.size(-1)), y.view(-1))
                    val_loss += loss.item()
            
            avg_val_loss = val_loss / len(val_loader)
            logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
            
            # Learning rate scheduler step
            scheduler.step()
            
            # Checkpoint saving
            if (epoch + 1) % checkpoint_interval == 0:
                checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pth")
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': avg_train_loss,
                    'val_loss': avg_val_loss,
                    'best_val_loss': best_val_loss,
                    'vocab_size': vocab_size
                }, checkpoint_path)
                logger.info(f"Checkpoint saved: {checkpoint_path}")
            
            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                # Spara best model
                best_model_path = os.path.join(checkpoint_dir, "best_model.pth")
                torch.save(self.model.state_dict(), best_model_path)
                logger.info(f"Best model updated. Val Loss: {best_val_loss:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping triggered. Best val loss: {best_val_loss:.4f}")
                    break

        # Spara final modell
        torch.save(self.model.state_dict(), config.get("model_path", "aether_model.pth"))
        logger.info("Training complete")