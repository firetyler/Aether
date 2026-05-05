import sys
import os
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from ai_Model.training.Trainer import Trainer
from ai_Model.tokenizer.BpeTokenizer import BpeTokenizer
from ai_Model.transformer.stacktTransformer.stacked_transformer import AdvancedStackedTransformer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = BpeTokenizer()

model = AdvancedStackedTransformer(
    vocab_size=537,
    embed_size=256,
    num_layers=4,
    heads=8,
    forward_expansion=4,
    dropout=0.1
).to(device)

trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    device=device
)

# Alternativ 1: Ny träning
trainer.train(config_path="ai_Model/config.json")

# Alternativ 2: Fortsätt från checkpoint
# trainer.train(config_path="ai_Model/config.json", resume_from="checkpoints/checkpoint_epoch_5.pth")