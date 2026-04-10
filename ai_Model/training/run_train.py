import sys
import os
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from ai_Model.training.Trainer import Trainer
from ai_Model.tokenizer.BpeTokenizer import BpeTokenizer
from ai_Model.transformer.stacktTransformer import StackedTransformer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = BpeTokenizer()

model = StackedTransformer(
    vocab_size=537  # eller tokenizer vocab efter training
)

trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    device=device
)

trainer.train()