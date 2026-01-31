import faiss
import numpy as np
import torch
import wikipedia
import re
from ai_Model.utils.AetherMemoryLog import get_logger

logger = get_logger("AetherMemory")


class AetherMemory:
    def __init__(self, embed_model, tokenizer, device):
        self.embed_model = embed_model
        self.tokenizer = tokenizer
        self.device = device

        self.memories = []
        self.vector_dim = embed_model.embed_size if hasattr(embed_model, "embed_size") else 256

        # FAISS-index
        self.index = faiss.IndexFlatL2(self.vector_dim)

        # Kontrollera token_embedding
        if not hasattr(embed_model, "token_embedding"):
            logger.error("embed_model saknar `token_embedding`! Kontrollera att modellen är korrekt instansierad.")
            raise ValueError("embed_model saknar `token_embedding`")

    # =====================
    # Lägg till text i minnet
    # =====================
    def add(self, text):
        vector = self.text_to_vector(text)
        if vector is None:
            logger.warning(f"Memory skipped: '{text[:50]}...'")
            return

        self.memories.append(text)
        self.index.add(vector)
        logger.info(f"Memory added: '{text[:50]}...'")

    # =====================
    # Konvertera text -> vektor
    # =====================
    def text_to_vector(self, text):
        self.embed_model.eval()

        # 🔹 Uppdatera token_embedding om vokabulär växt
        if self.tokenizer.vocab_size > self.embed_model.token_embedding.num_embeddings:
            old_embed_size = self.embed_model.token_embedding.embedding_dim
            self.embed_model.token_embedding = torch.nn.Embedding(
                self.tokenizer.vocab_size, old_embed_size, padding_idx=0
            )
            logger.info(f"Token embedding updated: vocab_size={self.tokenizer.vocab_size}")

        # Tokenisera text
        token_ids = self.tokenizer.encode(text)
        if not token_ids:
            return None

        if max(token_ids) >= self.embed_model.token_embedding.num_embeddings:
            logger.warning(f"Token index {max(token_ids)} exceeds embedding vocab size. Using <UNK> token.")
            token_ids = [0 if t >= self.embed_model.token_embedding.num_embeddings else t for t in token_ids]

        input_ids = torch.tensor([token_ids], dtype=torch.long, device=self.device)

        with torch.no_grad():
            # Om din modell returnerar (batch, seq_len, embed_size)
            embeddings = self.embed_model(input_ids)
            if isinstance(embeddings, tuple) or isinstance(embeddings, list):
                embeddings = embeddings[0]  # Ta första elementet om output är tuple
            # Ta genomsnitt över sekvensdimensionen
            vector = embeddings.mean(dim=1).cpu().numpy().astype(np.float32)

        return vector

    # =====================
    # Semantisk sökning
    # =====================
    def semantic_search(self, query, top_k=3):
        if not self.memories:
            return []

        query_vector = self.text_to_vector(query)
        if query_vector is None:
            return []

        distances, indices = self.index.search(query_vector, top_k)
        results = [
            (self.memories[idx], float(distances[0][i]))
            for i, idx in enumerate(indices[0])
            if idx < len(self.memories)
        ]
        return results

    # =====================
    # Kalkylatorverktyg
    # =====================
    def calculator_tool(self, expression):
        import ast
        try:
            logger.info(f"Evaluating expression: {expression}")
            safe_expr = re.sub(r"[^0-9+\-*/(). ]", "", expression)
            result = ast.literal_eval(safe_expr)
            return f"Result: {result}"
        except Exception as e:
            logger.error(f"Error evaluating expression '{expression}': {e}")
            return f"Error in calculation: {e}"

    # =====================
    # Wikipedia-verktyg
    # =====================
    def wikipedia_tool(self, query):
        import wikipedia
        try:
            summary = wikipedia.summary(query, sentences=2)
            return summary
        except wikipedia.exceptions.DisambiguationError as e:
            return f"Disambiguation error. Try: {', '.join(e.options[:3])}"
        except wikipedia.exceptions.PageError:
            return f"No Wikipedia page found for: {query}"
        except Exception as e:
            return f"Wikipedia lookup failed: {e}"
