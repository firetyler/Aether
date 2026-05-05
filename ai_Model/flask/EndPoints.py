import json
import os
import sys

os.environ["PYTHONIOENCODING"] = "utf-8"
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

from flask import Flask, request, jsonify

from ai_Model.utils.EndPointLog import get_logger
from ai_Model.aether2 import AetherAgent
from ai_Model.database.DatabaseConnector import DatabaseConnector

# =====================
# LOGGER
# =====================
logger = get_logger("EndPoints")

# =====================
# APP
# =====================
app = Flask(__name__)

# =====================
# DB + AGENT
# =====================
db_connector = DatabaseConnector()
agent = AetherAgent(db_connector)

config = agent.load_config("ai_Model/config.json")

# =====================
# LOAD / TRAIN MODEL SAFELY
# =====================
try:
    agent.load_model()
    logger.info("Model loaded successfully.")
except Exception as e:
    logger.warning(f"Model not found – training new model... {e}")

    # Kör träning. Ändra resume_from om du vill fortsätta från checkpoint
    agent.train_model(config_path="ai_Model/config.json", resume_from=None)
    agent.save_model(filename=config["model_path"])

    logger.info("New model trained and saved.")

# =====================
# CLEAN RESPONSE
# =====================
def clean_response(raw_output):
    """Parse JSON response"""
    if not raw_output or not isinstance(raw_output, str):
        return "{}"
    try:
        parsed = json.loads(raw_output)
        return parsed.get("response", parsed)
    except Exception:
        return raw_output


# =====================
# GENERATE ROUTE - Med sampling-metoder
# =====================
@app.route("/generate", methods=["POST"])
def generate():
    """Generera text med optional sampling-metoder. Stöder både enkla och avancerade requests."""
    try:
        data = request.get_json(force=True, silent=True) or {}
        logger.info(f"Generate request: {data}")

        user_input = data.get("prompt", "").strip()
        if not user_input:
            return jsonify({"error": "Invalid input"}), 400

        # Hämta optional sampling-parametrar
        method = data.get("method", "top_k")
        temperature = float(data.get("temperature", 0.7))
        k = int(data.get("k", 10))
        p = float(data.get("p", 0.9))
        max_length = int(data.get("max_length", 50))

        logger.info(f"Generating with method={method}, temperature={temperature}")

        try:
            if method == "beam":
                response = agent.generate_text_beam(user_input, beam_width=int(data.get("beam_width", 3)), max_length=max_length)
            else:
                response = agent.generate_text(user_input, method=method, temperature=temperature, k=k, p=p, max_length=max_length)
            
            logger.info(f"Generated: {response}")

        except Exception as e:
            logger.exception("Generation failed")
            return jsonify({"error": str(e)}), 500

        if not response:
            return jsonify({"error": "Empty response"}), 500

        db_connector.insert_conversation("User", user_input, response)

        return jsonify({
            "prompt": user_input,
            "response": response,
            "method": method,
            "temperature": temperature
        }), 200

    except Exception as e:
        logger.exception("Request processing failed")
        return jsonify({"error": str(e)}), 500


# =====================
# TRAIN / RESUME ROUTE
# =====================
@app.route("/train", methods=["POST"])
def train():
    try:
        data = request.get_json(force=True, silent=True) or {}
        resume_from = data.get("resume_from")
        
        logger.info(f"Training request. Resume from: {resume_from}")
        
        agent.train_model(config_path="ai_Model/config.json", resume_from=resume_from)
        agent.save_model(filename=config["model_path"])
        
        return jsonify({
            "status": "Training complete",
            "model_path": config["model_path"]
        }), 200
    
    except Exception as e:
        logger.exception("Training failed")
        return jsonify({"error": str(e)}), 500


# =====================
# HEALTH CHECK ROUTE
# =====================
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "model_loaded": agent.model is not None,
        "device": str(agent.device)
    }), 200


# =====================
# CHAT ROUTE - Alias för /generate (kompatibilitet)
# =====================
@app.route("/chat", methods=["POST"])
def chat():
    """Alias för /generate route"""
    return generate()


# =====================
# RUN SERVER
# =====================
if __name__ == "__main__":
    logger.info("Starting Flask server...")
    app.run(port=5000, debug=False)