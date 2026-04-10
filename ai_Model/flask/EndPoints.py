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

    # ❌ FIX: använd agent, inte import av train_model
    agent.train_model(config_path="ai_Model/config.json")
    agent.save_model(filename=config["model_path"])

    logger.info("New model trained and saved.")

# =====================
# CLEAN RESPONSE
# =====================
def clean_response(raw_output):
    if not raw_output or not isinstance(raw_output, str):
        return "{}"

    try:
        parsed = json.loads(raw_output)
        return parsed.get("response", parsed)
    except Exception:
        return raw_output


# =====================
# GENERATE ROUTE
# =====================
@app.route("/generate", methods=["POST"])
def generate():
    try:
        data = request.get_json(force=True, silent=True) or {}
        logger.info(f"Received request: {data}")

        user_input = data.get("prompt", "").strip()
        if not user_input:
            return jsonify({"error": "Invalid input"}), 400

        logger.info("Calling agent.run...")

        try:
            agent_response = agent.run(user_input)
            logger.info(f"RAW RESPONSE: {agent_response}")

        except AssertionError:
            logger.exception("FAISS dimension mismatch")
            return jsonify({
                "error": "Memory mismatch (FAISS)",
                "fix": "Reset memory / reinitialize embeddings"
            }), 500

        except Exception as e:
            logger.exception("agent.run crashed")
            return jsonify({"error": str(e)}), 500

        if not agent_response:
            return jsonify({"error": "Empty response"}), 500

        if not isinstance(agent_response, str):
            agent_response = str(agent_response)

        cleaned_response = clean_response(agent_response)
        logger.info(f"CLEANED RESPONSE: {cleaned_response}")

        try:
            db_connector.insert_conversation("User", user_input, cleaned_response)
        except Exception:
            logger.exception("DB insert failed (ignored)")

        return jsonify({"response": cleaned_response}), 200

    except Exception:
        logger.exception("FULL ERROR in /generate")
        return jsonify({"error": "internal crash"}), 500


# =====================
# ASK ROUTE
# =====================
@app.route("/ask", methods=["POST"])
def ask():
    try:
        data = request.get_json() or {}
        user_input = data.get("prompt", "").strip()

        if not user_input:
            return jsonify({"error": "No prompt provided"}), 400

        logger.info(f"Question: {user_input}")

        response = agent.run(user_input)

        db_connector.insert_conversation("User", user_input, response)

        return jsonify({
            "input": user_input,
            "output": response
        }), 200

    except Exception as e:
        logger.exception("ASK route error")
        return jsonify({"error": str(e)}), 500


# =====================
# RUN SERVER
# =====================
if __name__ == "__main__":
    logger.info("Starting Flask server...")
    app.run(port=5000, debug=False)