import json
from ai_Model.utils.EndPointLog import get_logger

from flask import Flask, request, jsonify
from ai_Model.aether2  import AetherAgent
from ai_Model.database.DatabaseConnector import DatabaseConnector
from ai_Model.training.Traning.train_model import Trainer

# ✅ Initialize logger correctly
logger = get_logger("EndPoints")

# ✅ Create Flask app
app = Flask(__name__)

# ✅ Initialize database connector BEFORE usage
db_connector = DatabaseConnector()

# ✅ Create AI agent
agent = AetherAgent(db_connector)

# ✅ Load configuration
config = agent.load_config("ai_Model/config.json")

# ✅ Ensure model is trained before running
try:
    agent.load_model()
    logger.info("Model loaded successfully.")
except Exception as e:
    logger.warning(f"Model not found – training new model... ({e})")
    agent.train_model(config_path="ai_Model/config.json")
    agent.save_model(filename=config["model_path"])
    logger.info("New model trained and saved.")
# ✅ Function to clean AI responses
def clean_response(raw_output):
    """Fix nested JSON encoding issues and ensure valid structure."""
    if not raw_output or not isinstance(raw_output, str):
        logger.error("Invalid raw_output received in clean_response")
        return "{}"

    try:
        parsed = json.loads(raw_output)  # Decode first level
        return parsed.get("response", parsed)  # Extract actual response if available
    except json.JSONDecodeError:
        logger.warning("JSON decoding failed, returning raw output")
        return raw_output  # ✅ Return safely instead of crashing


# ✅ Route for text generation
@app.route("/generate", methods=["POST"])
def generate():
    try:
        data = request.get_json()
        logger.info(f"Received request: {data}")  # 🔹 Log input data for debugging

        user_input = data.get("prompt", "").strip()
        if not user_input:
            logger.error("No prompt provided or empty input!")
            return jsonify({"error": "Invalid input"}), 400

        agent_response = agent.run(user_input)
        if not agent_response or not isinstance(agent_response, str):
            logger.error(f"Invalid agent response: {agent_response}")
            return jsonify({"error": "Agent failed to generate a valid response"}), 500

        cleaned_response = clean_response(agent_response)
        db_connector.insert_conversation("User", user_input, cleaned_response)

        return jsonify({"response": cleaned_response}), 200

    except Exception as e:
        logger.error(f"Error occurred in /generate: {e}")
        return jsonify({"error": str(e)}), 500


# ✅ Route for answering questions
@app.route("/ask", methods=["POST"])
def ask():
    try:
        data = request.get_json()
        user_input = data.get("prompt", "").strip()

        if not user_input:
            return jsonify({"error": "No prompt provided"}), 400

        logger.info(f"❔ Question received: {user_input}")
        agent_response = agent.run(user_input)

        if not agent_response:
            return jsonify({"error": "Agent failed to generate a response"}), 500

        db_connector.insert_conversation("User", user_input, agent_response)
        return jsonify({"input": user_input, "output": agent_response}), 200

    except Exception as e:
        logger.error(f"Error occurred: {e}")
        return jsonify({"error": str(e)}), 500


# ✅ Run Flask app
if __name__ == "__main__":
    logger.info("Starting Flask server...")
    app.run(port=5000, debug=False)
