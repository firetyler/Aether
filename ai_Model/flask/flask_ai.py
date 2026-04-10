import json
import os
import sys
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")
from flask import Flask, request, jsonify

# Lägg till projektets root (en nivå upp) i sökvägar för imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ai_Model.database.DatabaseConnector import DatabaseConnector
from aether2 import AetherAgent
from ai_Model.utils.EndPointLog import get_logger


logger = get_logger("Endpoints")
app = Flask(__name__)

# Initiera global DB-connector
db_connector = DatabaseConnector()

# Håll AI-agent-instansier i en dict {instance_id: AetherAgent}
agents = {}

def start_agent(instance_id, db_connector):
    # Base dir är ai_Model-mappen, där config ligger
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    config_path = os.path.join(base_dir, 'configEndpoint.json')
    if not os.path.exists(config_path):
        logger.error(f"Config file does not exist at: {config_path}")
        return None

    # Läs config
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        logger.info(f"[{instance_id}] Loaded config from {config_path}")
    except Exception as e:
        logger.error(f"[{instance_id}] Failed to load config: {e}")
        return None

    # Initiera agent med DB-connector
    agent = AetherAgent(db_connector)
    try:
        agent.load_config(config_path)
    except Exception as e:
        logger.error(f"[{instance_id}] Failed to load config into agent: {e}")
        return None

    # Försök ladda modell
    try:
        agent.load_model()
        logger.info(f"[{instance_id}] Model loaded successfully.")
    except Exception as e:
        logger.warning(f"[{instance_id}] Model not found or failed to load: {e}")

        # Kolla att träningsdata finns innan träning
        train_paths = config.get('train_data_paths', [])
        if not train_paths:
            logger.error(f"[{instance_id}] No training data paths specified in config.")
            return None

        # Kontrollera att träningsfilerna finns (fulla sökvägar byggs här)
        missing_files = []
        for relative_path in train_paths:
            full_path = os.path.join(base_dir, relative_path)
            if not os.path.exists(full_path):
                missing_files.append(relative_path)

        if missing_files:
            logger.error(f"[{instance_id}] Training data files missing: {missing_files}")
            return None

        # Träna modellen
        try:
            agent.train_model(config_path=config_path)
            model_path = config.get('model_path')
            if model_path:
                full_model_path = os.path.join(base_dir, model_path)
                agent.save_model(filename=full_model_path)
                logger.info(f"[{instance_id}] Model trained and saved to {full_model_path}")
            else:
                logger.error(f"[{instance_id}] model_path not specified in config, cannot save model.")
                return None
        except Exception as train_e:
            logger.error(f"[{instance_id}] Failed to train or save model: {train_e}")
            return None

    return agent

# Funktion för att städa AI-svar
def clean_response(raw_output):
    if not raw_output or not isinstance(raw_output, str):
        logger.error("Invalid raw_output received in clean_response")
        return "{}"
    try:
        parsed = json.loads(raw_output)
        return parsed.get("response", parsed)
    except json.JSONDecodeError:
        logger.warning("JSON decoding failed, returning raw output")
        return raw_output

@app.route("/generate/<instance_id>", methods=["POST"])
def ask_ai(instance_id):
    if instance_id not in agents:
        return jsonify({"error": f"AI instance {instance_id} does not exist."}), 404

    user_input = request.json.get("prompt", "").strip()
    if not user_input:
        return jsonify({"error": "No prompt provided"}), 400

    logger.info(f"[{instance_id}] Question received: {user_input}")
    try:
        response = agents[instance_id].run(user_input)
    except Exception as e:
        logger.error(f"[{instance_id}] Error during agent run: {e}")
        return jsonify({"error": "Agent failed to generate response"}), 500

    if not response:
        return jsonify({"error": "Agent failed to generate a response"}), 500

    cleaned_resp = clean_response(response)
    try:
        db_connector.insert_conversation(instance_id, user_input, cleaned_resp)
    except Exception as e:
        logger.error(f"[{instance_id}] Failed to insert conversation to DB: {e}")

    return jsonify({"instance": instance_id, "response": cleaned_resp}), 200

@app.route("/instances", methods=["GET"])
def list_instances():
    return jsonify({"instances": list(agents.keys())})

if __name__ == "__main__":
    num_instances = 3
    for i in range(num_instances):
        agent = start_agent(str(i), db_connector)
        if agent:
            agents[str(i)] = agent
            logger.info(f"Initialized AI instance {i}")
        else:
            logger.error(f"Failed to initialize AI instance {i}")

    logger.info(f"Starting Flask server with {len(agents)} AI instances...")
    app.run(host="0.0.0.0", port=5000, debug=False)
