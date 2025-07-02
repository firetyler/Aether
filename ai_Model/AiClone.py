import multiprocessing
from ai_Model.aether2 import AetherAgent
#from aether2 import AetherAgent
from ai_Model.database.DatabaseConnector import DatabaseConnector

# ✅ Shared database connection
db_connector = DatabaseConnector()

def start_agent(instance_id):
    """Creates and runs an AI instance."""
    agent = AetherAgent(db_connector)
    print(f"AI instance {instance_id} started.")

    while True:
        user_input = input(f"({instance_id}) You: ")
        if user_input.lower() in ["exit", "quit"]:
            print(f"AI instance {instance_id} shutting down...")
            break

        response = agent.run(user_input)
        print(f"({instance_id}) AI: {response}")

if __name__ == "__main__":
    num_instances = 3  # 🔹 Number of AI clones to start
    processes = []

    for i in range(num_instances):
        p = multiprocessing.Process(target=start_agent, args=(i,))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
