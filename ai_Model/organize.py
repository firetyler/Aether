import os
import shutil

base = os.getcwd()
src_root = os.path.join(base, "ai_Model")

folders = {
    "archive": [],
    "chat_dataset": ["chatDataset"],
    "data": [],
    "database": ["DatabaseConnector"],
    "flask": ["flask_ai.py", "EndPoints.py"],
    "inference": [],
    "transformer": ["stacktTransformer"],
    "tokenizer": ["tockeniser", "SimpleTokenizer.py"],
    "training": ["Traning"],
    "utils": [
        "EndPointLog.py",
        "logger_setup.py",
        "loggerDatabase.py",
        "logToken.py",
        "logTrainer.py",
        "AetherMemoryLog.py"
    ],
    "vocab": ["__init__.py"],
    "extra": [
        "aether2.py", "AetherMemory.py", "AetherMemoryLog.py", "AiClone.py",
        "code_executor.py", "mask_utils.py"
    ],
    "config": [  # all .json and .properties
        "*.json", "*.properties"
    ]
}

def make_init(path):
    os.makedirs(path, exist_ok=True)
    init = os.path.join(path, "__init__.py")
    if not os.path.exists(init):
        open(init, "w").close()

def move_file(file_path, dest_folder):
    # Construct the full destination path including filename
    dest_path = os.path.join(dest_folder, os.path.basename(file_path))

    # If the destination file exists, remove it first
    if os.path.exists(dest_path):
        os.remove(dest_path)

    # Now move the file
    shutil.move(file_path, dest_path)

    for folder, items in folders.items():
        target_path = os.path.join(src_root, folder)
        make_init(target_path)

        for item in items:
            abs_path = os.path.join(src_root, item)
            if os.path.isdir(abs_path):
                shutil.move(abs_path, target_path)
            elif "*" in item:
                ext = item.split("*.")[-1]
                for f in os.listdir(src_root):
                    if f.endswith(f".{ext}"):
                        move_file(os.path.join(src_root, f), target_path)
            else:
                move_file(abs_path, target_path)

print("✅ Alla filer har flyttats till rätt plats!")
