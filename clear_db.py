import shutil
import os

# Clear ChromaDB data
db_path = "./data/chroma_db"
if os.path.exists(db_path):
    shutil.rmtree(db_path)
    print(f"Cleared database at {db_path}")
else:
    print("No existing database found")

# Recreate directory
os.makedirs(db_path, exist_ok=True)
print("Database directory recreated")