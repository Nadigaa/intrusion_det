import pandas as pd
import requests
import json

API_URL = "http://127.0.0.1:8000/predict_csv"
CSV_FILE = "attack-add.csv"
CHUNK_SIZE = 50

print("Loading large CSV locally...")
df = pd.read_csv(CSV_FILE, low_memory=False)
total_rows = len(df)
print(f"Total rows: {total_rows}\n")

all_results = []

for start in range(0, total_rows, CHUNK_SIZE):
    end = start + CHUNK_SIZE
    chunk = df.iloc[start:end]

    print(f"Processing rows {start} to {min(end, total_rows)}...")

    # Export chunk to temp CSV
    chunk.to_csv("temp_chunk.csv", index=False)

    # Upload to FastAPI
    with open("temp_chunk.csv", "rb") as f:
        response = requests.post(API_URL, files={"file": f})

    if response.status_code == 200:
        result = response.json()
        all_results.extend(result["preview"])  # only 50 rows returned
    else:
        print("Error:", response.text)
        break

# Save all combined predictions
with open("predictions_all.json", "w") as f:
    json.dump(all_results, f, indent=4)

print("\n🎉 Finished!")
print(f"Total predictions collected: {len(all_results)}")
print("Saved combined output to predictions_all.json")
