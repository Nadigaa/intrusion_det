from fastapi import FastAPI, UploadFile, File
import pandas as pd
import joblib
import io

app = FastAPI(title="Zeek Attack Classifier API")

# Load model
model = joblib.load("models/zeek_attack_model.pkl")

# ----------------------------------------------------
# MODEL FEATURES (from train_zeek_ids.py)
# ----------------------------------------------------
MODEL_FEATURES = [
    "duration_x", "orig_bytes", "resp_bytes",
    "orig_pkts", "resp_pkts", "orig_ip_bytes",
    "resp_ip_bytes", "trans_depth",
    "id.orig_p_x", "id.resp_p_x",
    "ts", "ts_x"
]

@app.get("/")
def root():
    return {"message": "Zeek-based Attack Classifier. Use /predict_csv to upload a CSV."}

# ----------------------------------------------------
# Predict from CSV endpoint
# ----------------------------------------------------
@app.post("/predict_csv")
async def predict_csv(file: UploadFile = File(...)):
    try:
        # ----- STREAMING READ: read only first 20k rows -----
        chunk_iter = pd.read_csv(
            file.file,
            chunksize=5000,       # process 5k rows per chunk
            low_memory=False,
            dtype=str             # avoid mixed-type warnings
        )

        combined = []            # only store what we need (first 50)
        
        for chunk in chunk_iter:
            # Convert important numeric columns
            numeric_cols = [
                "duration_x","orig_bytes","resp_bytes",
                "orig_pkts","resp_pkts","orig_ip_bytes","resp_ip_bytes",
                "trans_depth","id.orig_p_x","id.resp_p_x",
                "ts","ts_x"
            ]
            for col in numeric_cols:
                if col in chunk.columns:
                    chunk[col] = pd.to_numeric(chunk[col], errors="coerce")
            
            # Predict
            preds = model.predict(chunk)
            chunk["attack_type"] = preds

            # Collect only up to 50 rows
            combined.append(chunk)

            # Stop after enough rows
            if sum(len(c) for c in combined) >= 50:
                break
        
        # Combine only processed chunks
        result = pd.concat(combined)
        result = result.head(50)

        # Replace problematic values
        result = result.replace([float("inf"), float("-inf")], None)
        result = result.where(pd.notnull(result), None)

        return {
            "note": "Showing only first 50 results (streamed).",
            "rows_returned": len(result),
            "preview": result.astype(str).to_dict(orient="records")
        }

    except Exception as e:
        print("\n[ERROR] Exception in /predict_csv:", str(e))
        return {"error": str(e)}
