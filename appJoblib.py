from fastapi import FastAPI, HTTPException, Request
import pandas as pd
import joblib
import os

app = FastAPI(title="BTC LogisticRegression API", version="1.0")

MODEL_PATH = "btc_latest.joblib"
model = None
ORIGINAL_COLUMNS = None
CLEAN_COLUMNS = None
COL_MAP = None

@app.on_event("startup")
def load_model():
    global model, ORIGINAL_COLUMNS, CLEAN_COLUMNS, COL_MAP
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"No se encontró el modelo: {MODEL_PATH}")

    model = joblib.load(MODEL_PATH)

    if not hasattr(model, "feature_names_in_"):
        raise ValueError("El modelo no tiene información de columnas de entrada.")

    ORIGINAL_COLUMNS = list(model.feature_names_in_)
    CLEAN_COLUMNS = [c.strip() for c in ORIGINAL_COLUMNS]
    COL_MAP = {clean: orig for clean, orig in zip(CLEAN_COLUMNS, ORIGINAL_COLUMNS)}

    print("✅ Modelo cargado.")
    print("Original:", ORIGINAL_COLUMNS)
    print("Limpias :", CLEAN_COLUMNS)


@app.get("/predict")
def predict(request: Request):
    try:
        # obtener todos los parámetros de la query
        kwargs = dict(request.query_params)

        # limpiar nombres
        cleaned_kwargs = {k.strip(): v for k, v in kwargs.items()}

        # validar columnas faltantes
        missing = [col for col in CLEAN_COLUMNS if col not in cleaned_kwargs]
        if missing:
            raise HTTPException(status_code=400, detail=f"Faltan columnas: {missing}")

        # construir fila en orden
        row = []
        for col in CLEAN_COLUMNS:
            val = cleaned_kwargs[col]
            try:
                row.append(float(val))
            except Exception:
                raise HTTPException(status_code=400, detail=f"El parámetro '{col}' no es numérico: '{val}'")

        df = pd.DataFrame([row], columns=CLEAN_COLUMNS)
        df.columns = [COL_MAP[c] for c in CLEAN_COLUMNS]

        pred = model.predict(df)[0]
        proba = model.predict_proba(df)[0].tolist()

        return {"prediction": int(pred), "probabilities": proba}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
