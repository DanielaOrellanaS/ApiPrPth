# uvicorn app:app --host 0.0.0.0 --port 8000 --reload

from fastapi import FastAPI, Query
from datetime import datetime
import torch
import torch.nn as nn
import pickle
import os
import pandas as pd

app = FastAPI()

# ========= Configuración por símbolo =========
SYMBOL_CONFIG = {
    "GBPAUD": {
        "model_path": "Trading_Model/trading_model_GBPAUD.pth",
        "minmax_path": "Trading_Model/min_max_GBPAUD.pkl",
        "min_profit": 0.0004
    },
    "AUDUSD": {
        "model_path": "Trading_Model/trading_model_AUDUSD.pth",
        "minmax_path": "Trading_Model/min_max_AUDUSD.pkl",
        "min_profit": 0.0005
    },
    "EURUSD": {
        "model_path": "Trading_Model/trading_model_EURUSD.pth",
        "minmax_path": "Trading_Model/min_max_EURUSD.pkl",
        "min_profit": 0.0005
    },
    "GBPUSD": {
        "model_path": "Trading_Model/trading_model_GBPUSD.pth",
        "minmax_path": "Trading_Model/min_max_GBPUSD.pkl",
        "min_profit": 0.0005
    }
}

# ========= Modelo base =========
class TradingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(20, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)

# ========= Funciones auxiliares =========
def normalize(val, min_val, max_val):
    return (val - min_val) / (max_val - min_val)

def denormalize(val, min_val, max_val):
    return val * (max_val - min_val) + min_val

def calcular_operacion(profit, minimo):
    if abs(profit) > minimo:
        return "BUY" if profit > 0 else "SELL"
    return "NADA"

# ========= Endpoint principal =========
@app.get("/predict")
def predict(
    symbol: str,
    fecha: str,
    o5: float = Query(...), c5: float = Query(...),
    h5: float = Query(...), l5: float = Query(...), v5: float = Query(...),
    o15: float = Query(...), c15: float = Query(...),
    h15: float = Query(...), l15: float = Query(...), v15: float = Query(...),
    r5: float = Query(...), r15: float = Query(...),
    m5: float = Query(...), s5: float = Query(...),
    m15: float = Query(...), s15: float = Query(...)
):
    try:
        if symbol not in SYMBOL_CONFIG:
            return {"error": f"Símbolo '{symbol}' no encontrado"}

        config = SYMBOL_CONFIG[symbol]

        # Cargar modelo
        model = TradingModel()
        model.load_state_dict(torch.load(config["model_path"]))
        model.eval()

        # Cargar min/max
        with open(config["minmax_path"], "rb") as f:
            min_max = pickle.load(f)

        # Procesar fecha
        dt = datetime.fromisoformat(fecha)
        dia_semana = dt.weekday() / 6.0
        hora = dt.hour / 23.0
        minuto = dt.minute / 55.0

        # Normalizar inputs
        input_data = [
            dia_semana, hora, minuto,
            normalize(o5, min_max["min_precio5"], min_max["max_precio5"]),
            normalize(c5, min_max["min_precio5"], min_max["max_precio5"]),
            normalize(c5, min_max["min_precio5"], min_max["max_precio5"]),
            normalize(h5, min_max["min_precio5"], min_max["max_precio5"]),
            normalize(l5, min_max["min_precio5"], min_max["max_precio5"]),
            normalize(v5, min_max["min_volume5"], min_max["max_volume5"]),
            normalize(o15, min_max["min_precio15"], min_max["max_precio15"]),
            normalize(c15, min_max["min_precio15"], min_max["max_precio15"]),
            normalize(h15, min_max["min_precio15"], min_max["max_precio15"]),
            normalize(l15, min_max["min_precio15"], min_max["max_precio15"]),
            normalize(v15, min_max["min_volume15"], min_max["max_volume15"]),
            r5 / 100.0, r15 / 100.0,
            m5 / 100.0, s5 / 100.0,
            m15 / 100.0, s15 / 100.0
        ]

        input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0)
        raw_output = model(input_tensor).item()
        profit = denormalize(raw_output, min_max["min_profit"], min_max["max_profit"])
        tipo = calcular_operacion(profit, config["min_profit"])

        # Guardar predicción
        save_dir = "Predicciones"
        os.makedirs(save_dir, exist_ok=True)
        file_path = os.path.join(save_dir, f"save_predictions_{symbol}.xlsx")

        row = {
            "timestamp": fecha, "symbol": symbol, "tipo": tipo, "profit": profit,
            "precioopen5": o5, "precioclose5": c5, "preciohigh5": h5, "preciolow5": l5, "volume5": v5,
            "precioopen15": o15, "precioclose15": c15, "preciohigh15": h15, "preciolow15": l15, "volume15": v15,
            "rsi5": r5, "rsi15": r15,
            "iStochaMain5": m5, "iStochaSign5": s5, "iStochaMain15": m15, "iStochaSign15": s15
        }

        if os.path.exists(file_path):
            df = pd.read_excel(file_path)
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        else:
            df = pd.DataFrame([row])
        df.to_excel(file_path, index=False)

        return {
            "valor_profit": profit,
            "RESULTADO": tipo
        }

    except Exception as e:
        return {"error": str(e)}
