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
        "min_profit": 0.0005
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

# ========= Cargar modelos y minmax una vez =========
MODELS = {}
MINMAX = {}

for symbol, config in SYMBOL_CONFIG.items():
    try:
        model = TradingModel()
        model.load_state_dict(torch.load(config["model_path"]))
        model.eval()
        MODELS[symbol] = model

        with open(config["minmax_path"], "rb") as f:
            MINMAX[symbol] = pickle.load(f)

        print(f"✅ Cargado {symbol}")
    except Exception as e:
        print(f"❌ Error cargando {symbol}: {e}")

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
        model = MODELS.get(symbol)
        min_max = MINMAX.get(symbol)

        if model is None or min_max is None:
            return {"error": f"No se pudo cargar modelo o minmax para {symbol}"}

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
        
        return {
            "valor_profit": profit,
            "RESULTADO": tipo
        }

    except Exception as e:
        return {"error": str(e)}

# ========= Endpoint ping =========
@app.get("/ping")
def ping():
    return {"status": "ok"}
