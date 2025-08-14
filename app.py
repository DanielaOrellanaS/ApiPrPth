# uvicorn app:app --host 0.0.0.0 --port 8000 --reload

from fastapi import FastAPI, Query
from datetime import datetime
import torch
import torch.nn as nn
import pickle
import os
import pandas as pd
import threading

app = FastAPI()

# ========= Configuración por símbolo =========
SYMBOL_CONFIG = {
    "GBPAUD": {
        "model_path": "Trading_Model/trading_model_GBPAUD.pth",
        "minmax_path": "Trading_Model/min_max_GBPAUD.pkl",
        "min_profit": 0.0005,
        "input_size": 20
    },
    "AUDUSD": {
        "model_path": "Trading_Model/trading_model_AUDUSD.pth",
        "minmax_path": "Trading_Model/min_max_AUDUSD.pkl",
        "min_profit": 0.0005,
        "input_size": 20
    },
    "EURUSD": {
        "model_path": "Trading_Model/trading_model_EURUSD.pth",
        "minmax_path": "Trading_Model/min_max_EURUSD.pkl",
        "min_profit": 0.0005,
        "input_size": 20
    },
    "EURCHF": {
        "model_path": "Trading_Model/trading_model_EURCHF.pth",
        "minmax_path": "Trading_Model/min_max_EURCHF.pkl",
        "min_profit": 0.0005,
        "input_size": 20
    },
    "GBPUSD": {
        "model_path": "Trading_Model/trading_model_GBPUSD.pth",
        "minmax_path": "Trading_Model/min_max_GBPUSD.pkl",
        "min_profit": 0.0005,
        "input_size": 20
    },
    "BTCUSD": {
        "model_path": "Trading_Model/trading_model_BTCUSD.pth",
        "minmax_path": "Trading_Model/min_max_BTCUSD.pkl",
        "min_profit": 600,
        "input_size": 20
    },
    "US30": {
        "model_path": "Trading_Model/trading_model_US30.pth",
        "minmax_path": "Trading_Model/min_max_US30.pkl",
        "min_profit": 600,
        "input_size": 42
    },
    "GER40": {
        "model_path": "Trading_Model/trading_model_GER40.pth",
        "minmax_path": "Trading_Model/min_max_GER40.pkl",
        "min_profit": 10,
        "input_size": 42
    },
    "NAS100": {
        "model_path": "Trading_Model/trading_model_NAS100.pth",
        "minmax_path": "Trading_Model/min_max_NAS100.pkl",
        "min_profit": 600,
        "input_size": 42
    }
}

# Locks por símbolo
MODEL_LOCKS = {
    symbol: threading.Lock()
    for symbol in SYMBOL_CONFIG
}

# ========= Modelo base =========
class TradingModel(nn.Module):
    def __init__(self, input_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 64),
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
        model = model = TradingModel(input_size=config["input_size"])
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
    m15: float = Query(...), s15: float = Query(...),

    # Indicadores adicionales (opcional)
    ema550: float = Query(None), ema5200: float = Query(None),
    ema50_prev: float = Query(None), ema5200_prev: float = Query(None),
    macdLine5: float = Query(None), signalLine5: float = Query(None),
    macdLine_prev5: float = Query(None), signalLine_prev5: float = Query(None),
    adx5: float = Query(None), diPlus5: float = Query(None), diMinus5: float = Query(None),
    ema5015: float = Query(None), ema20015: float = Query(None),
    ema50_prev15: float = Query(None), ema200_prev15: float = Query(None),
    macdLine15: float = Query(None), signalLine15: float = Query(None),
    macdLine_prev15: float = Query(None), signalLine_prev15: float = Query(None),
    adx15: float = Query(None), diPlus15: float = Query(None), diMinus15: float = Query(None)
):
    try:
        if symbol not in SYMBOL_CONFIG:
            return {"error": f"Símbolo '{symbol}' no encontrado"}

        config = SYMBOL_CONFIG[symbol]
        model = MODELS.get(symbol)
        min_max = MINMAX.get(symbol)
        lock = MODEL_LOCKS[symbol]

        if model is None or min_max is None:
            return {"error": f"No se pudo cargar modelo o minmax para {symbol}"}

        # Procesar fecha
        dt = datetime.fromisoformat(fecha)
        dia_semana = dt.weekday() / 6.0
        hora = dt.hour / 23.0
        minuto = dt.minute / 55.0

        # Inputs base (20)
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

        # Si el modelo requiere más de 20 entradas, validar y agregar
        if config["input_size"] > 20:
            extra = [ema550, ema5200, ema50_prev, ema5200_prev,
                     macdLine5, signalLine5, macdLine_prev5, signalLine_prev5,
                     adx5, diPlus5, diMinus5,
                     ema5015, ema20015, ema50_prev15, ema200_prev15,
                     macdLine15, signalLine15, macdLine_prev15, signalLine_prev15,
                     adx15, diPlus15, diMinus15]

            if any(v is None for v in extra):
                return {"error": f"Faltan parámetros extendidos para el símbolo '{symbol}'"}

            extra_data = [
                normalize(ema550, min_max["min_ema550"], min_max["max_ema550"]),
                normalize(ema5200, min_max["min_ema5200"], min_max["max_ema5200"]),
                normalize(ema50_prev, min_max["min_ema50_prev"], min_max["max_ema50_prev"]),
                normalize(ema5200_prev, min_max["min_ema5200_prev"], min_max["max_ema5200_prev"]),
                normalize(macdLine5, min_max["min_macdLine5"], min_max["max_macdLine5"]),
                normalize(signalLine5, min_max["min_signalLine5"], min_max["max_signalLine5"]),
                normalize(macdLine_prev5, min_max["min_macdLine_prev5"], min_max["max_macdLine_prev5"]),
                normalize(signalLine_prev5, min_max["min_signalLine_prev5"], min_max["max_signalLine_prev5"]),
                adx5 / 100.0,
                diPlus5 / 100.0,
                diMinus5 / 100.0,
                normalize(ema5015, min_max["min_ema5015"], min_max["max_ema5015"]),
                normalize(ema20015, min_max["min_ema20015"], min_max["max_ema20015"]),
                normalize(ema50_prev15, min_max["min_ema50_prev15"], min_max["max_ema50_prev15"]),
                normalize(ema200_prev15, min_max["min_ema200_prev15"], min_max["max_ema200_prev15"]),
                normalize(macdLine15, min_max["min_macdLine15"], min_max["max_macdLine15"]),
                normalize(signalLine15, min_max["min_signalLine15"], min_max["max_signalLine15"]),
                normalize(macdLine_prev15, min_max["min_macdLine_prev15"], min_max["max_macdLine_prev15"]),
                normalize(signalLine_prev15, min_max["min_signalLine_prev15"], min_max["max_signalLine_prev15"]),
                adx15 / 100.0,
                diPlus15 / 100.0,
                diMinus15 / 100.0
            ]
            input_data.extend(extra_data)

        # Predicción
        input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0)
        with lock:
            with torch.no_grad():
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
    # Comprobacion 
    return {"status": "ok"}
