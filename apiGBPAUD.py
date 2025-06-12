import torch
import torch.nn as nn
import pickle
from fastapi import FastAPI, Request
import pandas as pd
from datetime import datetime

app = FastAPI()


# ========= Normalización =========
def normalize(value, min_val, max_val):
    return (value - min_val) / (max_val - min_val)

def denormalize(value, min_val, max_val):
    return value * (max_val - min_val) + min_val

# ========= Configuraciones =========
MINIMO_GLOBAL = 0.0005

# ========= Modelo =========
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

# Cargar modelo
model = TradingModel()
model.load_state_dict(torch.load("Trading_Model/trading_model_GBPAUD.pth"))
model.eval()

# Cargar min/max
with open("Trading_Model/min_max_GBPAUD.pkl", "rb") as f:
    min_max = pickle.load(f)

# ========= Función principal para 1 registro =========
def predecir_registro(data: dict):
    # Procesar fecha
    fecha = datetime.strptime(data['fecha'], "%Y-%m-%d %H:%M:%S")
    dia_semana = fecha.weekday() / 6.0
    hora = fecha.hour / 23.0
    minuto = fecha.minute / 55.0

    # Normalización precios 5min
    open5 = normalize(data['precioopen5'], min_max['min_precio5'], min_max['max_precio5'])
    close5 = normalize(data['precioclose5'], min_max['min_precio5'], min_max['max_precio5'])
    high5 = normalize(data['preciohigh5'], min_max['min_precio5'], min_max['max_precio5'])
    low5 = normalize(data['preciolow5'], min_max['min_precio5'], min_max['max_precio5'])
    volume5 = normalize(data['volume5'], data['volume5_min'], data['volume5_max'])  # puedes ajustar esto

    # Normalización precios 15min
    open15 = normalize(data['precioopen15'], min_max['min_precio15'], min_max['max_precio15'])
    close15 = normalize(data['precioclose15'], min_max['min_precio15'], min_max['max_precio15'])
    high15 = normalize(data['preciohigh15'], min_max['min_precio15'], min_max['max_precio15'])
    low15 = normalize(data['preciolow15'], min_max['min_precio15'], min_max['max_precio15'])
    volume15 = normalize(data['volume15'], data['volume15_min'], data['volume15_max'])  # puedes ajustar esto

    # Normalización indicadores
    rsi5 = data['rsi5'] / 100.0
    rsi15 = data['rsi15'] / 100.0
    stocha_main5 = data['iStochaMain5'] / 100.0
    stocha_sign5 = data['iStochaSign5'] / 100.0
    stocha_main15 = data['iStochaMain15'] / 100.0
    stocha_sign15 = data['iStochaSign15'] / 100.0

    # Vector de entrada
    input_vector = torch.tensor([[
        dia_semana, hora, minuto,
        open5, close5, close5,  # repite close5 como en el modelo original
        high5, low5, volume5,
        open15, close15, high15, low15, volume15,
        rsi5, rsi15, stocha_main5, stocha_sign5, stocha_main15, stocha_sign15
    ]], dtype=torch.float32)

    # Predicción
    with torch.no_grad():
        pred_raw = model(input_vector).item()
    profit_pred = denormalize(pred_raw, min_max['min_profit'], min_max['max_profit'])

    # Clasificación
    if abs(profit_pred) > MINIMO_GLOBAL:
        tipo = 'BUY' if profit_pred > 0 else 'SELL'
    else:
        tipo = 'NO_SIGNAL'

    return {
        "profit_prediction": round(profit_pred, 6),
        "tipo_prediction": tipo
    }

def map_get_params_to_registro(params):
    registro = {
        "fecha": params["fecha"].replace("T", " "),
        "precioopen5": float(params["o5"]),
        "precioclose5": float(params["c5"]),
        "preciohigh5": float(params["h5"]),
        "preciolow5": float(params["l5"]),
        "volume5": float(params["v5"]),
        "volume5_min": 200,
        "volume5_max": 1200,
        "precioopen15": float(params["o15"]),
        "precioclose15": float(params["c15"]),
        "preciohigh15": float(params["h15"]),
        "preciolow15": float(params["l15"]),
        "volume15": float(params["v15"]),
        "volume15_min": 500,
        "volume15_max": 3000,
        "rsi5": float(params["r5"]) * 100,
        "rsi15": float(params["r15"]) * 100,
        "iStochaMain5": float(params["m5"]) * 100,
        "iStochaSign5": float(params["s5"]) * 100,
        "iStochaMain15": float(params["m15"]) * 100,
        "iStochaSign15": float(params["s15"]) * 100,
    }
    return registro

@app.get("/predict_get_with_file")
async def predict_get(request: Request):
    params = request.query_params
    registro = map_get_params_to_registro(params)
    resultado = predecir_registro(registro)  
    return resultado


