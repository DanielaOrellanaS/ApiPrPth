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
        raw_output = model(x).item()

    profit = denormalize(raw_output, min_profit, max_profit)
    tipo = calcular_operacion(profit, MINIMO_GLOBAL)

    print(f"Respuesta modelo: raw_output={raw_output}, profit={profit}, tipo={tipo}")

    return {
        "raw_output": raw_output,
        "profit_prediction": profit,
        "tipo_operacion": tipo
    }
