from fastapi import FastAPI, Query
from datetime import datetime
import torch
import torch.nn as nn
import pickle

app = FastAPI()

input_columns = [
    'dia_semana', 'hora', 'minuto',
    "precioopen5", "precioclose5", "precioclose5_dup",
    "preciohigh5", "preciolow5", "volume5",
    "precioopen15", "precioclose15", "preciohigh15", "preciolow15", "volume15",
    "rsi5", "rsi15", "iStochaMain5", "iStochaSign5", "iStochaMain15", "iStochaSign15"
]

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

model_path = "Trading_Model/trading_model_GBPAUD.pth"
model = TradingModel()
model.load_state_dict(torch.load(model_path))
model.eval()

with open("Trading_Model/min_max_GBPAUD.pkl", "rb") as f:
    min_max = pickle.load(f)

min_profit = min_max["min_profit"]
max_profit = min_max["max_profit"]

MINIMO_GLOBAL = 0.0005
MINUTO_MAX = 55.0
HORA_MAX = 23.0
DIA_SEMANA_MAX = 6.0

def denormalize(value, min_val, max_val):
    return value * (max_val - min_val) + min_val

def calcular_operacion(profit, minimo):
    if abs(profit) > minimo:
        return 'BUY' if profit > 0 else 'SELL'
    else:
        return 'NO_SIGNAL'

@app.get("/")
def home():
    return {"message": "API funcionando correctamente"}

@app.get("/predict_get")
def predict_get(
    fecha: str = Query(..., description="Fecha y hora en formato 'YYYY-MM-DD HH:MM:SS' o ISO"),
    precioopen5: float = Query(...),
    precioclose5: float = Query(...),
    precioclose5_dup: float = Query(...),
    preciohigh5: float = Query(...),
    preciolow5: float = Query(...),
    volume5: float = Query(...),
    precioopen15: float = Query(...),
    precioclose15: float = Query(...),
    preciohigh15: float = Query(...),
    preciolow15: float = Query(...),
    volume15: float = Query(...),
    rsi5: float = Query(...),
    rsi15: float = Query(...),
    iStochaMain5: float = Query(...),
    iStochaSign5: float = Query(...),
    iStochaMain15: float = Query(...),
    iStochaSign15: float = Query(...)
):
    # Convertir fecha a datetime
    try:
        dt = datetime.fromisoformat(fecha)
    except ValueError:
        dt = datetime.strptime(fecha, "%Y-%m-%d %H:%M:%S")

    # Extraer y normalizar
    dia_semana = dt.weekday() / DIA_SEMANA_MAX
    hora = dt.hour / HORA_MAX
    minuto = dt.minute / MINUTO_MAX

    features = [
        dia_semana, hora, minuto,
        precioopen5, precioclose5, precioclose5_dup,
        preciohigh5, preciolow5, volume5,
        precioopen15, precioclose15, preciohigh15, preciolow15, volume15,
        rsi5, rsi15, iStochaMain5, iStochaSign5, iStochaMain15, iStochaSign15
    ]

    x = torch.tensor([features], dtype=torch.float32)
    with torch.no_grad():
        raw_output = model(x).item()

    profit = denormalize(raw_output, min_profit, max_profit)
    tipo = calcular_operacion(profit, MINIMO_GLOBAL)

    return {
        "raw_output": raw_output,
        "profit_prediction": profit,
        "tipo_operacion": tipo
    }
