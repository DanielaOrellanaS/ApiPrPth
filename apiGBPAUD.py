# uvicorn apiGBPAUD:app --host 0.0.0.0 --port 8000 --reload

from fastapi import FastAPI, Query
from datetime import datetime
import torch
import torch.nn as nn
import pickle
import os

app = FastAPI()

# Modelo PyTorch
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

# Cargar modelo y min/max
model = TradingModel()
model.load_state_dict(torch.load("Trading_Model/trading_model_GBPAUD.pth"))
model.eval()

with open("Trading_Model/min_max_GBPAUD.pkl", "rb") as f:
    min_max = pickle.load(f)

def normalize(val, min_val, max_val):
    return (val - min_val) / (max_val - min_val)

def denormalize(val, min_val, max_val):
    return val * (max_val - min_val) + min_val

def calcular_operacion(profit, minimo=0.0004):
    if abs(profit) > minimo:
        return "BUY" if profit > 0 else "SELL"
    return "NO_SIGNAL"

@app.get("/predict")
def predict(
    symbol: str,
    fecha: str,
    o5: float = Query(..., alias="o5"), c5: float = Query(..., alias="c5"),
    h5: float = Query(..., alias="h5"), l5: float = Query(..., alias="l5"), v5: float = Query(..., alias="v5"),
    o15: float = Query(..., alias="o15"), c15: float = Query(..., alias="c15"),
    h15: float = Query(..., alias="h15"), l15: float = Query(..., alias="l15"), v15: float = Query(..., alias="v15"),
    r5: float = Query(..., alias="r5"), r15: float = Query(..., alias="r15"),
    m5: float = Query(..., alias="m5"), s5: float = Query(..., alias="s5"),
    m15: float = Query(..., alias="m15"), s15: float = Query(..., alias="s15")
):
    try:
        # Procesar fecha
        dt = datetime.fromisoformat(fecha)
        dia_semana = dt.weekday() / 6.0
        hora = dt.hour / 23.0
        minuto = dt.minute / 55.0

        # Normalizar precios 5min
        n_precioopen5 = normalize(o5, min_max["min_precio5"], min_max["max_precio5"])
        n_precioclose5 = normalize(c5, min_max["min_precio5"], min_max["max_precio5"])
        n_preciohigh5 = normalize(h5, min_max["min_precio5"], min_max["max_precio5"])
        n_preciolow5 = normalize(l5, min_max["min_precio5"], min_max["max_precio5"])
        n_volume5 = normalize(v5, min_max["min_volume5"], min_max["max_volume5"])

        # Normalizar precios 15min
        n_precioopen15 = normalize(o15, min_max["min_precio15"], min_max["max_precio15"])
        n_precioclose15 = normalize(c15, min_max["min_precio15"], min_max["max_precio15"])
        n_preciohigh15 = normalize(h15, min_max["min_precio15"], min_max["max_precio15"])
        n_preciolow15 = normalize(l15, min_max["min_precio15"], min_max["max_precio15"])
        n_volume15 = normalize(v15, min_max["min_volume15"], min_max["max_volume15"])

        # Normalizar RSI y stochastic (escala 0-1)
        n_rsi5 = r5 / 100.0
        n_rsi15 = r15 / 100.0
        n_iStochaMain5 = m5 / 100.0
        n_iStochaSign5 = s5 / 100.0
        n_iStochaMain15 = m15 / 100.0
        n_iStochaSign15 = s15 / 100.0
        # Construir vector input (20 features)
        input_data = [
            dia_semana, hora, minuto,
            n_precioopen5, n_precioclose5, n_precioclose5,  # duplicado para dar más peso
            n_preciohigh5, n_preciolow5, n_volume5,
            n_precioopen15, n_precioclose15, n_preciohigh15, n_preciolow15, n_volume15,
            n_rsi5, n_rsi15, n_iStochaMain5, n_iStochaSign5, n_iStochaMain15, n_iStochaSign15
        ]

        input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            raw_output = model(input_tensor).item()

        profit = denormalize(raw_output, min_max["min_profit"], min_max["max_profit"])
        tipo = calcular_operacion(profit)

        # Opcional: guardar predicción (puedes comentar o quitar esta parte si no quieres guardarla)
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
            import pandas as pd
            df = pd.read_excel(file_path)
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        else:
            import pandas as pd
            df = pd.DataFrame([row])
        df.to_excel(file_path, index=False)

        return {
            #"input_normalizado": input_data,
            "valor_profit": profit,
            "RESULTADO": tipo
        }

    except Exception as e:
        return {"error": str(e)}
