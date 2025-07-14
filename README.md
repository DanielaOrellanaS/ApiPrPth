# API de Predicción de Trading con PyTorch

API REST para predecir señales de trading (BUY, SELL, NADA) y profit esperado, usando modelos entrenados en PyTorch.  
Está desplegada en Render y disponible en: https://apiprpth.onrender.com/

---

## 📂 Estructura principal

- `app.py`: Código principal de la API con FastAPI y la lógica para predicción.
- `Trading_Model/`: Carpeta con modelos `.pth` y archivos `.pkl` con parámetros de normalización.
- `.gitignore`: Archivo para ignorar archivos innecesarios.
- `requirements.txt`: Dependencias para instalar.

---

## 🚀 Cómo usar

### Ejecutar localmente

```bash
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

---

## 🎯 Endpoint principal `/predict`

URL base:  
- Local: `http://localhost:8000/predict`  
- Render: `https://apiprpth.onrender.com/predict`

### Parámetros GET

| Parámetro | Descripción                               | Tipo    | Obligatorio |
|-----------|-------------------------------------------|---------|-------------|
| symbol    | Símbolo del par (e.g. GBPAUD, AUDUSD)     | string  | Sí          |
| fecha     | Fecha en formato ISO (YYYY-MM-DDTHH:MM)   | string  | Sí          |
| o5        | Precio apertura 5 minutos                 | float   | Sí          |
| c5        | Precio cierre 5 minutos                   | float   | Sí          |
| h5        | Precio máximo 5 minutos                   | float   | Sí          |
| l5        | Precio mínimo 5 minutos                   | float   | Sí          |
| v5        | Volumen 5 minutos                         | float   | Sí          |
| o15       | Precio apertura 15 minutos                | float   | Sí          |
| c15       | Precio cierre 15 minutos                  | float   | Sí          |
| h15       | Precio máximo 15 minutos                  | float   | Sí          |
| l15       | Precio mínimo 15 minutos                  | float   | Sí          |
| v15       | Volumen 15 minutos                        | float   | Sí          |
| r5        | RSI 5 minutos (0-100)                     | float   | Sí          |
| r15       | RSI 15 minutos (0-100)                    | float   | Sí          |
| m5        | Stochastic Main 5 minutos (0-100)         | float   | Sí          |
| s5        | Stochastic Signal 5 minutos (0-100)       | float   | Sí          |
| m15       | Stochastic Main 15 minutos (0-100)        | float   | Sí          |
| s15       | Stochastic Signal 15 minutos (0-100)      | float   | Sí          |

---

### Respuesta JSON exitosa

```json
{
  "valor_profit": 0.001234,
  "RESULTADO": "BUY"
}
```

- `valor_profit`: Valor numérico estimado del profit esperado.
- `RESULTADO`: Señal generada por el modelo, puede ser `"BUY"`, `"SELL"` o `"NADA"`.

---

### Respuesta JSON en caso de error

```json
{
  "error": "Descripción del error"
}
```

---

## ⚙️ Detalles técnicos

- Los modelos PyTorch y parámetros de normalización se cargan al iniciar la API para mejorar tiempos de respuesta.
- La normalización de datos se realiza internamente usando los parámetros guardados (`min` y `max`) para cada símbolo.
- La API recibe los parámetros por query string, normaliza internamente, ejecuta la predicción y desnormaliza el resultado para entregar el profit esperado.
- El cálculo de la señal ("BUY", "SELL" o "NADA") depende de un umbral mínimo configurado por símbolo (por defecto 0.0005).
- La API tiene un endpoint `/ping` para verificar que está activa y funcionando.

---

## 📝 Comandos útiles

- Ejecutar API localmente con autoreload:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

- Probar endpoint local (ejemplo):

```
http://localhost:8000/predict?symbol=GBPAUD&fecha=2025-07-14T08:30&o5=1.2345&c5=1.2350&h5=1.2360&l5=1.2330&v5=1000&o15=1.2300&c15=1.2350&h15=1.2370&l15=1.2290&v15=1500&r5=50&r15=60&m5=40&s5=30&m15=45&s15=35
```

---