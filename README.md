# ♻️ RecicláIA — Clasificación de residuos con IA

App web que clasifica la foto de un residuo en una de 6 categorías (cartón, vidrio, metal, papel, plástico o resto) y entrega una **recomendación de reciclaje generada por IA**.

**Machine Learning · Universidad del Desarrollo · Certamen I**
**Grupo 1:** Tomás Allende · Felipe Castillo · Juan Costa

🔗 **App publicada:** _(pegar aquí el link de Streamlit cuando esté)_

## 🧠 ¿Cómo funciona?

📷 Imagen → 🧠 EfficientNet-B0 (clase + confianza) → 📊 Probabilidades → 💬 Gemini (LLM) → ✅ Recomendación

- **Modelo:** EfficientNet-B0 (librería `timm`), ajustado con *transfer learning*.
- **Dataset:** [TrashNet](https://huggingface.co/datasets/garythung/trashnet) (Gary Thung y Mindy Yang, Stanford, 2016 · licencia MIT). 2.527 fotos, 6 clases, divididas 70/15/15 (train/val/test) de forma estratificada.
- **Comparación:** se comparó contra CLIP sin entrenar (línea base). EfficientNet (~80% de accuracy en prueba) supera claramente a CLIP (~65%).
- **App:** Streamlit, publicada en Streamlit Community Cloud.
- **LLM:** Google Gemini (API gratuita), vía el SDK `google-genai`. Solo se le envía texto (clase y confianza), no la imagen.

## 📁 Estructura

```
recicla-ia/
├── app.py                        # la aplicación (Streamlit)
├── requirements.txt              # librerías necesarias
├── README.md                     # este archivo
├── .streamlit/config.toml        # tema de la app (colores)
├── notebooks/
│   ├── entrenamiento.ipynb            # notebook para ejecutar
│   └── entrenamiento_ejecutado.ipynb  # notebook ya ejecutado con resultados
├── modelo/
│   ├── modelo.pt                 # pesos entrenados
│   ├── clases.txt                # clases en el orden del modelo
│   └── config.json               # arquitectura + accuracy
└── ejemplos/                     # imágenes de prueba
```

## 🔑 Clave del LLM (secreto)

La clave **nunca** se sube al repositorio. Se guarda como secreto de la plataforma:
- **Streamlit Community Cloud:** al desplegar, *Advanced settings → Secrets*, agregar:
  ```
  GEMINI_API_KEY = "tu_clave"
  ```
- **Local:** crear `.streamlit/secrets.toml` con esa misma línea (ese archivo está en `.gitignore`), o definir la variable de entorno `GEMINI_API_KEY`.

Clave gratuita en [Google AI Studio](https://aistudio.google.com/apikey).

## ▶️ Ejecutar localmente

```
pip install -r requirements.txt
streamlit run app.py
```

## 🚀 Publicar en Streamlit Community Cloud

1. Subir este repositorio a GitHub (público).
2. Entrar a [share.streamlit.io](https://share.streamlit.io) con la cuenta de GitHub.
3. **Create app → From repo**: elegir el repositorio, rama `main` y archivo `app.py`.
4. En *Advanced settings → Secrets*, pegar `GEMINI_API_KEY = "tu_clave"`.
5. **Deploy**. Cada cambio que suban a GitHub actualiza la app automáticamente.

## 🤖 Uso de IA

Usamos asistentes de IA (Claude) para ayudar a escribir y ordenar el código; el equipo adaptó, probó y entiende cada parte del notebook y de `app.py`.

## ⚠️ Aviso

Prototipo educativo. La clasificación es orientativa y puede equivocarse, sobre todo con fotos reales. Ante la duda, revisar manualmente y consultar las reglas de reciclaje de la comuna.
