# ReciclaIA - Clasificacion de residuos con IA

Aplicacion web que clasifica la foto de un residuo en una de 6 categorias (carton, vidrio, metal, papel, plastico o resto) y entrega una recomendacion de reciclaje generada por IA.

Machine Learning - Universidad del Desarrollo - Certamen I
Grupo 1: Tomas Allende - Felipe Castillo - Juan Costa

App publicada: _(pegar aqui el link de Streamlit cuando este)_

## Como funciona

Imagen -> EfficientNet-B0 (clase + confianza) -> probabilidades -> Gemini (LLM) -> recomendacion

- Modelo: EfficientNet-B0 (libreria `timm`), ajustado con transfer learning.
- Dataset: TrashNet (Gary Thung y Mindy Yang, Stanford, 2016, licencia MIT). 2.527 fotos, 6 clases, divididas 70/15/15 (train/val/test) de forma estratificada.
- Comparacion: se comparo contra CLIP sin entrenar (linea base). EfficientNet (cerca de 80% de accuracy en prueba) supera claramente a CLIP (cerca de 65%).
- App: Streamlit, publicada en Streamlit Community Cloud.
- LLM: Google Gemini (API gratuita), via el SDK `google-genai`. Solo se le envia texto (clase y confianza), no la imagen.

## Estructura

```
recicla-ia/
  app.py                        la aplicacion (Streamlit)
  requirements.txt              librerias necesarias
  README.md                     este archivo
  .streamlit/config.toml        tema de la app (colores)
  notebooks/
    entrenamiento.ipynb            notebook para ejecutar
    entrenamiento_ejecutado.ipynb  notebook ya ejecutado con resultados
  modelo/
    modelo.pt                   pesos entrenados
    clases.txt                  clases en el orden del modelo
    config.json                 arquitectura + accuracy
  ejemplos/                     imagenes de prueba
```

## Clave del LLM (secreto)

La clave nunca se sube al repositorio. Se guarda como secreto de la plataforma:
- Streamlit Community Cloud: al desplegar, Advanced settings -> Secrets, agregar:
  ```
  GEMINI_API_KEY = "tu_clave"
  ```
- Local: crear `.streamlit/secrets.toml` con esa misma linea (ese archivo esta en `.gitignore`), o definir la variable de entorno `GEMINI_API_KEY`.

Clave gratuita en Google AI Studio (https://aistudio.google.com/apikey).

## Ejecutar localmente

```
pip install -r requirements.txt
streamlit run app.py
```

## Publicar en Streamlit Community Cloud

1. Subir este repositorio a GitHub (publico).
2. Entrar a https://share.streamlit.io con la cuenta de GitHub dueña del repositorio.
3. Create app -> Deploy a public app from GitHub: elegir el repositorio, rama `main` y archivo `app.py`.
4. En Advanced settings, elegir Python 3.12 y pegar en Secrets `GEMINI_API_KEY = "tu_clave"`.
5. Deploy. Cada cambio que se suba a GitHub actualiza la app automaticamente.

## Uso de IA

Se usaron asistentes de IA para ayudar a escribir y ordenar el codigo; el equipo adapto, probo y entiende cada parte del notebook y de `app.py`.

## Aviso

Prototipo educativo. La clasificacion es orientativa y puede equivocarse, sobre todo con fotos reales. Ante la duda, revisar manualmente y consultar las reglas de reciclaje de la comuna.
