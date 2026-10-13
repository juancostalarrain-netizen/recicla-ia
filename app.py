# -*- coding: utf-8 -*-
# RecicláIA · App de clasificación de residuos + recomendación con IA
# Machine Learning · Universidad del Desarrollo · Certamen I · Grupo 1
# Flujo: foto -> EfficientNet-B0 (clase + confianza) -> barras -> Gemini (recomendación)

import os                                           # para leer la clave del LLM
import json                                         # para leer config.json (accuracy del modelo)
import timm                                         # modelos de imágenes preentrenados
import torch                                        # motor de deep learning
import streamlit as st                              # interfaz web
from PIL import Image                               # manejo de imágenes

# ───────────────────────── Configuración de la página ─────────────────────────
st.set_page_config(page_title="RecicláIA", page_icon="♻️", layout="wide")

# Texto bonito e ícono para cada clase (solo para mostrar)
INFO = {
    "carton":   ("📦", "Cartón"),
    "metal":    ("🥫", "Metal"),
    "papel":    ("📄", "Papel"),
    "plastico": ("🧴", "Plástico"),
    "resto":    ("🗑️", "Resto (no reciclable)"),
    "vidrio":   ("🍾", "Vidrio"),
}

# ───────────────────────── 1. Cargar el modelo (una sola vez) ─────────────────────────
@st.cache_resource                                  # Streamlit guarda el modelo en memoria y no lo recarga en cada clic
def cargar_modelo():
    clases = open("modelo/clases.txt", encoding="utf-8").read().splitlines()   # clases en el orden del modelo
    config = json.load(open("modelo/config.json", encoding="utf-8"))           # datos guardados en el notebook
    modelo = timm.create_model("efficientnet_b0", pretrained=False, num_classes=len(clases))  # misma red del notebook
    modelo.load_state_dict(torch.load("modelo/modelo.pt", map_location="cpu"))  # pesos entrenados
    modelo.eval()                                                               # modo evaluación
    prep = timm.data.create_transform(**timm.data.resolve_model_data_config(modelo))  # mismo preprocesamiento
    return modelo, prep, clases, config

modelo, prep, CLASES, CONFIG = cargar_modelo()
ACC = CONFIG.get("accuracy_prueba", 0)              # accuracy del modelo en el conjunto de prueba

def clasificar(imagen):
    x = prep(imagen.convert("RGB")).unsqueeze(0)    # preparamos la foto igual que en el entrenamiento
    with torch.no_grad():                           # no calculamos gradientes (solo predecimos)
        probs = torch.softmax(modelo(x), dim=1)[0]  # probabilidad de cada clase (suman 1)
    return {CLASES[i]: float(probs[i]) for i in range(len(CLASES))}   # {clase: probabilidad}

# ───────────────────────── 2. Recomendación con el LLM (Gemini) ─────────────────────────
def leer_clave():
    try:
        if "GEMINI_API_KEY" in st.secrets:          # en Streamlit Cloud la clave va en los "Secrets"
            return st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass
    return os.environ.get("GEMINI_API_KEY")         # o en una variable de entorno (local)

CANDIDATOS = ["gemini-flash-latest", "gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"]
_MODELO_OK = None                                   # recordamos el modelo de Gemini que sí respondió

def _generar(cliente, prompt):
    global _MODELO_OK
    nombres = ([_MODELO_OK] if _MODELO_OK else []) + CANDIDATOS   # primero el que ya funcionó
    try:                                            # agregamos los modelos "flash" que ofrezca la cuenta
        for m in cliente.models.list():
            n = m.name.split("/")[-1]
            if "flash" in n and n not in nombres:
                nombres.append(n)
    except Exception:
        pass
    ultimo_error = None
    for nombre in nombres:
        try:
            r = cliente.models.generate_content(model=nombre, contents=prompt)
            _MODELO_OK = nombre
            return r.text
        except Exception as e:
            ultimo_error = e
    raise ultimo_error if ultimo_error else RuntimeError("sin modelo disponible")

def recomendar(clase, confianza):
    clave = leer_clave()
    if not clave:                                   # si no hay clave, avisamos con calma (la clasificación igual funciona)
        return "⚠️ Falta configurar la clave del LLM (`GEMINI_API_KEY`) como secreto."
    nombre = INFO[clase][1]

    # Prompt RESPONSABLE: limita el tema, pide revisión si la confianza es baja y aclara que es educativo.
    prompt = f"""Eres un asistente de reciclaje para una app educativa en Chile.
El modelo clasificó un residuo como "{nombre}" con una confianza de {confianza:.0%}.

Entrega una recomendación breve (3 o 4 frases) y concreta sobre cómo reciclar o disponer correctamente ese tipo de residuo.
Reglas:
- Habla SOLO de reciclaje y manejo de ese residuo; no cambies de tema.
- Si la confianza es menor a 60%, parte advirtiendo que la predicción es poco segura y que conviene revisar la foto o la categoría.
- Recuerda que es orientativo y educativo, no una norma oficial.
Responde en español, con un tono cercano."""

    try:
        from google import genai                    # SDK de Gemini
        cliente = genai.Client(api_key=clave)
        texto = _generar(cliente, prompt).strip()
    except Exception as e:
        return f"No se pudo generar la recomendación ahora mismo ({e})."

    if confianza < 0.60:                             # recordatorio visible de revisión humana
        texto = "🔎 *La predicción es poco segura, conviene revisar la foto.*\n\n" + texto
    return texto

# ───────────────────────── 3. Interfaz ─────────────────────────
st.title("♻️ RecicláIA")
st.markdown(
    "Sube la foto de un residuo y te digo a qué categoría va, con una **recomendación de reciclaje generada por IA**.  \n"
    "**Grupo 1** · Tomás Allende · Felipe Castillo · Juan Costa — Machine Learning, UDD"
)
st.info(f"🧠 Modelo: EfficientNet-B0 · **Accuracy en prueba: {ACC:.0%}** "
        "(este número es del modelo completo, no de cada foto).")

if "imagen" not in st.session_state:
    st.session_state.imagen = None                  # aquí guardamos la foto elegida

col_izq, col_der = st.columns(2)

with col_izq:
    st.subheader("📷 Tu imagen")
    subida = st.file_uploader("Sube una foto de un residuo", type=["jpg", "jpeg", "png"])
    if subida is not None:
        st.session_state.imagen = Image.open(subida).convert("RGB")

    st.caption("O prueba con un ejemplo:")
    ejemplos = {"📦 Cartón": "ejemplos/ejemplo_carton.jpg", "🍾 Vidrio": "ejemplos/ejemplo_vidrio.jpg",
                "🧴 Plástico": "ejemplos/ejemplo_plastico.jpg", "📄 Papel": "ejemplos/ejemplo_papel.jpg"}
    cols = st.columns(len(ejemplos))
    for (texto, ruta), c in zip(ejemplos.items(), cols):
        if c.button(texto) and os.path.exists(ruta):
            st.session_state.imagen = Image.open(ruta).convert("RGB")

    if st.session_state.imagen is not None:
        st.image(st.session_state.imagen, use_container_width=True)

with col_der:
    st.subheader("📊 Resultado")
    if st.session_state.imagen is None:
        st.write("Sube una imagen o elige un ejemplo para clasificar.")
    else:
        probs = clasificar(st.session_state.imagen)                     # {clase: probabilidad}
        orden = sorted(probs, key=probs.get, reverse=True)              # de mayor a menor
        top = orden[0]; conf = probs[top]
        st.markdown(f"### {INFO[top][0]} {INFO[top][1]} — {conf:.0%}")   # clase más probable

        for c in orden:                                                 # barras de todas las clases
            st.write(f"{INFO[c][0]} {INFO[c][1]}")
            st.progress(probs[c])

        st.divider()
        if st.button("💬 Recomendación con IA", type="primary"):
            with st.spinner("Pensando una recomendación…"):
                st.markdown(recomendar(top, conf))

st.divider()
st.warning(
    "⚠️ **Aviso:** esta app es un **prototipo educativo**. La clasificación es orientativa y puede equivocarse, "
    "sobre todo con fotos reales (mala luz, residuos mezclados). Cuando la confianza sea baja, revisa tú mismo "
    "antes de decidir. Ante la duda, consulta las reglas de reciclaje de tu comuna."
)
