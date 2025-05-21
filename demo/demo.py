# ui.py

import gradio as gr

# funzione di prova (ritorna sempre lo stesso testo)
def placeholder(image):
    return "⏳ Attendi integrazione modello OCR..."

# definizione interfaccia
iface = gr.Interface(
    fn=placeholder,
    inputs=gr.Image(type="pil", label="Carica lo screenshot"),
    outputs=gr.Textbox(label="Output OCR"),
    title="📸 OCR UI",
    description="Interfaccia di test: carica un’immagine e premi Submit."
)

# avvio UI
if __name__ == "__main__":
    iface.launch()  # di default http://127.0.0.1:7860
