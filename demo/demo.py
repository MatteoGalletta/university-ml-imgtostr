import gradio as gr

def placeholder(image, testo):
    testo_output = "⏳ Attendi integrazione modello OCR..."
    return image, testo_output

def clear_inputs():
    return None, "", "📭 Testo cancellato o in attesa di input."

custom_css = """
#fixed-height-image {
    height: 300px;
}
"""

with gr.Blocks(css = custom_css) as demo:

    # Seconda riga: descrizione algoritmo
    with gr.Row():
        with gr.Column(scale=1):
            gr.Image(label="Steps", interactive=False, type="filepath", elem_id="fixed-height-image")


    with gr.Row():
        # Colonna sinistra: input immagine + pulsanti
        with gr.Column(scale=1):
            gr.Markdown("# Carica screenshot")
            img_input = gr.Image(label="Carica lo screenshot", sources="upload", type="pil")
            with gr.Row():
                btn_submit = gr.Button("Submit")
                btn_cancel = gr.Button("Cancel")
        
        # Colonna destra: immagine pre-processata sopra il risultato
        with gr.Column(scale=1):
            gr.Markdown("# Output")
            output_img = gr.Image(label="Immagine pre-processata")

            testo_output = gr.Markdown(value="📭 In attesa di input...")

    # TODO: tensorspace
    with gr.Row():
        gr.HTML(
            """
            <div>
                <iframe src="http://localhost:8000/tensorspace.html" width="100%" height="600" style="border:none;"></iframe>
            </div>
            """
        )

    btn_submit.click(fn=placeholder, inputs=[img_input, testo_output], outputs=[output_img, testo_output])
    btn_cancel.click(fn=clear_inputs, inputs=None, outputs=[img_input, testo_output])

if __name__ == "__main__":
    demo.launch()