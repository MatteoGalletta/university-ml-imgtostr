import gradio as gr

def placeholder(image, testo):
    testo_output = "⏳ Attendi integrazione modello OCR..."
    return image, testo_output

def clear_inputs():
    return None, "", "📭 Testo cancellato o in attesa di input."

custom_css = """
#fixed-image {
    height: 300px;
    width: 100%;
    object-fit: contain;
}
#fixed-textbox, #fixed-readonly {
    height: 150px;
}
"""

with gr.Blocks(css=custom_css) as demo:
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Cosa fa l'algoritmo?")
            gr.Markdown("""
            1. NOME STEP 1: \n
                - descrizione  
            2. NOME STEP 2 \n
                - descrizione
            """)
        with gr.Column(scale=1):
            gr.Markdown("### Carica screenshoot")
            img_input = gr.Image(label="Carica lo screenshot", sources="upload", type="pil", elem_id="fixed-image")
            with gr.Row():
                btn_submit = gr.Button("Submit")
                btn_cancel = gr.Button("Cancel")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Immagine pre-processata")
            output_img = gr.Image(label="", show_label=False, elem_id="fixed-image")
        with gr.Column(scale=1):
            gr.Markdown("### Risultato")
            testo_output = gr.Markdown(
                value="📭 In attesa di input...", 
                elem_id="fixed-readonly",
            )

    # Nuovo blocco iframe a larghezza intera
    with gr.Row(): # Questa riga, essendo un figlio diretto di gr.Blocks, occuperà la larghezza disponibile.
        gr.HTML(
            """
            <div id="iframe-container">
                <iframe src="http://localhost:8000/tensorspace.html" width="100%" height="600" style="border:none; display: block;"></iframe>
            </div>
            """
        )

    btn_submit.click(fn=placeholder, inputs=[img_input], outputs=[output_img, testo_output])
    btn_cancel.click(fn=clear_inputs, inputs=None, outputs=[img_input, testo_output])

if __name__ == "__main__":
    demo.launch()
