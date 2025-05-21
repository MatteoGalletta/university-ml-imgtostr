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
            testo_input = gr.Textbox(label="Scrivi testo formattato",
                                     placeholder="Puoi scrivere qui...",
                                     lines=5,
                                     interactive=False,
                                     elem_id="fixed-textbox")
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
            testo_output = gr.Textbox(label="Testo riconosciuto",
                                      value="📭 In attesa di input...",
                                      lines=5,
                                      interactive=False,
                                      elem_id="fixed-readonly")

    btn_submit.click(fn=placeholder, inputs=[img_input, testo_input], outputs=[output_img, testo_output])
    btn_cancel.click(fn=clear_inputs, inputs=None, outputs=[img_input, testo_input, testo_output])

if __name__ == "__main__":
    demo.launch()
