import gradio as gr
import numpy as np
import sys

sys.path.append('../core')
from ImageToStringPreprocessing import ImageToStringPreprocessing

def binary(image_pil):
    image_np = np.array(image_pil)  # PIL → NumPy
    preprocessing = ImageToStringPreprocessing(image_np)
    bboxed_image = preprocessing.get_bboxed_image()
    return bboxed_image

def clear_inputs():
    return None

custom_css = """
#fixed-height-image {
    height: 300px;
}
"""

with gr.Blocks(css=custom_css) as demo:

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
            output_img = gr.Image(label="Immagine binarizzata")

    # Tensorspace viewer
    with gr.Row():
        gr.HTML(
            """
            <div>
                <iframe src="http://localhost:8000/tensorspace.html" width="100%" height="600" style="border:none;"></iframe>
            </div>
            """
        )

    # Collega i bottoni alle funzioni
    btn_submit.click(fn=binary, inputs=[img_input], outputs=[output_img])
    btn_cancel.click(fn=clear_inputs, inputs=None, outputs=[img_input])

if __name__ == "__main__":
    demo.launch()
