import gradio as gr
import numpy as np
import sys

sys.path.append('../core')
from ImageToStringClassifier import ImageToStringClassifier

def processing(image_pil):
    if image_pil is None:
        return None, ""
    
    image_np = np.array(image_pil)  # PIL → NumPy
    classifier = ImageToStringClassifier(image_np)
    bboxed_image = classifier.preprocessor.get_bboxed_image()
    string_output = classifier.get_string()
    return bboxed_image, string_output

def clear_inputs():
    return None, None, ""

custom_css = """

#image img{
  padding: 30px;
  background-color: #27272a;
  box-sizing: border-box;
  display:block;
}
"""

with gr.Blocks(css=custom_css) as demo:

    # Seconda riga: descrizione algoritmo
    with gr.Row():
        with gr.Column(scale=1):
            gr.Image(label="Steps", interactive=False, type="filepath")

    with gr.Row():
        # Colonna sinistra: input immagine + pulsanti
        with gr.Column(scale=1):
            gr.Markdown("# Carica screenshot")
            img_input = gr.Image(label="Screenshot caricato", sources="upload", type="pil",  elem_id="image")
            with gr.Row():
                btn_submit = gr.Button("Submit")
                btn_cancel = gr.Button("Cancel")
        
        # Colonna destra: immagine pre-processata sopra il risultato
        with gr.Column(scale=1):
            gr.Markdown("# Output")
            output_img = gr.Image(label="Immagine preprocessata", elem_id="image")

        with gr.Column(scale=1):
            gr.Markdown("# Risultato")
            output_text = gr.Textbox(label="Output algoritmo", interactive=False)


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
    btn_submit.click(fn=processing, inputs=[img_input], outputs=[output_img, output_text])
    btn_cancel.click(fn=clear_inputs, inputs=None, outputs=[img_input, output_img, output_text])

if __name__ == "__main__":
    demo.launch()
