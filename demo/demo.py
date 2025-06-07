import gradio as gr
import numpy as np
import sys

sys.path.append('../core')
from main import process_image

def clear_inputs():
    return None, None, ""

custom_css = """
#output_text_area textarea {
    font-size: 100px;
    line-height: 1.6;
}

textarea{
    height: 405px !important;
}

#input_image {
    min-height: 400px;
    width: 100% !important;
}

#input_image img {
    width: 100% !important;
    height: 100% !important;
    object-fit: contain;
    border-radius: 8px;
}

#output_image {
    min-height: 400px;
}

#output_image img {
    width: 100% !important;
    height: 100% !important;
    object-fit: contain;
    border-radius: 8px;
}

footer .svelte-1rjryqp{
    display: none !important;
}
"""

with gr.Blocks(css=custom_css) as demo:
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("# 1. Carica screenshot")
            img_input = gr.Image(label="Screenshot caricato", sources=["upload", "clipboard"], type="pil", elem_id="input_image", show_label=False)
            with gr.Row():
                btn_submit = gr.Button("ELABORA")

        with gr.Column(scale=1):
            gr.Markdown("# 2. Pre-processing")
            output_img = gr.Image(label="Output immagine", elem_id="output_image", show_label=False)

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("# 3. OCR")
            output_text = gr.Textbox(lines=10, interactive=False, elem_id="output_text_area")

    # Collegamenti ai pulsanti
    btn_submit.click(fn=process_image, inputs=[img_input], outputs=[output_img, output_text])

if __name__ == "__main__":
    demo.launch()