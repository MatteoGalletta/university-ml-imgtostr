import gradio as gr
import numpy as np
import sys

sys.path.append('../core')
from ImageToStringClassifier import ImageToStringClassifier

def processing(image_pil):
    if image_pil is None:
        return None, ""

    image_np = np.array(image_pil)
    classifier = ImageToStringClassifier(image_np)
    bboxed_image = classifier.preprocessor.get_bboxed_image()
    string_output = classifier.get_string()
    return bboxed_image, string_output

def clear_inputs():
    return None, None, ""

custom_css = """
#output_text_area textarea {
    font-size: 100px;
    line-height: 1.6;
    font-family: monospace;
}

/* Adjustments for the input image to take full width */
#input_image {
    min-height: 400px; /* Keep minimum height for consistency */
    width: 100% !important; /* Make the input image take full width */
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
"""

with gr.Blocks(css=custom_css) as demo:
    # OCR Tool content directly within the block
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("# 1. Carica screenshot")
            # Adjusted the img_input to take full width within its column
            img_input = gr.Image(label="Screenshot caricato", sources=["upload", "clipboard"], type="pil", elem_id="input_image", show_label=False)
            with gr.Row():
                btn_submit = gr.Button("Submit")
                btn_cancel = gr.Button("Cancel")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("# 2. Pre-processing")
            output_img = gr.Image(label="Output immagine", elem_id="output_image", show_label=False)

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("# 3. OCR")
            output_text = gr.Textbox(lines=10, interactive=False, elem_id="output_text_area")

    # Collegamenti ai pulsanti
    btn_submit.click(fn=processing, inputs=[img_input], outputs=[output_img, output_text])
    btn_cancel.click(fn=clear_inputs, inputs=None, outputs=[img_input, output_img, output_text])

if __name__ == "__main__":
    demo.launch()