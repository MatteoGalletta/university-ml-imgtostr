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

#input_image, #output_image {
    min-height: 400px;
}

#input_image img,
#output_image img {
    width: 100% !important;
    height: 100% !important;
    object-fit: contain;
    border-radius: 8px;
}
"""

with gr.Blocks(css=custom_css) as demo:
    with gr.Tabs():
        # Tab 1: OCR Tool
        with gr.TabItem("OCR Tool"):
            # Riga 1: Input immagine + pulsanti
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("# 1. Carica screenshot")
                    img_input = gr.Image(label="Screenshot caricato", sources=["upload", "clipboard"], type="pil", elem_id="input_image", show_label=False)
                    with gr.Row():
                        btn_submit = gr.Button("Submit")
                        btn_cancel = gr.Button("Cancel")

            # Riga 2: Output immagine
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("# 2. Pre-processing")
                    output_img = gr.Image(label="Output immagine", elem_id="output_image", show_label=False)

            # Riga 3: Output testuale
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("# 3. OCR")
                    output_text = gr.Textbox(lines=10, interactive=False, elem_id="output_text_area")

        # Tab 2: Tensorspace Viewer
        with gr.TabItem("3D view in Tensorspace"):
            gr.HTML(
                """
                <div>
                    <iframe src="http://localhost:8000/tensorspace.html" width="100%" height="600" style="border:none;"></iframe>
                </div>
                """
            )

    # Collegamenti ai pulsanti
    btn_submit.click(fn=processing, inputs=[img_input], outputs=[output_img, output_text])
    btn_cancel.click(fn=clear_inputs, inputs=None, outputs=[img_input, output_img, output_text])

if __name__ == "__main__":
    demo.launch()