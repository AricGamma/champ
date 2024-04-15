import gradio as gr
import numpy as np
from PIL import Image
from pathlib import Path

from omegaconf import OmegaConf

from inference import main

INFERENCE_CONFIG_PATH = Path(__file__).parent.parent / "configs" / "inference" / "inference.yaml"

def predict(ref_img: Image, driving_video: str):
    cfg = OmegaConf.load(INFERENCE_CONFIG_PATH)
    main(cfg, ref_img=ref_img)
    return None


with gr.Blocks() as demo:
    with gr.Row():
        ref_img = gr.Image(type="pil")
        driving_video = gr.Dropdown(choices=["motion-01", "motion-02", "motion-03"], type="value")
    
    btn = gr.Button("Predict")
    btn.click(fn=predict, inputs=[ref_img, driving_video], outputs=gr.PlayableVideo())


demo.launch()
