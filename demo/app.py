# pylint: disable=wrong-import-position
# pylint: disable=no-member

import gc
import logging
import os
import sys
import uuid
from pathlib import Path

import gradio as gr

from omegaconf import OmegaConf
from PIL import Image
import torch

sys.path.append(str(Path(__file__).parent.parent))
from inference import (combine_guidance_data, get_weight_dtype,
                             inference, resize_tensor_frames, save_videos_grid,
                             setup_pretrained_models)

logging.basicConfig(level=logging.INFO, format="%(asctime)s: [%(levelname)s] [%(funcName)s] - %(message)s")

CACHE_DIR = Path(__file__).parent.parent / "results"

CACHE_INFERENCE_DIR = CACHE_DIR / "inference"

os.makedirs(CACHE_INFERENCE_DIR, exist_ok=True)


class InferenceService:
    def __init__(self, config_path: str, motions_path: str):
        self.config_path = config_path
        self.motions_path = motions_path
        self.noise_scheduler = None
        self.image_enc = None
        self.vae = None
        self.model = None
        self.weight_dtype = None
        self.cfg = OmegaConf.load(self.config_path)

    def setup_models(self):
        if self.weight_dtype is None:
            self.weight_dtype = get_weight_dtype(self.cfg)

        if (
            self.noise_scheduler is None
            and self.image_enc is None
            and self.vae is None
            and self.model is None
        ):
            logging.info("setup pretrained models...")
            (self.noise_scheduler, self.image_enc, self.vae, self.model) = setup_pretrained_models(self.cfg)

    def gc(self):
        self.noise_scheduler = None
        self.image_enc = None
        self.vae = None
        self.model = None

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    def inference(self, ref_img: Image, driving_motion: str, session_id: str, frame_start: int, frame_end: int):
        if not ref_img:
            raise gr.Error("Please upload a reference image")

        if not driving_motion:
            raise gr.Error("Please select a motion")

        if frame_start >= frame_end:
            raise gr.Error("Frame start must less than frame end")

        self.setup_models()

        try:
            ref_image_w, ref_image_h = ref_img.size

            motion_path = self.motions_path / driving_motion
            cfg = OmegaConf.load(self.config_path)
            cfg.data.guidance_data_folder = motion_path

            cfg.data.frame_range = [frame_start, frame_end]

            guidance_pil_group, video_length = combine_guidance_data(cfg)

            result_video_tensor = inference(
                cfg=cfg,
                vae=self.vae,
                image_enc=self.image_enc,
                model=self.model,
                scheduler=self.noise_scheduler,
                ref_image_pil=ref_img,
                guidance_pil_group=guidance_pil_group,
                video_length=video_length,
                width=cfg.width,
                height=cfg.height,
                device="cuda",
                dtype=self.weight_dtype,
            )

            result_video_tensor = resize_tensor_frames(
                result_video_tensor, (ref_image_h, ref_image_w)
            )
            saved_video_path = self.get_inference_path(session_id)
            save_videos_grid(result_video_tensor, saved_video_path)
            return saved_video_path
        finally:
            self.gc()

    def render_guidance_video(self, motion: str):
        return self.motions_path / motion / "motion_grid.mp4"

    def get_inference_path(self, session_id: str):
        return f"{CACHE_DIR}/inference/{session_id}.mp4"

    def get_motion_frame_count(self, motion: str):
        p = self.motions_path / motion
        return len(os.listdir(p / "depth"))

    def clean_cache(self, session_id: str):
        # clean inference
        if os.path.exists(self.get_inference_path(session_id)):
            os.remove(self.get_inference_path(session_id))
        print(f"{session_id} cleaned")


def launch(config_path: str, motions_path: str):
    service = InferenceService(config_path, motions_path)
    with gr.Blocks(delete_cache=(86400, 86400)) as app:
        session_id = gr.State(str(uuid.uuid4()))

        with gr.Column():
            with gr.Row():
                ref_img_component = gr.Image(type="pil", height=405, label="Reference Image")
                with gr.Column():
                    with gr.Row():
                        guidance_path_component = gr.Dropdown(choices=sorted(os.listdir(motions_path)), type="value", label="Select a Motion")
                        frame_start = gr.Number(label="Frame start", value=0, interactive=True)
                        frame_end = gr.Number(label="Frame end", value=0, interactive=True)
                    guidance_preview = gr.Video(interactive=False, label="Preview Motion", height=300)
            submit_btn_component = gr.Button("Predict")
            output_video_component = gr.Video(interactive=False, label="Inference Output", height=400)

        guidance_path_component.change(
            fn=service.render_guidance_video,
            inputs=[guidance_path_component],
            outputs=[guidance_preview]
        )

        guidance_path_component.change(fn=service.get_motion_frame_count, inputs=[guidance_path_component], outputs=[frame_end])
        submit_btn_component.click(
            fn=service.inference,
            inputs=[ref_img_component,guidance_path_component, session_id, frame_start, frame_end],
            outputs=output_video_component
        )

        def clean_session():
            service.clean_cache(session_id.value)

        app.unload(fn=clean_session)
    app.queue().launch()

if __name__ == "__main__":
    inference_config_path = Path(__file__).parent.parent / "configs" / "inference" / "inference.yaml"
    inference_motions_path = Path(__file__).parent.parent / "example_data" / "motions"
    launch(inference_config_path, inference_motions_path)
