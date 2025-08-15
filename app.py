import gradio as gr
import torch
import time
from PIL import Image
import tempfile
import os

from data.video import save_video
from wan_loader import load_wan_pipe
from models.set_condition_branch import set_stand_in
from preprocessor import FaceProcessor

print("Loading model, please wait...")
try:
    ANTELOPEV2_PATH = "checkpoints/antelopev2"
    BASE_MODEL_PATH = "checkpoints/base_model/"
    LORA_MODEL_PATH = "checkpoints/Stand-In/Stand-In_wan2.1_T2V_14B_ver1.0.ckpt"

    if not os.path.exists(ANTELOPEV2_PATH):
        raise FileNotFoundError(
            f"AntelopeV2 checkpoint not found at: {ANTELOPEV2_PATH}"
        )
    if not os.path.exists(BASE_MODEL_PATH):
        raise FileNotFoundError(f"Base model not found at: {BASE_MODEL_PATH}")
    if not os.path.exists(LORA_MODEL_PATH):
        raise FileNotFoundError(f"LoRA model not found at: {LORA_MODEL_PATH}")

    face_processor = FaceProcessor(antelopv2_path=ANTELOPEV2_PATH)
    pipe = load_wan_pipe(base_path=BASE_MODEL_PATH, torch_dtype=torch.bfloat16)
    set_stand_in(pipe, model_path=LORA_MODEL_PATH)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Model loading failed: {e}")
    with gr.Blocks() as demo:
        gr.Markdown("# Error: Model Loading Failed")
        gr.Markdown(f"""
        Please check the following:
        1.  Make sure the checkpoint files are placed in the correct directory.
        2.  Ensure all dependencies are properly installed.
        3.  Check the console output for detailed error information.
        
        **Error details**: {e}
        """)
    demo.launch()
    exit()


def generate_video(
    pil_image: Image.Image,
    prompt: str,
    seed: int,
    negative_prompt: str,
    num_steps: int,
    fps: int,
    quality: int,
):
    if pil_image is None:
        raise gr.Error("Please upload a face image first!")

    print("Processing face...")
    ip_image = face_processor.process(pil_image)
    print("Face processing completed.")

    print("Generating video...")
    start_time = time.time()
    video = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        seed=int(seed),
        ip_image=ip_image,
        num_inference_steps=int(num_steps),
        tiled=False,
    )
    end_time = time.time()
    print(f"Video generated in {end_time - start_time:.2f} seconds.")

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_file:
        video_path = temp_file.name
        save_video(video, video_path, fps=int(fps), quality=quality)
        print(f"Video saved to: {video_path}")
        return video_path


with gr.Blocks(theme=gr.themes.Soft(), css="footer {display: none !important}") as demo:
    gr.Markdown(
        """
        # Stand-In IP2V
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 1. Upload a Face Image")
            input_image = gr.Image(
                label="Upload Image",
                type="pil",
                image_mode="RGB",
                height=300,
            )

            gr.Markdown("### 2. Enter Core Parameters")
            input_prompt = gr.Textbox(
                label="Prompt",
                lines=4,
                value="一位男性舒适地坐在书桌前，正对着镜头，仿佛在与屏幕前的亲友对话。他的眼神专注而温柔，嘴角带着自然的笑意。背景是他精心布置的个人空间，墙上贴着照片和一张世界地图，传达出一种亲密而现代的沟通感。",
                placeholder="Please enter a detailed description of the scene, character actions, expressions, etc...",
            )

            input_seed = gr.Slider(
                label="Seed",
                minimum=0,
                maximum=100000,
                step=1,
                value=0,
                info="The same seed and parameters will generate the same result.",
            )

            with gr.Accordion("Advanced Options", open=False):
                input_negative_prompt = gr.Textbox(
                    label="Negative Prompt",
                    lines=3,
                    value="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
                )
                input_steps = gr.Slider(
                    label="Inference Steps",
                    minimum=10,
                    maximum=50,
                    step=1,
                    value=20,
                    info="More steps may improve details but will take longer to generate.",
                )
                output_fps = gr.Slider(
                    label="Video FPS", minimum=10, maximum=30, step=1, value=25
                )
                output_quality = gr.Slider(
                    label="Video Quality", minimum=1, maximum=10, step=1, value=9
                )

            generate_btn = gr.Button("Generate Video", variant="primary")

        with gr.Column(scale=1):
            gr.Markdown("### 3. View Generated Result")
            output_video = gr.Video(
                label="Generated Video",
                height=480,
            )
    generate_btn.click(
        fn=generate_video,
        inputs=[
            input_image,
            input_prompt,
            input_seed,
            input_negative_prompt,
            input_steps,
            output_fps,
            output_quality,
        ],
        outputs=output_video,
        api_name="generate_video",
    )

if __name__ == "__main__":
    demo.launch(share=True, server_name="0.0.0.0", server_port=8080)
