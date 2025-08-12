<div align="center">

  <h1>
    <img src="assets/Stand-In.png" width="85" alt="Logo" valign="middle">
    Stand-In
  </h1>

  <h3>A Lightweight and Plug-and-Play Identity Control for Video Generation</h3>



[![arXiv](https://img.shields.io/badge/arXiv-2508.07901-b31b1b)](https://arxiv.org/abs/2508.07901)
[![Project Page](https://img.shields.io/badge/Project_Page-Link-green)](https://www.stand-in.tech)
[![🤗 HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Model-orange)](https://huggingface.co/BowenXue/Stand-In)

</div>

<img width="5333" height="2983" alt="Image" src="https://github.com/user-attachments/assets/2fe1e505-bcf7-4eb6-8628-f23e70020966" />

> **Stand-In** is a lightweight, plug-and-play framework for identity-preserving video generation. By training only **1%** additional parameters compared to the base video generation model, we achieve state-of-the-art results in both Face Similarity and Naturalness, outperforming various full-parameter training methods. Moreover, **Stand-In** can be seamlessly integrated into other tasks such as subject-driven video generation, pose-controlled video generation, video stylization, and face swapping.

---

## 🔥 News
* **[2025.08.09]** Released Stand-In v1.0 (153M parameters), the Wan2.1-14B-T2V–adapted weights and inference code are now open-sourced.

---

## 🌟 Showcase

### Identity-Preserving Text-to-Video Generation

| Reference Image | Prompt | Generated Video |
| :---: | :---: | :---: |
|![Image](https://github.com/user-attachments/assets/86ce50d7-8ccb-45bf-9538-aea7f167a541)| "In a corridor where the walls ripple like water, a woman reaches out to touch the flowing surface, causing circles of ripples to spread. The camera moves from a medium shot to a close-up, capturing her curious expression as she sees her distorted reflection." |![Image](https://github.com/user-attachments/assets/c3c80bbf-a1cc-46a1-b47b-1b28bcad34a3) |
|![Image](https://github.com/user-attachments/assets/de10285e-7983-42bb-8534-80ac02210172)| "A young man dressed in traditional attire draws the long sword from his waist and begins to wield it. The blade flashes with light as he moves—his eyes sharp, his actions swift and powerful, with his flowing robes dancing in the wind." |![Image](https://github.com/user-attachments/assets/1532c701-ef01-47be-86da-d33c8c6894ab)|

---
### Non-Human Subjects-Preserving Video Generation

| Reference Image | Prompt | Generated Video |
| :---: | :---: | :---: |
|<img width="415" height="415" alt="Image" src="https://github.com/user-attachments/assets/b929444d-d724-4cf9-b422-be82b380ff78" />|"A chibi-style boy speeding on a skateboard, holding a detective novel in one hand. The background features city streets, with trees, streetlights, and billboards along the roads."|![Image](https://github.com/user-attachments/assets/a7239232-77bc-478b-a0d9-ecc77db97aa5) |

---

### Identity-Preserving Stylized Video Generation

| Reference Image | LoRA | Generated Video |
| :---: | :---: | :---: |
|![Image](https://github.com/user-attachments/assets/9c0687f9-e465-4bc5-bc62-8ac46d5f38b1)|Ghibli LoRA|![Image](https://github.com/user-attachments/assets/c6ca1858-de39-4fff-825a-26e6d04e695f)|
---

### Video Face Swapping

| Reference Video | Identity | Generated Video |
| :---: | :---: | :---: |
|![Image](https://github.com/user-attachments/assets/33370ac7-364a-4f97-8ba9-14e1009cd701)|<img width="415" height="415" alt="Image" src="https://github.com/user-attachments/assets/d2cd8da0-7aa0-4ee4-a61d-b52718c33756" />|![Image](https://github.com/user-attachments/assets/0db8aedd-411f-414a-9227-88f4e4050b50)|


---

### Pose-Guided Video Generation (With VACE)

| Reference Pose | First Frame | Generated Video |
| :---: | :---: | :---: |
|![Image](https://github.com/user-attachments/assets/5df5eec8-b71c-4270-8a78-906a488f9a94)|<img width="719" height="415" alt="Image" src="https://github.com/user-attachments/assets/1c2a69e1-e530-4164-848b-e7ea85a99763" />|![Image](https://github.com/user-attachments/assets/1c8a54da-01d6-43c1-a5fd-cab0c9e32c44)|

---
### For more results, please visit [https://stand-in-video.github.io/](https://www.Stand-In.tech)

## 📖 Key Features
- Efficient Training: Only 1% of the base model parameters need to be trained.
- High Fidelity: Outstanding identity consistency without sacrificing video generation quality.
- Plug-and-Play: Easily integrates into existing T2V (Text-to-Video) models.
- Highly Extensible: Compatible with community models such as LoRA, and supports various downstream video tasks.

---

## ✅ Todo List 
- [x] Release IP2V inference script (compatible with community LoRA).
- [x] Open-source model weights compatible with Wan2.1-14B-T2V: `Stand-In_Wan2.1-T2V-14B_153M_v1.0`。
- [ ] Open-source model weights compatible with Wan2.2-T2V-A14B.
- [ ] Release training dataset, data preprocessing scripts, and training code.

---

## 🚀 Quick Start

### 1. Environment Setup
```bash
# Clone the project repository
git clone https://github.com/KBRASK/Stand-In.git
cd Stand-In

# Create and activate Conda environment
conda create -n Stand-In python=3.11 -y
conda activate Stand-In

# Install dependencies
pip install -r requirements.txt

# (Optional) Install Flash Attention for faster inference
# Note: Make sure your GPU and CUDA version are compatible with Flash Attention
pip install flash-attn --no-build-isolation
```

### 2. Model Download
We provide an automatic download script that will fetch all required model weights into the  `checkpoints` directory.
```bash
python download_models.py
```
This script will download the following models:
* `wan2.1-T2V-14B` (base text-to-video model)
* `antelopev2` (face recognition model)
* `Stand-In` (our Stand-In model)

> Note: If you already have the `wan2.1-T2V-14B model` locally, you can manually edit the `download_model.py` script to comment out the relevant download code and place the model in the `checkpoints/wan2.1-T2V-14B` directory.

---

## 🧪 Usage

### Standard Inference

Use the `infer.py` script for standard identity-preserving text-to-video generation.

```bash
python infer.py \
    --prompt "A man sits comfortably at a desk, facing the camera as if talking to a friend or family member on the screen. His gaze is focused and gentle, with a natural smile. The background is his carefully decorated personal space, with photos and a world map on the wall, conveying a sense of intimate and modern communication." \
    --ip_image "test/input/lecun.jpg" \
    --output "test/output/lecun.mp4"
```

### Inference with Community LoRA

Use the `infer_with_lora.py` script to load one or more community LoRA models alongside Stand-In.

```bash
python infer_with_lora.py \
    --prompt "A man sits comfortably at a desk, facing the camera as if talking to a friend or family member on the screen. His gaze is focused and gentle, with a natural smile. The background is his carefully decorated personal space, with photos and a world map on the wall, conveying a sense of intimate and modern communication." \
    --ip_image "test/input/lecun.jpg" \
    --output "test/output/lecun.mp4" \
    --lora_path "path/to/your/lora.safetensors" \
    --lora_scale 1.0
```

We recommend using this stylization LoRA: [https://civitai.com/models/1404755/studio-ghibli-wan21-t2v-14b](https://civitai.com/models/1404755/studio-ghibli-wan21-t2v-14b)

---

## 🤝 Acknowledgements

This project is built upon the following excellent open-source projects:
* [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio) (training/inference framework)
* [Wan2.1](https://github.com/Wan-Video/Wan2.1) (base video generation model)

We sincerely thank the authors and contributors of these projects.

---

## ✏ Citation

If you find our work helpful for your research, please consider citing our paper:

```bibtex
@article{xue2025standin,
    title={Stand-In: A Lightweight and Plug-and-Play Identity Control for Video Generation}, 
    author={Xue, Bowen and Yan, Qixin and Wang, Wenjing and Liu, Hao and Li, Chen},
    journal={arXiv preprint arXiv:2508.xxxxx},
    year={2025},
}
```

---

## 📬 Contact Us

If you have any questions or suggestions, feel free to reach out via [GitHub Issues](https://github.com/WeChatCV/Stand-In/issues) . We look forward to your feedback!
