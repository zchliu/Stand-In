import os
import cv2
import torch
import numpy as np
import imageio
from insightface.app import FaceAnalysis
from facexlib.parsing import init_parsing_model
from torchvision.transforms.functional import normalize
from tqdm import tqdm
from typing import Optional, List, Tuple


class VideoMaskGenerator:
    """
    A class to process videos and generate face masks.
    Models for face detection and parsing are pre-loaded upon initialization for efficiency.
    """

    def __init__(
        self, antelopv2_path: str = "./", device: Optional[torch.device] = None
    ):
        """
        Initializes the VideoMaskGenerator by loading the necessary models.

        Args:
            root_path (str): The root directory for model storage.
            device (Optional[torch.device]): The device to run the models on.
                                              If None, it defaults to CUDA if available, otherwise CPU.
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        print(f"Using device: {self.device}")

        # Load FaceAnalysis model
        print("Loading FaceAnalysis model (antelopev2)...")
        providers = (
            ["CUDAExecutionProvider"]
            if self.device.type == "cuda"
            else ["CPUExecutionProvider"]
        )
        self.face_app = FaceAnalysis(
            name="antelopev2",
            root=antelopv2_path,
            providers=providers,
        )
        self.face_app.prepare(ctx_id=0, det_size=(640, 640))
        print("FaceAnalysis model loaded successfully.")

        # Load BiSeNet parsing model
        print("Loading BiSeNet model for face parsing...")
        self.parsing_model = init_parsing_model(
            model_name="bisenet", device=self.device
        )
        self.parsing_model.eval()
        print("BiSeNet model loaded successfully.")

    def _img2tensor(self, img: np.ndarray, bgr2rgb: bool = True) -> torch.Tensor:
        """Converts a NumPy image array (BGR) to a PyTorch tensor (RGB)."""
        if bgr2rgb:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        return torch.from_numpy(img)

    def _pad_to_square(
        self, img: np.ndarray, pad_color: int = 0
    ) -> Tuple[np.ndarray, int, int]:
        """Pads a NumPy image to make it square."""
        h, w = img.shape[:2]
        if h == w:
            return img, 0, 0

        if h > w:
            pad_left = (h - w) // 2
            pad_right = h - w - pad_left
            img_square = cv2.copyMakeBorder(
                img,
                0,
                0,
                pad_left,
                pad_right,
                cv2.BORDER_CONSTANT,
                value=[pad_color] * 3,
            )
            return img_square, pad_left, 0
        else:
            pad_top = (w - h) // 2
            pad_bottom = w - h - pad_top
            img_square = cv2.copyMakeBorder(
                img,
                pad_top,
                pad_bottom,
                0,
                0,
                cv2.BORDER_CONSTANT,
                value=[pad_color] * 3,
            )
            return img_square, 0, pad_top

    def _create_mask_for_frame(
        self, frame: np.ndarray, resize_to: int = 512, dilation_kernel_size: int = 0
    ) -> np.ndarray:
        """Analyzes a single video frame to create a full-size face mask."""
        orig_h, orig_w = frame.shape[:2]
        frame_padded, pad_left, pad_top = self._pad_to_square(frame, pad_color=0)

        final_mask = np.zeros(
            (frame_padded.shape[0], frame_padded.shape[1]), dtype=np.uint8
        )
        faces = self.face_app.get(frame_padded)

        if faces:
            largest_face = max(
                faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
            )
            x1, y1, x2, y2 = map(int, largest_face.bbox)

            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
            side_len = int(max(x2 - x1, y2 - y1) * 1.5)
            half_side = side_len // 2

            crop_x1 = max(center_x - half_side, 0)
            crop_y1 = max(center_y - half_side, 0)
            crop_x2 = min(center_x + half_side, frame_padded.shape[1])
            crop_y2 = min(center_y + half_side, frame_padded.shape[0])

            face_crop = frame_padded[crop_y1:crop_y2, crop_x1:crop_x2]

            if face_crop.size > 0:
                face_resized = cv2.resize(
                    face_crop, (resize_to, resize_to), interpolation=cv2.INTER_AREA
                )
                face_tensor = (
                    self._img2tensor(face_resized, bgr2rgb=True)
                    .unsqueeze(0)
                    .to(self.device)
                )

                with torch.no_grad():
                    normalized_face = normalize(
                        face_tensor, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
                    )
                    parsing_out = self.parsing_model(normalized_face)[0]
                    parsing_map = parsing_out.argmax(dim=1, keepdim=True)

                mask = (parsing_map.squeeze().cpu().numpy() != 0).astype(np.uint8) * 255

                if dilation_kernel_size > 0:
                    kernel = cv2.getStructuringElement(
                        cv2.MORPH_ELLIPSE, (dilation_kernel_size, dilation_kernel_size)
                    )
                    mask = cv2.dilate(mask, kernel, iterations=1)

                mask_resized_to_crop = cv2.resize(
                    mask,
                    (face_crop.shape[1], face_crop.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
                final_mask[crop_y1:crop_y2, crop_x1:crop_x2] = mask_resized_to_crop

        unpadded_mask = final_mask[
            pad_top : pad_top + orig_h, pad_left : pad_left + orig_w
        ]
        return cv2.cvtColor(unpadded_mask, cv2.COLOR_GRAY2BGR)

    def _save_video(
        self, frames: List[np.ndarray], save_path: str, fps: float, quality: int = 9
    ):
        """Saves a list of frames to a video file using imageio."""
        print(f"Preparing to save video to: {save_path} with imageio.")
        # Using a high-quality codec and crf value for better compatibility
        writer = imageio.get_writer(
            save_path,
            fps=fps,
            quality=quality,
            codec="libx264",
            ffmpeg_params=["-crf", "18"],
        )
        for frame in tqdm(frames, desc=f"Saving to {os.path.basename(save_path)}"):
            writer.append_data(frame)
        writer.close()
        print("Video saving complete.")

    def process(
        self,
        input_path: str,
        dilation_kernel_size: int = 0,
        output_path: Optional[str] = None,
    ) -> Optional[List[np.ndarray]]:
        """
        Reads a video, creates a mask for each frame, and returns the frames.
        Optionally saves the result to a file if output_path is provided.

        Args:
            input_path (str): Path to the input video file.
            dilation_kernel_size (int): The kernel size for mask dilation. If 0, no dilation is applied.
            output_path (Optional[str]): If provided, the path to save the output mask video.

        Returns:
            Optional[List[np.ndarray]]: A list of processed frames (RGB format) on success, None on failure.
        """
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file: {input_path}")
            return None

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(
            f"Processing video: {os.path.basename(input_path)} "
            f"(Resolution: {width}x{height}, FPS: {fps:.2f}, Frames: {total_frames})"
        )

        processed_frames_rgb = []
        try:
            for i in tqdm(
                range(total_frames),
                desc=f"Creating mask for {os.path.basename(input_path)}",
            ):
                ret, frame = cap.read()
                if not ret:
                    print(f"Info: Video stream ended at frame {i + 1}.")
                    break

                mask_frame_bgr = self._create_mask_for_frame(
                    frame, dilation_kernel_size=dilation_kernel_size
                )
                mask_frame_rgb = cv2.cvtColor(mask_frame_bgr, cv2.COLOR_BGR2RGB)
                processed_frames_rgb.append(mask_frame_rgb)
        finally:
            print("\nReleasing video capture resources...")
            cap.release()
            print("Resources released.")

        if not processed_frames_rgb:
            print("Error: No frames were generated.")
            return None

        # Activate saving logic only if an output path is provided
        if output_path:
            self._save_video(processed_frames_rgb, output_path, fps)

        return processed_frames_rgb
