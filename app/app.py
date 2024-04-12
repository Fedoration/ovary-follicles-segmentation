import os
import subprocess
from io import BytesIO
from pathlib import Path
from time import sleep
from typing import Tuple, Union

import cv2
import numpy as np
import streamlit as st
import torch
from download_button import download_button
from PIL import Image, ImageDraw
from shapely.geometry import Polygon, box
from torchvision.transforms import Compose, Resize, ToTensor
from ultralytics import YOLO


class SegmentationApp:
    """
    This app allows users to upload an ultrasound image, runs a pre-trained
    segmentation model to predict the ovary mask, and overlays the mask on the
    original image.
    """

    def __init__(
        self,
        seg_model_path: str,
        det_model_path: str,
        foll_det_model_path: str,
        device: torch.device,
        transform,
    ) -> None:
        self.device = device
        self.seg_model = torch.load(seg_model_path, map_location=self.device)
        self.det_model = YOLO(det_model_path).to(self.device)
        self.foll_det_model = YOLO(foll_det_model_path).to(self.device)
        self.transform = transform
        self.image_formats = ["png", "jpg", "jpeg"]
        self.video_formats = ["mp4", "avi", "wmv"]

    def prepare_image(self, image: Image.Image) -> torch.Tensor:
        """
        Prepare an image for model input. Converts the image to grayscale and applies any required transforms.

        Args:
            image (PIL Image): The input image to prepare.

        Returns:
            torch.Tensor: The transformed image tensor.
        """
        image = image.convert("L")
        image = self.transform(image)

        return image

    def make_segmentation(self, input: torch.Tensor) -> np.ndarray:
        """Make segmentation predictions on the given input image.

        Args:
            input (torch.Tensor): The input image tensor to run segmentation on.

        Returns:
            np.ndarray: The predicted segmentation mask array.
        """
        with torch.no_grad():
            self.seg_model.eval()
            input = input.unsqueeze(0).to(self.device)
            logits = self.seg_model(input)
        pr_mask = logits.sigmoid()
        pr_mask = pr_mask.detach().cpu().numpy().squeeze()

        return pr_mask

    @staticmethod
    def prepare_ovary(
        img: Image.Image, bbox: np.ndarray
    ) -> Tuple[Image.Image, int, int]:
        """Crop and pad an image to extract the ovary region.

        Args:
            img (Image): The original input image.
            bbox (np.ndarray): The bounding box coordinates (x1, y1, x2, y2) for the ovary region.

        Returns:
            Image: The cropped and padded image containing the ovary region.
            int: Absolute x coordinate to restore follicle boxes.
            int:  Absolute y coordinate to restore follicle boxes.
        """
        bbox_pad = bbox.copy()
        bbox_pad[0] = max(0, bbox_pad[0] - 10)
        bbox_pad[1] = max(0, bbox_pad[1] - 10)
        bbox_pad[2] = min(img.size[0], bbox_pad[2] + 10)
        bbox_pad[3] = min(img.size[1], bbox_pad[3] + 10)
        ovary = img.crop(bbox_pad)

        return ovary, bbox_pad[0], bbox_pad[1]

    def make_detection(
        self, input: Image.Image, confidence_ovary: float, confidence_foll: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Make ovary and follicles detection on the input image.

        Args:
            input (Image): The input image to run detection on.
            confidence_ovary (float): The confidence threshold to draw ovary boxes.
            confidence_foll (float): The confidence threshold to draw follicle boxes.

        Returns:
            np.ndarray: The detected box coordinates for the ovary
            np.ndarray: The detected box coordinates for the follicles
        """
        bbox = None
        correct_foll_boxes = None
        output = self.det_model(input, conf=confidence_ovary)[0].boxes.xyxy
        if output.shape[0] > 0:
            bbox = output[0].cpu().numpy()
            print("box: ", bbox)

            prepared_ovary, x, y = self.prepare_ovary(input, bbox)
            foll_boxes = (
                self.foll_det_model(prepared_ovary, conf=confidence_foll)[0]
                .boxes.xyxy.cpu()
                .numpy()
            )
            for foll_box in foll_boxes:
                foll_box[0] += x
                foll_box[1] += y
                foll_box[2] += x
                foll_box[3] += y

                if foll_box[0] < bbox[0]:
                    foll_box[0] = bbox[0]
                if foll_box[1] < bbox[1]:
                    foll_box[1] = bbox[1]
                if foll_box[2] > bbox[2]:
                    foll_box[2] = bbox[2]
                if foll_box[3] > bbox[3]:
                    foll_box[3] = bbox[3]

            correct_foll_boxes = []
            for foll_box in foll_boxes:
                if foll_box[0] < foll_box[2] and foll_box[1] < foll_box[3]:
                    correct_foll_boxes.append(foll_box)
                    print("foll_boxes: ", foll_boxes)

        return bbox, correct_foll_boxes

    @staticmethod
    def make_boxes(img: Image, bbox: np.ndarray, foll_boxes: np.ndarray) -> Image:
        """Make an image with bounding boxes.

        Args:
            img (Image): The original image.
            bbox (np.ndarray): The ovary bounding box to draw.
            foll_boxes (np.ndarray): List of the follicle bounding boxes to draw.

        Returns:
            Image: New image with bounding boxes drawn on it.
        """
        det_img = img.copy()
        draw = ImageDraw.Draw(det_img)
        if bbox is not None:
            draw.rectangle(bbox, outline=(255, 77, 77), width=3)
            if foll_boxes is not None:
                for foll_box in foll_boxes:
                    draw.rectangle(foll_box, outline=(217, 217, 217), width=2)

        return det_img

    @staticmethod
    def calculate_follicle_area(foll_boxes: np.ndarray, bbox: np.ndarray) -> float:
        """Calculate the percentage area that follicles covers within ovary.

        Args:
            foll_boxes: 2D array of follicle box coordinates in format [[x1, y1, x2, y2], ...]
            bbox: 1D array of ovary box coordinates in format [x1, y1, x2, y2]

        Returns:
            area_percent: The percentage area covered by foll_boxes within bbox

        """
        poly_total = Polygon()
        for box_coords in foll_boxes:
            poly = box(
                min(box_coords[0], box_coords[2]),
                min(box_coords[1], box_coords[3]),
                max(box_coords[0], box_coords[2]),
                max(box_coords[1], box_coords[3]),
            )
            poly_total = poly_total.union(poly)

        area_percent = (
            100 * poly_total.area / ((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
        )
        return area_percent

    @staticmethod
    def make_mask(image: Image, mask: np.ndarray, confidence: float) -> np.ndarray:
        """Generate a masked image based on the input image and mask.

        Args:
            image (np.ndarray): The input image.
            mask (np.ndarray): The mask to apply to the image. Values above 0.6 will be set to 1, others to 0.
            confidence: (float): The confidence threshold for the mask.

        Returns:
            np.ndarray: The masked image.
        """
        mask[mask > confidence] = 1
        mask[mask <= confidence] = 0
        mask = mask.astype("uint8")

        # width_origin = image.size[0]
        # height_origin = image.size[1]
        # image = image.resize((640, 480))
        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

        masked_img = np.zeros(image.shape, dtype=image.dtype)
        print("masked_img.shape: ", masked_img.shape)
        masked_img[:, :] = (255, 0, 0)

        masked_img = cv2.bitwise_and(masked_img, masked_img, mask=mask)
        masked_img = cv2.addWeighted(masked_img, 0.5, image, 1, 0)
        # masked_img = cv2.resize(masked_img, (width_origin, height_origin))

        return masked_img

    @staticmethod
    def make_conclusion(
        foll_boxes: Union[np.ndarray, int, None],
        bbox: Union[np.ndarray, None],
        statistics: Tuple[int, int],
    ) -> Tuple[str]:
        """
        Make a conclusion about the patient's ovarian state based on the detected ovary and follicles.

        Parameters:
            foll_boxes (Union[np.ndarray, int, None]): Detected follicle boxes array, average follicle count or None.
            bbox (Union[np.ndarray, None]): Bounding box array for the ovary or None.
            statistics (Tuple[int, int]): Tuple containing max and total follicle count.

        Returns:
            Tuple[str, str, str]: Tuple containing strings with area percentage, average count,
            max and total count and conclusion.
        """
        area_res = ""
        max_cnt_res = ""
        total_cnt_res = ""
        if foll_boxes is None:
            cnt = 0
        elif isinstance(foll_boxes, np.ndarray):
            cnt = len(foll_boxes)
            area_percent = SegmentationApp.calculate_follicle_area(foll_boxes, bbox)
            area_res = f"Процент площади яичника, занимаемой фолликулами: {area_percent:.2f}%\n"
        elif isinstance(foll_boxes, int):
            cnt = foll_boxes
            max_cnt_res = (
                f"Максимальное количество фолликулов в видеопотоке: {statistics[0]}\n"
            )
            total_cnt_res = f"Всего фолликулов в видеопотоке: {statistics[1]}\n"

        if statistics[0] == 0:
            cnt_res, res = (
                f"Было распознано фолликулов (в среднем): {cnt}.",
                "Яичники уменьшены, фолликулов нет или присутствуют единичные и очень маленькие. Вероятно, наступил"
                " период менопаузы.",
            )
        elif statistics[0] < 5:
            cnt_res, res = (
                f"Было распознано фолликулов (в среднем): {cnt}.",
                "Вероятно, фолликулярный резерв пациента снижен.",
            )
        elif statistics[0] < 15:
            cnt_res, res = (
                f"Было распознано фолликулов (в среднем): {cnt}.",
                "Вероятно, фолликулярный резерв пациента находится в пределах"
                " нормы.",
            )
        else:
            cnt_res, res = (
                f"Было распознано фолликулов (в среднем): {cnt}.",
                "Наблюдается большое количество фолликулов. Вероятно, яичники"
                " являются мультифолликулярными, могут быть увеличены.",
            )

        return area_res, cnt_res, max_cnt_res, total_cnt_res, res

    def image_ui(self, file: BytesIO) -> None:
        """Display UI for image segmentation and object detection.

        Args:
            file (BytesIO): Byte stream containing the image file to process.

        Returns:
            None
        """
        title_slot = st.empty()
        tab_seg, tab_det, tab_res = st.tabs(["Сегментация", "Детекция", "Заключение"])

        img_seg_slot = tab_seg.empty()
        img_det_slot = tab_det.empty()

        if file is not None:
            title_slot.title("УЗИ-снимок")
            img = Image.open(file)
            with tab_seg:
                img_seg_slot.image(img)

                with st.spinner("Подождите..."):
                    mask = self.make_segmentation(self.prepare_image(img))
                    mask = cv2.resize(mask, (img.size[0], img.size[1]))

                    confidence = st.select_slider(
                        label="Степень чувствительности алгоритма",
                        options=np.arange(0.51, 1.01, 0.01).round(decimals=2),
                        value=0.6,
                    )
                    masked_img = self.make_mask(img, mask, confidence)

                mask_on = st.toggle("Выделить яичник", value=True, key="segmentation")

                if mask_on:
                    img_seg_slot.image(masked_img)
                else:
                    img_seg_slot.image(img)

            with tab_det:
                img_det_slot.image(img)

                with st.spinner("Подождите..."):
                    confidence_ovary = st.select_slider(
                        label="Степень чувствительности детекции яичника",
                        options=np.arange(0.1, 1.01, 0.01).round(decimals=2),
                        value=0.25,
                    )
                    confidence_foll = st.select_slider(
                        label="Степень чувствительности детекции фолликулов",
                        options=np.arange(0.1, 1.01, 0.01).round(decimals=2),
                        value=0.25,
                    )
                    bbox, foll_boxes = self.make_detection(
                        img, confidence_ovary, confidence_foll
                    )
                    det_img = self.make_boxes(img, bbox, foll_boxes)

                box_on = st.toggle("Выделить яичник", value=True, key="detection")

                if box_on:
                    img_det_slot.image(det_img)
                else:
                    img_det_slot.image(img)

            with tab_res:
                area_conclusion, cnt_conclusion, _, _, conclusion = (
                    self.make_conclusion(
                        foll_boxes=foll_boxes, bbox=bbox, statistics=None
                    )
                )
                st.write(area_conclusion)
                st.write(cnt_conclusion)
                st.write(conclusion)

    def make_detection_video(
        self, file_path: Path, confidence_ovary: float, confidence_foll: float
    ) -> Tuple[str, bytes, int, int, int]:
        """Generate a video with object detections from an input video.

        Args:
            file_path (Path): Path to the input video file
            confidence_ovary (float): Confidence threshold for the ovaries detection
            confidence_foll (float): Confidence threshold for the follicles detection

        Returns:
            Tuple[str, bytes, int, int, int]: Video path, video file in bytes,
            mean follicles detected, max follicles detected, total follicles detected
        """
        capture = cv2.VideoCapture(str(file_path))

        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = capture.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        num_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        out_video_path = os.path.join(
            file_path.parent, file_path.stem + "_output_tmp.mp4"
        )
        out_video = cv2.VideoWriter(
            filename=out_video_path,
            fourcc=fourcc,
            fps=fps,
            frameSize=(width, height),
        )

        pbar = st.progress(0, text="Выполняется детекция...")
        frame_counter = 0
        foll_counter = []
        while capture.isOpened():
            ret, frame = capture.read()
            if not ret:
                break

            frame_counter += 1
            if (frame_counter + 2) % 3 == 0:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = Image.fromarray(frame)
                bbox, foll_boxes = self.make_detection(
                    frame, confidence_ovary, confidence_foll
                )
                if foll_boxes is not None:
                    foll_counter.append(len(foll_boxes))

                det_frame = self.make_boxes(frame, bbox, foll_boxes)
                out_frame = cv2.cvtColor(np.array(det_frame), cv2.COLOR_RGB2BGR)

                out_video.write(out_frame)
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = Image.fromarray(frame)
                det_frame = self.make_boxes(frame, bbox, foll_boxes)
                out_frame = cv2.cvtColor(np.array(det_frame), cv2.COLOR_RGB2BGR)
                out_video.write(out_frame)

            progress_percent = int(100 * (frame_counter / num_frames))
            pbar.progress(progress_percent, text="Выполняется детекция...")
        pbar.progress(100, text="Детекция завершена!")

        capture.release()
        out_video.release()
        cv2.destroyAllWindows()

        final_video_path = os.path.join(
            file_path.parent, file_path.stem + "_output.mp4"
        )
        subprocess.call(
            args=[
                "ffmpeg",
                "-y",
                "-i",
                out_video_path,
                "-c:v",
                "libx264",
                final_video_path,
            ]
        )
        with open(final_video_path, "rb") as f:
            video_bytes = f.read()
        os.remove(out_video_path)
        os.remove(final_video_path)

        mean_folls = round(3 * (sum(foll_counter) / frame_counter))
        max_folls = max(foll_counter)
        total_folls = sum(foll_counter)

        return final_video_path, video_bytes, mean_folls, max_folls, total_folls

    def video_ui(self, file: Path):
        """
        Display a video file with detections overlayed and provide option to download processed video.

        Args:
            file (Path): The input video file path

        Returns:
            None
        """
        # Сохраним видео локально
        file_path = Path(file.name).name
        file_path = Path(os.path.join(os.path.dirname(__file__), file_path))
        with open(file_path, "wb") as f:
            f.write(file.read())

        title_slot = st.empty()
        tab_det, tab_res = st.tabs(["Детекция", "Заключение"])
        if file is not None:
            title_slot.title("УЗИ-видео")

            with tab_det:
                confidence_ovary = st.select_slider(
                    label="Степень чувствительности детекции яичника",
                    options=np.arange(0.1, 1.01, 0.01).round(decimals=2),
                    value=0.25,
                )
                confidence_foll = st.select_slider(
                    label="Степень чувствительности детекции фолликулов",
                    options=np.arange(0.1, 1.01, 0.01).round(decimals=2),
                    value=0.25,
                )
                (
                    out_file_path,
                    video_bytes,
                    mean_foll_cnt,
                    max_foll_cnt,
                    total_foll_cnt,
                ) = self.make_detection_video(
                    file_path, confidence_ovary, confidence_foll
                )
                os.remove(file_path)
                st.video(video_bytes)

                download_button_str = download_button(
                    video_bytes, out_file_path, "Скачать результат"
                )
                st.markdown(download_button_str, unsafe_allow_html=True)

            with tab_res:
                (
                    _,
                    cnt_conclusion,
                    max_cnt_conclusion,
                    total_cnt_conclusion,
                    conclusion,
                ) = self.make_conclusion(
                    foll_boxes=mean_foll_cnt,
                    bbox=None,
                    statistics=(max_foll_cnt, total_foll_cnt),
                )
                st.write(cnt_conclusion)
                st.write(max_cnt_conclusion)
                st.write(total_cnt_conclusion)
                st.write(conclusion)

    def run(self):
        """Run the main app functionality.

        This function handles loading the ultrasound image file uploaded by the user,
        displaying the image, segmenting the ovaries mask, overlaying the mask on the image,
        and toggling between the masked and unmasked image.
        """
        file = st.file_uploader(
            "Загрузите УЗИ-снимок", type=self.image_formats + self.video_formats
        )
        try:
            file_path = Path(file.name)
            file_ext = file_path.suffix[1:].lower()
            if file_ext in self.image_formats:
                self.image_ui(file)
            elif file_ext in self.video_formats:
                self.video_ui(file)
        except AttributeError:
            pass


if __name__ == "__main__":
    transform = Compose(
        [
            Resize((480, 640)),
            ToTensor(),
        ]
    )
    seg_model_path = "app_models/ovary_model.pth"
    det_model_path = "app_models/yolov8-ovary.pt"
    foll_det_model_path = "app_models/yolov8-follicle.pt"
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # device = torch.device("cpu")

    app = SegmentationApp(
        seg_model_path=seg_model_path,
        det_model_path=det_model_path,
        foll_det_model_path=foll_det_model_path,
        device=device,
        transform=transform,
    )
    app.run()
