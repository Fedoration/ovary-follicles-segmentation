import os
from pathlib import Path

from PIL import Image
from ultralytics import YOLO


def crop_dataset(
    data_path: str, target_data_path: str, model_path: str, device: str = "cuda"
) -> None:
    """Обрезает УЗИ-снимки и соответствующие маски при помощи обученной модели.

    Args:
        data_path (str): путь до датасета с УЗИ-снимками и масками
        target_data_path (str): путь, куда сохранять новый датасет
        model_path (str): путь до модели YOLO, обученной на детекцию границ снимка
        device (str, optional): девайс для обучения. Defaults to "cuda".
    """
    model = YOLO(model_path).to(device)
    for path in Path(data_path).iterdir():
        if path.is_file():
            if not path.match(os.path.join(data_path, "*_ovar.*")):
                img = Image.open(path)
                mask_path = path.name.split(".jpeg")[0] + "_ovar.png"
                mask = Image.open(os.path.join(data_path, mask_path))

                box = model(img)[0].boxes.xyxy[0].cpu().numpy()

                img = img.crop(box)
                mask = mask.crop(box)

                img.save(os.path.join(target_data_path, path.name))
                mask.save(os.path.join(target_data_path, mask_path))
