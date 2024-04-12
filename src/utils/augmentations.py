import os
import random
from pathlib import Path
from typing import List

from PIL import Image


def make_augmentations(data_paths: List[str], save_path: str) -> List[str]:
    """
    Augment a dataset by randomly flipping images and masks left-right.

    Args:
        data_paths (List[str]): List of paths to images and masks to augment.
        save_path (str): Path to save the augmented images and masks.

    Returns:
        List[str]: List of paths to original and augmented images and masks.
    """
    augmented_data_paths = []
    for path in data_paths:
        if ("ovar" not in path) and ("foll" not in path) and ("flipped" not in path):
            path = Path(path)
            is_apply = bool(random.getrandbits(1))

            img = Image.open(path)
            mask_path = path.stem + "_ovar.png"
            mask = Image.open(os.path.join(path.parent, mask_path))

            if is_apply:
                img_flipped = img.transpose(Image.FLIP_LEFT_RIGHT)
                mask_flipped = mask.transpose(Image.FLIP_LEFT_RIGHT)
                # mask_foll_flipped = mask_foll.transpose(Image.FLIP_LEFT_RIGHT)

                img_augm_path = os.path.join(save_path, path.stem + "_flipped.jpeg")
                mask_augm_path = os.path.join(
                    save_path, path.stem + "_flipped_ovar.png"
                )
                # mask_foll_augm_path = os.path.join(save_path, path.stem + "_flipped_foll.png")

                # сохраняем аугментированные изображения
                augmented_data_paths.append(img_augm_path)
                augmented_data_paths.append(mask_augm_path)
                # augmented_data_paths.append(mask_foll_augm_path)

                img_flipped.save(img_augm_path)
                mask_flipped.save(mask_augm_path)
                # mask_foll_flipped.save(mask_foll_augm_path)

                # а также исходное
                augmented_data_paths.append(os.path.join(save_path, path.name))
                augmented_data_paths.append(os.path.join(save_path, mask_path))
                # augmented_data_paths.append(os.path.join(save_path, mask_foll_path))

                img.save(os.path.join(save_path, path.name))
                mask.save(os.path.join(save_path, mask_path))
                # mask_foll.save(os.path.join(save_path, mask_foll_path))
            else:
                # если аугментации не применялись, то сохраняем только исходную тройку
                augmented_data_paths.append(os.path.join(save_path, path.name))
                augmented_data_paths.append(os.path.join(save_path, mask_path))
                # augmented_data_paths.append(os.path.join(save_path, mask_foll_path))

                img.save(os.path.join(save_path, path.name))
                mask.save(os.path.join(save_path, mask_path))
                # mask_foll.save(os.path.join(save_path, mask_foll_path))

    return augmented_data_paths
