from typing import Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms


class ScaleAugmentation:
    pad_mod = 8

    def __init__(self, model_path: str, device: str) -> None:
        self.device = device
        self.model = torch.jit.load(model_path, map_location=device).to(device)
        self.model.eval()

    @staticmethod
    def _normalize_img(np_img: np.ndarray) -> np.ndarray:
        """Normalizes the input image by converting it to a float32 array and scaling the pixel values to be between 0 and 1.

        Args:
            np_img (np.ndarray): image to normalize.

        Returns:
            np.ndarray: normalized image.
        """
        if len(np_img.shape) == 2:
            np_img = np_img[:, :, np.newaxis]
        np_img = np.transpose(np_img, (2, 0, 1))
        np_img = np_img.astype("float32") / 255
        return np_img

    def _forward(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Performs the forward pass through the PyTorch model using the input image and mask.

        Args:
            image (np.ndarray): image to process.
            mask (np.ndarray): mask with the damaged areas.

        Returns:
            np.ndarray: the resulting image with the missing parts filled in.
        """
        image = self._normalize_img(image)
        mask = self._normalize_img(mask)

        mask = (mask > 0) * 1
        image = torch.from_numpy(image).unsqueeze(0).to(self.device)
        mask = torch.from_numpy(mask).unsqueeze(0).to(self.device)

        inpainted_image = self.model(image, mask)

        result = inpainted_image[0].permute(1, 2, 0).detach().cpu().numpy()
        result = np.clip(result * 255, 0, 255).astype("uint8")

        return cv2.cvtColor(result, cv2.COLOR_RGB2BGR)

    @staticmethod
    def _ceil_size(x: int, mod: int) -> int:
        if x % mod == 0:
            return x
        return (x // mod + 1) * mod

    @staticmethod
    def _pad_img(img: np.ndarray, mod: int) -> np.ndarray:
        if len(img.shape) == 2:
            img = img[:, :, np.newaxis]
        height, width = img.shape[:2]
        out_height = ScaleAugmentation._ceil_size(height, mod)
        out_width = ScaleAugmentation._ceil_size(width, mod)

        return np.pad(
            img,
            ((0, out_height - height), (0, out_width - width), (0, 0)),
            mode="symmetric",
        )

    @staticmethod
    def _expand_mask(mask: np.ndarray) -> np.ndarray:
        """Dilates binary mask to make erasing process more effective.

        Args:
            mask (np.ndarray): Binary mask array to dilate.

        Returns:
            np.ndarray: The dilated mask array.
        """
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=3)

        return mask

    @staticmethod
    def _apply_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        # Convert grayscale image to RGB
        mask = cv2.merge([mask, mask, mask])
        # Multiply arrays
        masked_img = (image * mask) / 255

        return masked_img

    @staticmethod
    def _extract_object_and_resize(
        image: np.ndarray, mask: np.ndarray, scale_factor: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extracts the object from the image using the mask and resize it.

        Args:
            image (np.ndarray): The input image.
            mask (np.ndarray): The mask for the object to extract.
            scale_factor (float): The factor to scale the object.

        Returns:
            Tuple[np.ndarray, np.ndarray]: The extracted and resized object image and mask.
        """
        # object extraction
        ovary_img = ScaleAugmentation._apply_mask(image, mask)
        ovary_img = Image.fromarray(ovary_img.astype(np.uint8))
        mask = Image.fromarray(mask.astype(np.uint8))

        # object resizing
        orig_width = ovary_img.size[0]
        orig_height = ovary_img.size[1]
        width = int(orig_width * scale_factor)
        height = int(orig_height * scale_factor)

        ovary_img_scaled = ovary_img.resize((width, height))
        mask_scaled = mask.resize((width, height))

        left = (width - orig_width) / 2
        top = (height - orig_height) / 2
        right = (width + orig_width) / 2
        bottom = (height + orig_height) / 2

        ovary_img_scaled = ovary_img_scaled.crop((left, top, right, bottom))
        mask_scaled = mask_scaled.crop((left, top, right, bottom))
        mask_scaled = transforms.ToTensor()(mask_scaled).round().numpy().squeeze()
        mask_scaled[mask_scaled == 1] = 255

        ovary_img_scaled = np.array(ovary_img_scaled)
        mask_scaled = np.array(mask_scaled)

        return ovary_img_scaled, mask_scaled

    @staticmethod
    def _place_new_object(
        image_erased: np.ndarray, ovary_img_scaled: np.ndarray, mask_scaled: np.ndarray
    ) -> np.ndarray:
        """Places the scaled ovary image onto the erased image using the scaled mask.

        Args:
            image_erased (np.ndarray): The image with the original ovary erased.
            ovary_img_scaled (np.ndarray): The resized ovary image to be placed.
            mask_scaled (np.ndarray): The binary mask for the scaled ovary image.

        Returns:
            np.ndarray: The output image with ovary image placed onto erased region.
        """
        mask_scaled = cv2.merge([mask_scaled, mask_scaled, mask_scaled])
        result = np.where(
            mask_scaled == (255, 255, 255), ovary_img_scaled, image_erased
        )

        return result

    @torch.no_grad()
    def _erase(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Performs the inpainting process by padding the input image and mask, passing them through the PyTorch model.

        Args:
            image (np.ndarray): image to process
            mask (np.ndarray): mask with the damaged areas.

        Returns:
            np.ndarray: the resulting image with the missing parts filled in.
        """
        origin_height, origin_width = image.shape[:2]
        mask = self._expand_mask(mask)
        pad_image = self._pad_img(image, mod=self.pad_mod)
        pad_mask = self._pad_img(mask, mod=self.pad_mod)

        result = self._forward(pad_image, pad_mask)
        result = result[0:origin_height, 0:origin_width, :]

        mask = mask[:, :, np.newaxis]
        result = result * (mask / 255) + image[:, :, ::-1] * (1 - (mask / 255))

        result = result.astype("float32")
        result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

        return result

    def __call__(
        self, image: np.ndarray, mask: np.ndarray, scale_factor: float = 1.2
    ) -> np.ndarray:
        """Augments an image by erasing an scaling an object and replacing the original one in the image.

        Args:
            image (np.ndarray): The original input image.
            mask (np.ndarray): The mask defining the object to change.
            scale_factor (float): The factor to scale the object.

        Returns:
            np.ndarray: The augmented image with the scaled object.
        """
        image_erased = self._erase(image, mask)
        ovary_img_scaled, mask_scaled = self._extract_object_and_resize(
            image, mask, scale_factor
        )
        image_augmented = self._place_new_object(
            image_erased, ovary_img_scaled, mask_scaled
        )

        return image_augmented, mask_scaled
