import torch

class PSBlendNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "blend_mode": (["Normal", "Dissolve", "Darken", "Multiply", "Color Burn", "Linear Burn", "Darker Color", "Lighten", "Screen", "Color Dodge", "Linear Dodge", "Lighter Color", "Overlay", "Soft Light", "Hard Light", "Vivid Light", "Linear Light", "Pin Light", "Hard Mix", "Difference", "Exclusion", "Subtract", "Divide", "Hue", "Saturation", "Color", "Luminosity"],),
                "opacity": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "blend_images"

    CATEGORY = "image/blending"

    def blend_images(self, image1, image2, blend_mode, opacity):
        # GPU availability check
        if torch.cuda.is_available():
            print("PS Blend Node is using CUDA.")
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            print("PS Blend Node is using  MPS.")
            device = torch.device("mps")
        else:
            print("No GPU acceleration available, using CPU")
            device = torch.device("cpu")

        img1 = image1.to(device)
        img2 = image2.to(device)

        # Ensure images have the same shape
        if img1.shape != img2.shape:
            raise ValueError("Images must have the same dimensions. Try running them through resize nodes first.")

        # Helper functions
        def lum(img):
            return 0.3 * img[..., 0] + 0.59 * img[..., 1] + 0.11 * img[..., 2]

        def sat(img):
            min_val, _ = torch.min(img[..., :3], dim=-1)
            max_val, _ = torch.max(img[..., :3], dim=-1)
            return max_val - min_val

        def set_lum(c, l):
            d = l - lum(c)
            c_rgb = c[..., :3] + d.unsqueeze(-1)
            return torch.cat([clip_color(c_rgb), c[..., 3:4]], dim=-1)

        def set_sat(c, s):
            c_rgb = c[..., :3]
            c_min, _ = torch.min(c_rgb, dim=-1, keepdim=True)
            c_max, _ = torch.max(c_rgb, dim=-1, keepdim=True)
            c_mid = c_rgb - c_min
            c_mid *= s.unsqueeze(-1) / (c_max - c_min + 1e-6)
            c_rgb_new = clip_color(c_mid + c_min)
            return torch.cat([c_rgb_new, c[..., 3:4]], dim=-1)

        def clip_color(c):
            l = lum(c).unsqueeze(-1)
            n = torch.min(c, dim=-1, keepdim=True)[0]
            x = torch.max(c, dim=-1, keepdim=True)[0]
            c_clipped = torch.clamp(c, 0, 1)
            c_clipped = torch.where(n < 0, l + ((c - l) * l) / (l - n + 1e-6), c_clipped)
            c_clipped = torch.where(x > 1, l + ((c - l) * (1 - l)) / (x - l + 1e-6), c_clipped)
            return c_clipped

        def hue_blend(img1, img2):
            lum2 = lum(img2)
            sat2 = sat(img2)
            result = set_lum(set_sat(img1, sat2), lum2)
            return result

        # Blend mode calculations
        if blend_mode == "Normal":
            result = img1
        elif blend_mode == "Dissolve":
            mask = torch.rand_like(img1) < opacity
            result = torch.where(mask, img1, img2)
        elif blend_mode == "Darken":
            result = torch.min(img1, img2)
        elif blend_mode == "Multiply":
            result = img1 * img2
        elif blend_mode == "Color Burn":
            result = torch.where(
                img1 == 1, 
                img2,
                torch.where(
                    img1 > 0,
                    1 - torch.clamp((1 - img2) / img1, 0, 1),
                    torch.zeros_like(img2)
                )
            )
        elif blend_mode == "Linear Burn":
            result = img1 + img2 - 1
        elif blend_mode == "Darker Color":
            result = torch.where(lum(img1).unsqueeze(-1) < lum(img2).unsqueeze(-1), img1, img2)
        elif blend_mode == "Lighten":
            result = torch.max(img1, img2)
        elif blend_mode == "Screen":
            result = 1 - (1 - img1) * (1 - img2)
        elif blend_mode == "Color Dodge":
            result = torch.where(img2 < 1, torch.min(torch.ones_like(img1), img1 / (1 - img2)), torch.ones_like(img1))
        elif blend_mode == "Linear Dodge":
            result = img1 + img2
        elif blend_mode == "Lighter Color":
            result = torch.where(lum(img1).unsqueeze(-1) > lum(img2).unsqueeze(-1), img1, img2)
        elif blend_mode == "Overlay":
            result = torch.where(img2 < 0.5, 2 * img1 * img2, 1 - 2 * (1 - img1) * (1 - img2))
        elif blend_mode == "Soft Light":
            d = torch.where(img2 <= 0.25,
                            ((16 * img2 - 12) * img2 + 4) * img2,
                            torch.sqrt(img2))
            result = torch.where(img1 <= 0.5,
                                 img2 - (1 - 2 * img1) * img2 * (1 - img2),
                                 img2 + (2 * img1 - 1) * (d - img2))
        elif blend_mode == "Hard Light":
            result = torch.where(img1 <= 0.5, 2 * img1 * img2, 1 - 2 * (1 - img1) * (1 - img2))
        elif blend_mode == "Vivid Light":
            result = torch.where(img1 < 0.5, 
                                1 - (1 - img2) / (2 * img1 + 1e-8), 
                                img2 / (2 * (1 - img1) + 1e-8))
        elif blend_mode == "Linear Light":
            result = img2 + 2 * img1 - 1
        elif blend_mode == "Pin Light":
            result = torch.where(img1 < 0.5, 
                                torch.min(img2, 2 * img1), 
                                torch.max(img2, 2 * img1 - 1))
        elif blend_mode == "Hard Mix":
            sum_img = img1 + img2
            result = torch.where(sum_img >= 1, torch.ones_like(img1), torch.zeros_like(img1))
        elif blend_mode == "Difference":
            result = torch.abs(img1 - img2)
        elif blend_mode == "Exclusion":
            result = img1 + img2 - 2 * img1 * img2
        elif blend_mode == "Subtract":
            result = img2 - img1
        elif blend_mode == "Divide":
            result = img2 / (img1 + 1e-8)
        elif blend_mode == "Hue":
            result = hue_blend(img1, img2)
        elif blend_mode == "Saturation":
            result = set_lum(set_sat(img2, sat(img1)), lum(img2))
        elif blend_mode == "Color":
            result = set_lum(img1, lum(img2))
        elif blend_mode == "Luminosity":
            result = set_lum(img2, lum(img1))
        else:
            print(f"Unsupported blend mode: {blend_mode}")
            result = img1

        # Apply opacity
        opacity_tensor = torch.tensor(opacity, device=device).view(1, 1, 1, 1)
        result = img2 * (1 - opacity_tensor) + result * opacity_tensor

        # Ensure result is clamped between 0 and 1
        result = torch.clamp(result, 0, 1)

        return (result,)

    # Debug method
    def print_tensor_shape(self, tensor, name):
        print(f"{name} shape: {tensor.shape}")
