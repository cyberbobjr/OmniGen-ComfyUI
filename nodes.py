import gc
import os
import torch
import tempfile
import shutil
import numpy as np
from PIL import Image


def get_vram_info():
    if torch.cuda.is_available():
        t = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        r = torch.cuda.memory_reserved() / (1024**3)
        a = torch.cuda.memory_allocated() / (1024**3)
        f = t - (r + a)
        return f"VRAM: Total {t:.2f}GB | Reserved {r:.2f}GB | Allocated {a:.2f}GB | Free {f:.2f}GB"
    return "CUDA not available"

class OmniGenNode:
    def __init__(self):
        if torch.backends.mps.is_available():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"        
            
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("OMNIGEN_MODEL",),
                "prompt_text": ("STRING", {
                    "multiline": True,
                    "default": "you only need image_1, text will auto be <img><|image_1|></img>",
                    "tooltip": "Enter your prompt text here. For images, use <img><|image_1|></img> syntax"
                }),
                "latent": ("LATENT",),
                "num_inference_steps": ("INT", {"default": 50, "min": 1, "max": 100, "step": 1}),
                "guidance_scale": ("FLOAT", {"default": 2.5, "min": 1.0, "max": 5.0, "step": 0.1}),
                "img_guidance_scale": ("FLOAT", {"default": 1.6, "min": 1.0, "max": 2.0, "step": 0.1}),
                "max_input_image_size": ("INT", {"default": 1024, "min": 128, "max": 2048, "step": 8}),
                "use_input_image_size_as_output": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Automatically adjust output image size to match input image"
                }),
                "seed": ("INT", {"default": 42})
            },
            "optional": {
                "image_1": ("IMAGE",),
                "image_2": ("IMAGE",),
                "image_3": ("IMAGE",)
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "gen"
    CATEGORY = "AIFSH_OmniGen"

    def save_input_img(self,image):
        with tempfile.NamedTemporaryFile(suffix=".png",delete=False,dir="tmp") as f:
            img_np = image.numpy()[0]*255
            img_pil = Image.fromarray(img_np.astype(np.uint8))
            img_pil.save(f.name)
        return f.name

        """ Save input image to a temporary file and return the file path.
        This is necessary because the OmniGen pipeline expects a file path for input images.
        latent: (B, C, H, W) tensor
        """    
    def gen(self, model, prompt_text, latent, num_inference_steps, guidance_scale,
            img_guidance_scale, max_input_image_size,
            use_input_image_size_as_output, seed, image_1=None, image_2=None, image_3=None):
        
        pipe, (store_in_vram) = model
        
        print("\n=== OmniGen Generation ===")
        print(f"Pre-generation {get_vram_info()}")

        # Get dimensions from latent
        height = latent["samples"].shape[2] * 8
        width = latent["samples"].shape[3] * 8
        
        input_images = []
        os.makedirs("tmp", exist_ok=True)
        if image_1 is not None:
            input_images.append(self.save_input_img(image_1))
            prompt_text = prompt_text.replace("image_1","<img><|image_1|></img>")
        
        if image_2 is not None:
            input_images.append(self.save_input_img(image_2))
            prompt_text = prompt_text.replace("image_2","<img><|image_2|></img>")
        
        if image_3 is not None:
            input_images.append(self.save_input_img(image_3))
            prompt_text = prompt_text.replace("image_3","<img><|image_3|></img>")
        
        if len(input_images) == 0:
            input_images = None
            
        print(f"\nGenerating with prompt: {prompt_text}")
        print(f"Before inference: {get_vram_info()}")
        pipe.to(self.device)
        output = pipe(
            input_images=input_images,
            prompt=prompt_text,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            output_type="pil",  # <-- Utilise PIL ici
            img_guidance_scale=img_guidance_scale,
            num_inference_steps=num_inference_steps,
            use_input_image_size_as_output=use_input_image_size_as_output,
            generator=torch.Generator(device="cpu").manual_seed(seed),
            num_images_per_prompt=latent["samples"].shape[0],
            max_input_image_size=max_input_image_size,
        )

        print(f"After inference: {get_vram_info()}")

        # Convertir toutes les images PIL en tenseurs torch et les empiler
        img_tensors = []
        for img_pil in output.images:
            img_np = np.array(img_pil)
            if img_np.ndim == 2:
                img_np = np.expand_dims(img_np, axis=-1)
            if img_np.shape[-1] == 4:
                img_np = img_np[..., :3]
            img_np = img_np.astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(img_np)
            img_tensors.append(img_tensor)
        img_batch = torch.stack(img_tensors, dim=0)
        # S'assure que la sortie est bien [B, H, W, C]
        if img_batch.ndim == 4 and img_batch.shape[1] in [1, 3, 4]:
            img_batch = img_batch.permute(0, 2, 3, 1) if img_batch.shape[1] < img_batch.shape[-1] else img_batch
        if img_batch.shape[-1] not in [3, 4]:
            img_batch = img_batch.permute(0, 2, 3, 1)

        shutil.rmtree("tmp")

        print("Output batch shape:", img_batch.shape, "dtype:", img_batch.dtype)        

        # Clean up if not storing in VRAM
        if not store_in_vram:
            print("Cleaning up pipeline")
            del pipe
            torch.cuda.empty_cache()
            gc.collect()
            print(f"Final state: {get_vram_info()}")
        else:
            print(f"Model kept in VRAM. Final state: {get_vram_info()}")

        return (img_batch,)
