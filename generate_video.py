from diffusers import StableDiffusionImg2ImgPipeline
import torch
import os

try:
    from moviepy.editor import ImageSequenceClip
    print("moviepy.editor imported successfully")
except ImportError as e:
    print(f"Error importing moviepy: {e}. Configuring manually with FFMPEG.")
    import moviepy
    moviepy.config.FFMPEG_BINARY = r"C:\ProgramData\chocolatey\bin\ffmpeg.exe"  # Замените на реальный путь
    from moviepy.editor import ImageSequenceClip
    print("moviepy configured with custom FFMPEG path.")

from PIL import Image
import numpy as np

# Загрузка модели с использованием GPU
model_id = "CompVis/stable-diffusion-v1-4"
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

# Путь к начальному изображению
init_image_path = "my_photo.jpg"  # Замените на путь к вашему фото
init_image = Image.open(init_image_path).convert("RGB").resize((512, 512))

# Параметры
prompts = ["Two men from the photo standing close", "Two men from the photo kissing gently"]
num_frames_per_prompt = 50
strength = 0.75
guidance_scale = 7.5
num_inference_steps = 50
fps = 24
output_dir = 'dreams'
name = 'men_kissing_video'
output_path = os.path.join(output_dir, f"{name}.mp4")

# Создание директории
os.makedirs(output_dir, exist_ok=True)

# Генерация кадров
frames = []
for prompt in prompts:
    for step in range(num_frames_per_prompt):
        interp_strength = strength * (step / (num_frames_per_prompt - 1)) if step < num_frames_per_prompt - 1 else strength
        image = pipe(prompt, init_image, strength=interp_strength, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale).images[0]
        frames.append(np.array(image))

# Сохранение в видео
clip = ImageSequenceClip([Image.fromarray(frame) for frame in frames], fps=fps)
clip.write_videofile(output_path, codec="libx264")

print(f"Video saved to {output_path}")