from diffusers import StableDiffusionImg2ImgPipeline
import torch
from PIL import Image
import os

# Проверка доступности CUDA
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Загрузка модели
model_id = "CompVis/stable-diffusion-v1-4"
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to(device)

# Путь к начальному изображению
init_image_path = "my_photo.jpg"  # Замените на путь к вашему фото
init_image = Image.open(init_image_path).convert("RGB").resize((512, 512))

# Параметры
prompts = ["Two men from the photo standing close", "Two men from the photo kissing gently"]
num_frames_per_prompt = 50
strength = 0.75
guidance_scale = 7.5
num_inference_steps = 50
output_dir = 'dreams'
frame_dir = os.path.join(output_dir, 'frames')
output_path = os.path.join(output_dir, 'men_kissing_video.mp4')

# Создание директорий
os.makedirs(frame_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

# Генерация кадров
frames = []
for i, prompt in enumerate(prompts):
    for step in range(num_frames_per_prompt):
        interp_strength = strength * (step / (num_frames_per_prompt - 1)) if step < num_frames_per_prompt - 1 else strength
        image = pipe(prompt, init_image, strength=interp_strength, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale).images[0]
        frame_path = os.path.join(frame_dir, f"frame_{i}_{step}.png")
        image.save(frame_path)
        frames.append(frame_path)

# Сборка видео с помощью ffmpeg
ffmpeg_cmd = [
    'ffmpeg',
    '-framerate', str(24),
    '-i', os.path.join(frame_dir, 'frame_%d_%d.png'),
    '-c:v', 'libx264',
    '-pix_fmt', 'yuv420p',
    output_path
]
os.system(' '.join(ffmpeg_cmd))

print(f"Video saved to {output_path}")