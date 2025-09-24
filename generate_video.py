from stable_diffusion_videos import walk

# Генерация видео из фото
video_path = walk(
    input_path="photo.png",     # твое фото в папке проекта
    fps=15,                     # кадров в секунду
    num_frames=20,              # длина видео (14–25 кадров)
    motion_bucket_id=90,        # сила движения (40–200)
    output_path="output.mp4"    # куда сохранить видео
)

print(f"Видео сохранено: {video_path}")
