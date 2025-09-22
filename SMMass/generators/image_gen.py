import requests
import os
import uuid
import time
from flask import current_app


class ImageGenerator:
    def __init__(self, huggingface_key):
        self.api_key = huggingface_key
        # Используем более надежную модель
        self.api_url = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-2-1"
        # Альтернативные модели на случай проблем:
        self.backup_models = [
            "https://api-inference.huggingface.co/models/runwayml/stable-diffusion-v1-5",
            "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"
        ]
        self.current_model_index = 0

    def generate_image(self, prompt):
        headers = {"Authorization": f"Bearer {self.api_key}"}

        # Пытаемся использовать разные модели в случае ошибок
        for attempt in range(3):  # Максимум 3 попытки
            try:
                current_api_url = self.api_url if attempt == 0 else self.backup_models[attempt - 1]

                response = requests.post(
                    current_api_url,
                    headers=headers,
                    json={"inputs": prompt, "options": {"wait_for_model": True}}
                )

                # Если модель загружается, ждем и пробуем снова
                if response.status_code == 503:
                    error_msg = response.json().get("error", "")
                    if "loading" in error_msg.lower():
                        estimated_time = response.json().get("estimated_time", 30)
                        time.sleep(estimated_time + 5)  # Ждем немного дольше расчетного времени
                        continue

                if response.status_code != 200:
                    raise Exception(f"Hugging Face API error: {response.status_code} - {response.text}")

                # Генерируем уникальное имя файла
                image_filename = f"{uuid.uuid4()}.png"
                image_path = os.path.join(current_app.root_path, "static", "generated", image_filename)

                # Создаем директорию, если она не существует
                os.makedirs(os.path.dirname(image_path), exist_ok=True)

                # Сохраняем изображение
                with open(image_path, "wb") as f:
                    f.write(response.content)

                # Возвращаем путь к изображению для использования в шаблоне
                return f"/static/generated/{image_filename}"

            except Exception as e:
                if attempt == 2:  # Если это последняя попытка
                    raise Exception(f"Failed to generate image after 3 attempts: {str(e)}")
                time.sleep(5)  # Ждем перед следующей попыткой

        raise Exception("All attempts to generate image failed")