import requests
import os
import uuid
import time
import logging

logger = logging.getLogger(__name__)


class ImageGenerator:
    def __init__(self, huggingface_key):
        self.api_key = huggingface_key.strip() if huggingface_key else None
        # Список доступных моделей для попытки использования
        self.available_models = [
            "runwayml/stable-diffusion-v1-5",
            "stabilityai/stable-diffusion-2-1",
            "CompVis/stable-diffusion-v1-4"
        ]

    def generate_image(self, prompt):
        if not self.api_key:
            raise Exception("Hugging Face API key is missing. Please check your configuration.")

        # Улучшаем промпт для лучших результатов
        enhanced_prompt = f"{prompt}, high quality, detailed, professional photography, 4k"

        headers = {"Authorization": f"Bearer {self.api_key}"}

        # Пробуем разные модели пока одна не сработает
        for model_name in self.available_models:
            api_url = f"https://api-inference.huggingface.co/models/{model_name}"

            try:
                logger.info(f"Trying model: {model_name}")

                # Добавляем параметры для лучшего контроля
                payload = {
                    "inputs": enhanced_prompt,
                    "parameters": {
                        "num_inference_steps": 20,  # Уменьшаем для экономии ресурсов
                        "guidance_scale": 7.5,
                        "height": 512,
                        "width": 512
                    },
                    "options": {
                        "wait_for_model": True,
                        "use_cache": True
                    }
                }

                response = requests.post(
                    api_url,
                    headers=headers,
                    json=payload,
                    timeout=60
                )

                logger.info(f"API response status for {model_name}: {response.status_code}")

                if response.status_code == 200:
                    # Генерируем уникальное имя файла
                    image_filename = f"{uuid.uuid4()}.png"
                    image_path = os.path.join("app", "static", "generated", image_filename)

                    # Создаем директорию, если она не существует
                    os.makedirs(os.path.dirname(image_path), exist_ok=True)

                    # Сохраняем изображение
                    with open(image_path, "wb") as f:
                        f.write(response.content)

                    logger.info(f"Image successfully generated using {model_name} and saved as {image_filename}")
                    return f"/static/generated/{image_filename}"

                elif response.status_code == 402 or response.status_code == 403:
                    # Ошибки, связанные с оплатой/лимитами
                    error_data = response.json()
                    error_msg = error_data.get("error", "Payment required")
                    logger.error(f"Payment/limit error for {model_name}: {error_msg}")
                    raise Exception(
                        "Hugging Face requires a payment method for inference API usage. Please add a payment method to your account.")

                elif response.status_code == 503:
                    # Модель загружается, пытаемся получить estimated_time
                    try:
                        error_data = response.json()
                        if "estimated_time" in error_data:
                            wait_time = error_data["estimated_time"] + 5
                            logger.info(f"Model {model_name} is loading, waiting {wait_time} seconds")
                            time.sleep(wait_time)
                            continue  # Пробуем снова после ожидания
                    except:
                        logger.warning(f"Model {model_name} is loading, waiting 30 seconds")
                        time.sleep(30)
                        continue

                elif response.status_code == 429:
                    # Слишком много запросов
                    logger.warning(f"Rate limit exceeded for {model_name}, trying next one")
                    time.sleep(10)  # Ждем перед следующей попыткой
                    continue

                else:
                    error_msg = response.text[:500] if response.text else "No error message"
                    logger.error(f"Hugging Face API error {response.status_code} for {model_name}: {error_msg}")
                    continue

            except requests.exceptions.Timeout:
                logger.warning(f"Timeout for model {model_name}, trying next one")
                continue
            except requests.exceptions.ConnectionError:
                logger.warning(f"Connection error for model {model_name}, trying next one")
                continue
            except Exception as e:
                logger.error(f"Unexpected error with model {model_name}: {str(e)}")
                continue

        # Если ни одна модель не сработала
        raise Exception(
            "All models failed. Please add a payment method to your Hugging Face account or try again later.")