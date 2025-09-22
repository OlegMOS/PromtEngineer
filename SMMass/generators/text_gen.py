class PostGenerator:
    def __init__(self, openai_client, tone, topic):
        self.client = openai_client
        self.tone = tone
        self.topic = topic

    def generate_post(self):
        response = self.client.chat.completions.create(
          model="gpt-3.5-turbo",  # Изменено с "gpt-4o" на "gpt-3.5-turbo"
          messages=[
            {"role": "system", "content": "Ты высококвалифицированный SMM специалист, который будет помогать в генерации текста для постов с заданной теме тематикой и заданным тоном."},
            {"role": "user", "content": f"Сгенерируй пост для соцсетей с темой {self.topic}, используя тон: {self.tone}"}
          ]
        )
        return response.choices[0].message.content

    def generate_post_image_description(self):
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": """Ты ассистент, который составляет точные и детальные промпты для нейросети, генерирующей изображения.
                Следуй этим правилам:
                1. Промпт должен быть на английском языке (это важно для качества генерации)
                2. Указывай конкретные детали: стиль, цвет, композицию, фон
                3. Добавляй ключевые слова для улучшения качества: 'professional photography', 'high quality', 'detailed'
                4. Избегай абстрактных описаний
                5. Для предметов одежды указывай: фасон, материал, контекст использования
                6. Указывай стиль изображения: 'product photography', 'fashion photo', 'editorial shot' и т.д."""
                },
                {
                    "role": "user",
                    "content": f"Создай детальный промпт на английском языке для генерации изображения {self.topic}. Укажи все relevant детали: стиль, цвета, композицию, фон. Промпт должен быть на английском."
                }
            ]
        )
        return response.choices[0].message.content