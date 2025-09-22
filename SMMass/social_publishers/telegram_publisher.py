import requests
import os

class TelegramPublisher:
    def __init__(self, bot_token, channel_id):
        self.bot_token = bot_token
        self.channel_id = channel_id
        self.base_url = f"https://api.telegram.org/bot{self.bot_token}"

    def publish_post(self, content, image_path=None):
        if image_path and os.path.exists(image_path):
            # Отправляем пост с изображением (файлом)
            method = "sendPhoto"
            with open(image_path, 'rb') as photo_file:
                files = {'photo': photo_file}
                params = {
                    'chat_id': self.channel_id,
                    'caption': content,
                    'parse_mode': 'HTML'
                }
                response = requests.post(f"{self.base_url}/{method}", files=files, params=params)
        else:
            # Если изображения нет, отправляем только текст
            method = "sendMessage"
            params = {
                'chat_id': self.channel_id,
                'text': content,
                'parse_mode': 'HTML'
            }
            response = requests.post(f"{self.base_url}/{method}", params=params)

        result = response.json()

        if not result.get('ok'):
            raise Exception(f"Telegram API error: {result.get('description')}")
        return result