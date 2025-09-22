import requests

class TelegramPublisher:
    def __init__(self, bot_token, channel_id):
        self.bot_token = bot_token
        self.channel_id = channel_id
        self.base_url = f"https://api.telegram.org/bot{self.bot_token}"

    def publish_post(self, content, image_url=None):
        method = "sendPhoto" if image_url else "sendMessage"
        params = {
            'chat_id': self.channel_id,
            'caption': content if image_url else None,
            'photo': image_url if image_url else None,
            'text': content if not image_url else None,
            'parse_mode': 'HTML'
        }

        # Удаляем None-значения из параметров
        params = {k: v for k, v in params.items() if v is not None}

        response = requests.post(f"{self.base_url}/{method}", params=params)
        result = response.json()

        if not result.get('ok'):
            raise Exception(f"Telegram API error: {result.get('description')}")
        return result