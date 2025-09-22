import requests


class TelegramStats:
    def __init__(self, bot_token, channel_id):
        self.bot_token = bot_token
        self.channel_id = channel_id
        self.base_url = f"https://api.telegram.org/bot{self.bot_token}"

    def get_channel_stats(self):
        """Получение базовой статистики канала"""
        # Для получения статистики канала нужно, чтобы бот был администратором
        # и использовались специальные методы API

        stats = {
            "Подписчики": "Недоступно (требуются права администратора)",
            "Просмотры постов": "Недоступно",
            "Активность": "Недоступно"
        }

        # Попытка получить информацию о канале
        try:
            method = "getChat"
            params = {'chat_id': self.channel_id}

            response = requests.post(f"{self.base_url}/{method}", params=params)
            result = response.json()

            if result.get('ok'):
                chat_info = result['result']
                if 'members_count' in chat_info:
                    stats['Подписчики'] = chat_info['members_count']
                stats['Название канала'] = chat_info.get('title', 'Неизвестно')
                stats['Описание'] = chat_info.get('description', 'Отсутствует')
        except Exception as e:
            stats['Ошибка'] = str(e)

        return stats