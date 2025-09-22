import requests
import logging

logger = logging.getLogger(__name__)


class TelegramStats:
    def __init__(self, bot_token, channel_id):
        self.bot_token = bot_token
        self.channel_id = channel_id
        self.base_url = f"https://api.telegram.org/bot{self.bot_token}"

    def get_channel_stats(self):
        """Получение статистики канала"""
        stats = {}

        try:
            # Получаем базовую информацию о канале
            method = "getChat"
            params = {'chat_id': self.channel_id}

            response = requests.post(f"{self.base_url}/{method}", params=params, timeout=10)
            result = response.json()

            if result.get('ok'):
                chat_info = result['result']

                # Основная информация о канале
                stats['Название канала'] = chat_info.get('title', 'Неизвестно')
                stats['Тип'] = chat_info.get('type', 'Неизвестно')
                stats['ID канала'] = chat_info.get('id', 'Неизвестно')

                # Количество подписчиков (доступно только для администраторов)
                if 'members_count' in chat_info:
                    stats['Количество подписчиков'] = chat_info['members_count']
                else:
                    stats['Количество подписчиков'] = 'Недоступно (бот не является администратором)'

                # Описание канала
                stats['Описание'] = chat_info.get('description', 'Отсутствует')
                stats['Username'] = chat_info.get('username', 'Отсутствует')

                # Проверяем права бота
                admin_status = self.check_bot_admin_status()
                stats['Статус бота'] = admin_status

                # Если бот администратор, пытаемся получить больше статистики
                if "администратор" in admin_status:
                    # Попытка получить информацию о последних сообщениях
                    recent_stats = self.get_recent_posts_stats()
                    stats.update(recent_stats)
            else:
                error_msg = result.get('description', 'Unknown error')
                stats['Ошибка'] = f"Не удалось получить информацию о канале: {error_msg}"
                stats['Инструкция'] = self.get_setup_instructions()

        except Exception as e:
            stats['Ошибка'] = str(e)
            stats['Инструкция'] = self.get_setup_instructions()

        return stats

    def check_bot_admin_status(self):
        """Проверяет, является ли бот администратором канала"""
        try:
            method = "getChatMember"
            params = {
                'chat_id': self.channel_id,
                'user_id': self.get_bot_user_id()
            }

            response = requests.post(f"{self.base_url}/{method}", params=params, timeout=10)
            result = response.json()

            if result.get('ok'):
                status = result['result'].get('status', 'unknown')
                status_map = {
                    'creator': 'Создатель канала',
                    'administrator': 'Администратор',
                    'member': 'Участник',
                    'left': 'Не участник',
                    'kicked': 'Заблокирован'
                }
                return status_map.get(status, status)

            return 'Не удалось проверить статус'

        except Exception as e:
            return f'Ошибка проверки: {str(e)}'

    def get_bot_user_id(self):
        """Получает user_id бота"""
        try:
            response = requests.post(f"{self.base_url}/getMe", timeout=10)
            result = response.json()
            if result.get('ok'):
                return result['result']['id']
            return None
        except:
            return None

    def get_recent_posts_stats(self):
        """Пытается получить статистику по последним сообщениям (ограниченно)"""
        stats = {}
        try:
            # Для каналов нет прямого API для получения статистики сообщений
            # Можно только проверить, может ли бот отправлять сообщения
            method = "sendChatAction"
            params = {
                'chat_id': self.channel_id,
                'action': 'typing'
            }

            response = requests.post(f"{self.base_url}/{method}", params=params, timeout=5)
            if response.json().get('ok'):
                stats['Возможность публикации'] = '✅ Доступна'
            else:
                stats['Возможность публикации'] = '❌ Не доступна'

        except:
            stats['Возможность публикации'] = 'Неизвестно'

        return stats

    def get_setup_instructions(self):
        """Возвращает инструкцию по настройке"""
        return """
        Для получения полной статистики канала:
        1. Добавьте бота как администратора в ваш канал
        2. Дайте боту права на просмотр участников
        3. Убедитесь, что канал публичный или бот имеет доступ

        Как добавить бота в администраторы:
        1. Откройте настройки канала
        2. Выберите "Администраторы"
        3. Нажмите "Добавить администратора"
        4. Найдите вашего бота по username
        5. Выдайте необходимые права
        """