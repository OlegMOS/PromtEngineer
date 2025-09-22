import requests
import logging
from telethon import TelegramClient
from telethon.tl.functions.channels import GetFullChannelRequest
from telethon.tl.types import InputChannel
import asyncio
import os

logger = logging.getLogger(__name__)


class TelegramStats:
    def __init__(self, bot_token, channel_id):
        self.bot_token = bot_token
        self.channel_id = channel_id
        self.base_url = f"https://api.telegram.org/bot{self.bot_token}"

        # Данные для Telethon (Client API)
        self.api_id = os.getenv('TELEGRAM_API_ID', 'YOUR_API_ID')
        self.api_hash = os.getenv('TELEGRAM_API_HASH', 'YOUR_API_HASH')
        self.phone = os.getenv('TELEGRAM_PHONE', 'YOUR_PHONE')

    async def get_channel_stats_async(self):
        """Получение статистики канала через Telethon"""
        stats = {}

        try:
            # Создаем клиент Telethon
            client = TelegramClient('session_name', self.api_id, self.api_hash)
            await client.start(phone=self.phone)

            # Получаем информацию о канале
            if self.channel_id.startswith('@'):
                channel = await client.get_entity(self.channel_id)
            else:
                channel = await client.get_entity(int(self.channel_id))

            # Получаем полную информацию о канале
            full_channel = await client(GetFullChannelRequest(channel=channel))

            # Основная статистика
            stats['Название канала'] = channel.title
            stats['Username'] = f"@{channel.username}" if channel.username else "Отсутствует"
            stats['ID канала'] = channel.id
            stats['Количество подписчиков'] = getattr(channel, 'participants_count', 'Недоступно')

            # Дополнительная информация из full_channel
            if full_channel:
                stats['Описание'] = full_channel.full_chat.about or "Отсутствует"
                stats['Количество сообщений'] = getattr(full_channel.full_chat, 'messages_count', 'Недоступно')

                # Статистика просмотров (если доступна)
                if hasattr(full_channel.full_chat, 'views'):
                    stats['Просмотры'] = full_channel.full_chat.views

            await client.disconnect()

        except Exception as e:
            stats['Ошибка Telethon'] = str('Возвращаем базовую статистику через Bot API')
            # Возвращаем базовую статистику через Bot API
            stats.update(self.get_basic_stats())

        return stats

    def get_channel_stats(self):
        """Синхронный метод для получения статистики"""
        try:
            # Запускаем асинхронную функцию
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            stats = loop.run_until_complete(self.get_channel_stats_async())
            loop.close()
            return stats
        except Exception as e:
            logger.error(f"Error getting Telegram stats: {e}")
            return self.get_basic_stats()

    def get_basic_stats(self):
        """Базовая статистика через Bot API"""
        stats = {}

        try:
            # Информация о канале
            method = "getChat"
            params = {'chat_id': self.channel_id}

            response = requests.post(f"{self.base_url}/{method}", params=params, timeout=10)
            result = response.json()

            if result.get('ok'):
                chat_info = result['result']
                stats['Название канала'] = chat_info.get('title', 'Неизвестно')
                stats['Тип'] = chat_info.get('type', 'Неизвестно')
                stats['ID канала'] = chat_info.get('id', 'Неизвестно')
                stats['Username'] = f"@{chat_info.get('username')}" if chat_info.get('username') else "Отсутствует"
                stats['Описание'] = chat_info.get('description', 'Отсутствует')

                # Проверяем статус бота
                stats['Статус бота'] = self.check_bot_admin_status()

                # Альтернативные методы получения статистики
                member_count = self.try_get_members_count()
                if member_count:
                    stats['Количество подписчиков'] = member_count
                else:
                    stats['Количество подписчиков'] = 'Используйте Telethon API для получения'

            else:
                stats['Ошибка Bot API'] = result.get('description', 'Unknown error')

        except Exception as e:
            stats['Ошибка'] = str(e)

        return stats

    def try_get_members_count(self):
        """Попытка получить количество участников через различные методы"""
        try:
            # Метод 1: getChatMembersCount (устаревший, но может работать)
            method = "getChatMembersCount"
            params = {'chat_id': self.channel_id}

            response = requests.post(f"{self.base_url}/{method}", params=params, timeout=10)
            result = response.json()

            if result.get('ok'):
                return result['result']

        except:
            pass

        return None

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

        except Exception as e:
            logger.error(f"Error checking admin status: {e}")

        return 'Не удалось проверить статус'

    def get_bot_user_id(self):
        """Получает user_id бота"""
        try:
            response = requests.post(f"{self.base_url}/getMe", timeout=10)
            result = response.json()
            if result.get('ok'):
                return result['result']['id']
        except:
            pass
        return None

    def get_setup_instructions(self):
        """Инструкция по настройке Telethon"""
        return """
        Для получения полной статистики подписчиков необходимо настроить Telethon:

        1. Получите API ID и API Hash на https://my.telegram.org
        2. Установите переменные окружения:
           - TELEGRAM_API_ID=ваш_api_id
           - TELEGRAM_API_HASH=ваш_api_hash  
           - TELEGRAM_PHONE=ваш_номер_телефона

        Или используйте бота-аналитику сторонних сервисов для получения статистики.
        """