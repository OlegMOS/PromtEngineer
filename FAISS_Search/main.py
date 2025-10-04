import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import re
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
import nest_asyncio
import asyncio
import os
from dotenv import load_dotenv
import pickle
import hashlib
import time

# Загрузка переменных окружения из .env файла
load_dotenv()

# Активация nest_asyncio для работы в Jupyter/Colab
nest_asyncio.apply()


class ResumeSearchEngine:
    def __init__(self, excel_file_path):
        self.df = pd.read_excel(excel_file_path)
        self.model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
        self.index = None
        self.resume_texts = []
        self.cache_dir = "vector_cache"
        self.excel_file_path = excel_file_path

        # Создаем директорию для кэша, если её нет
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

    def _get_cache_key(self):
        """Генерирует ключ кэша на основе содержимого файла и модели"""
        # Хэш содержимого файла
        file_hash = hashlib.md5()
        with open(self.excel_file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                file_hash.update(chunk)

        # Хэш имени модели
        model_hash = hashlib.md5(self.model.get_sentence_embedding_dimension().__str__().encode()).hexdigest()[:8]

        cache_key = f"{file_hash.hexdigest()}_{model_hash}"
        return cache_key

    def _cache_exists(self):
        """Проверяет существование кэша"""
        cache_key = self._get_cache_key()
        index_path = os.path.join(self.cache_dir, f"{cache_key}.faiss")
        texts_path = os.path.join(self.cache_dir, f"{cache_key}_texts.pkl")
        metadata_path = os.path.join(self.cache_dir, f"{cache_key}_metadata.pkl")

        return all(os.path.exists(path) for path in [index_path, texts_path, metadata_path])

    def _save_to_cache(self, embeddings):
        """Сохраняет векторные представления и индекс в кэш"""
        cache_key = self._get_cache_key()

        # Сохраняем FAISS индекс
        index_path = os.path.join(self.cache_dir, f"{cache_key}.faiss")
        faiss.write_index(self.index, index_path)

        # Сохраняем тексты резюме
        texts_path = os.path.join(self.cache_dir, f"{cache_key}_texts.pkl")
        with open(texts_path, 'wb') as f:
            pickle.dump(self.resume_texts, f)

        # Сохраняем метаданные
        metadata_path = os.path.join(self.cache_dir, f"{cache_key}_metadata.pkl")
        metadata = {
            'timestamp': time.time(),
            'data_shape': embeddings.shape,
            'model_name': 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
            'num_resumes': len(self.resume_texts)
        }
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)

        print(f"Векторные представления сохранены в кэш: {cache_key}")

    def _load_from_cache(self):
        """Загружает векторные представления и индекс из кэша"""
        cache_key = self._get_cache_key()

        try:
            # Загружаем FAISS индекс
            index_path = os.path.join(self.cache_dir, f"{cache_key}.faiss")
            self.index = faiss.read_index(index_path)

            # Загружаем тексты резюме
            texts_path = os.path.join(self.cache_dir, f"{cache_key}_texts.pkl")
            with open(texts_path, 'rb') as f:
                self.resume_texts = pickle.load(f)

            # Загружаем метаданные для проверки
            metadata_path = os.path.join(self.cache_dir, f"{cache_key}_metadata.pkl")
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)

            print(f"Векторные представления загружены из кэша: {cache_key}")
            print(f"Загружено {len(self.resume_texts)} резюме, размерность: {self.index.d}")
            return True

        except Exception as e:
            print(f"Ошибка загрузки из кэша: {e}")
            return False

    def preprocess_data(self):
        """Предобработка и нормализация данных"""
        print("Предобработка данных...")

        # Заполнение пропущенных значений
        text_columns = ['positionName', 'experience', 'educationList', 'workExperienceList',
                        'hardSkills', 'softSkills', 'scheduleType', 'busyType']

        for col in text_columns:
            if col in self.df.columns:
                self.df[col] = self.df[col].fillna('')

        # Создание полного текста для каждого резюме
        for idx, row in self.df.iterrows():
            resume_text = f"Должность: {row.get('positionName', '')}. "

            if row.get('experience', ''):
                resume_text += f"Опыт: {row['experience']} лет. "

            # Образование
            if pd.notna(row.get('educationList')) and row['educationList']:
                edu_text = str(row['educationList']).replace('[', '').replace(']', '').replace('{', '').replace('}', '')
                resume_text += f"Образование: {edu_text}. "

            # Опыт работы
            if pd.notna(row.get('workExperienceList')) and row['workExperienceList']:
                exp_text = str(row['workExperienceList'])[:500]  # Ограничиваем длину
                resume_text += f"Опыт работы: {exp_text}. "

            # Навыки
            if pd.notna(row.get('hardSkills')) and row['hardSkills']:
                skills_text = str(row['hardSkills']).replace('[', '').replace(']', '').replace('{', '').replace('}', '')
                resume_text += f"Навыки: {skills_text}. "

            # Местоположение
            if pd.notna(row.get('localityName')) and row['localityName']:
                location = str(row['localityName']).replace('-', ' ')
                resume_text += f"Местоположение: {location}. "

            self.resume_texts.append(resume_text)

        print(f"Обработано {len(self.resume_texts)} резюме")

    def create_faiss_index(self):
        """Создание или загрузка векторной базы данных FAISS"""

        # Пытаемся загрузить из кэша
        if self._cache_exists():
            if self._load_from_cache():
                return

        print("Создание новых векторных представлений...")

        # Создание эмбеддингов
        embeddings = self.model.encode(self.resume_texts, show_progress_bar=True)

        # Нормализация векторов для косинусного сходства
        faiss.normalize_L2(embeddings)

        # Создание FAISS индекса
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # IndexFlatIP для косинусного сходства
        self.index.add(embeddings.astype('float32'))

        # Сохраняем в кэш
        self._save_to_cache(embeddings)

        print(f"Индекс FAISS создан с {self.index.ntotal} векторами")

    def clear_old_cache(self, max_age_days=30):
        """Очистка устаревших кэш-файлов"""
        current_time = time.time()
        max_age_seconds = max_age_days * 24 * 60 * 60

        for filename in os.listdir(self.cache_dir):
            filepath = os.path.join(self.cache_dir, filename)
            if os.path.isfile(filepath):
                file_age = current_time - os.path.getmtime(filepath)
                if file_age > max_age_seconds:
                    os.remove(filepath)
                    print(f"Удален устаревший кэш: {filename}")

    def keyword_search(self, query, k=10):
        """Поиск по ключевым словам"""
        query_embedding = self.model.encode([query])
        faiss.normalize_L2(query_embedding)

        # Поиск в FAISS
        scores, indices = self.index.search(query_embedding.astype('float32'), k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.df):
                results.append({
                    'index': idx,
                    'score': float(score),
                    'type': 'keyword'
                })

        return results

    def location_search(self, location_query, k=10):
        """Поиск по геолокации"""
        location_results = []

        for idx, row in self.df.iterrows():
            if pd.notna(row.get('localityName')):
                location = str(row['localityName']).lower().replace('-', ' ')
                query = location_query.lower()

                # Простой поиск по вхождению
                if query in location:
                    score = 1.0
                else:
                    # Вычисление сходства на основе совпадающих слов
                    location_words = set(location.split())
                    query_words = set(query.split())
                    common_words = location_words.intersection(query_words)

                    if common_words:
                        score = len(common_words) / len(query_words)
                    else:
                        score = 0.0

                if score > 0:
                    location_results.append({
                        'index': idx,
                        'score': score,
                        'type': 'location'
                    })

        # Сортировка и возврат топ-k результатов
        location_results.sort(key=lambda x: x['score'], reverse=True)
        return location_results[:k]

    def hybrid_search(self, keyword_query, location_query, k=5, keyword_weight=0.3, location_weight=0.7):
        """Гибридный поиск: сначала по локации, затем по профессии"""
        print(f"Поиск: '{keyword_query}' в локации '{location_query}'")

        # Сначала ищем по локации
        location_results = self.location_search(location_query, k=50)  # Берем больше результатов для фильтрации

        if not location_results:
            # Если нет результатов по локации, ищем только по ключевым словам
            print("Не найдено резюме в указанной локации, использую поиск только по профессии")
            keyword_results = self.keyword_search(keyword_query, k=k)

            # Преобразуем в тот же формат
            final_results = []
            for result in keyword_results:
                final_results.append({
                    'index': result['index'],
                    'keyword_score': result['score'],
                    'location_score': 0,
                    'final_score': result['score'] * keyword_weight
                })

            return final_results[:k]

        # Фильтруем результаты по локации (оставляем только с достаточно высоким score локации)
        min_location_score = 0.3  # Минимальный порог совпадения по локации
        filtered_location_results = [r for r in location_results if r['score'] >= min_location_score]

        if not filtered_location_results:
            # Если после фильтрации не осталось результатов, ослабляем критерий
            filtered_location_results = location_results[:10]  # Берем топ-10 по локации

        # Среди отфильтрованных по локации резюме ищем по ключевым словам
        location_indices = [r['index'] for r in filtered_location_results]

        # Выполняем поиск по ключевым словам только среди этих индексов
        keyword_results = self._keyword_search_among_indices(keyword_query, location_indices, k=len(location_indices))

        # Объединяем результаты
        combined_results = {}

        # Создаем словарь для быстрого доступа к результатам локации
        location_dict = {r['index']: r for r in filtered_location_results}

        # Объединяем score
        for keyword_result in keyword_results:
            idx = keyword_result['index']
            if idx in location_dict:
                location_score = location_dict[idx]['score']
                keyword_score = keyword_result['score']

                # Нормализуем scores
                normalized_keyword = keyword_score
                normalized_location = location_score

                combined_results[idx] = {
                    'index': idx,
                    'keyword_score': normalized_keyword,
                    'location_score': normalized_location,
                    'final_score': (normalized_keyword * keyword_weight + normalized_location * location_weight)
                }

        # Если после объединения мало результатов, добавляем резюме только по локации
        if len(combined_results) < k:
            for location_result in filtered_location_results:
                idx = location_result['index']
                if idx not in combined_results:
                    combined_results[idx] = {
                        'index': idx,
                        'keyword_score': 0,
                        'location_score': location_result['score'],
                        'final_score': location_result['score'] * location_weight
                    }

        # Сортировка по итоговому score
        final_results = sorted(combined_results.values(),
                               key=lambda x: (x['final_score'], x['location_score'], x['keyword_score']),
                               reverse=True)

        return final_results[:k]

    def _keyword_search_among_indices(self, query, indices, k=10):
        """Поиск по ключевым словам только среди указанных индексов"""
        if not indices:
            return []

        query_embedding = self.model.encode([query])
        faiss.normalize_L2(query_embedding)

        # Получаем все эмбеддинги для указанных индексов
        all_embeddings = []
        valid_indices = []

        # Создаем временный индекс только для выбранных резюме
        dimension = self.index.d
        temp_index = faiss.IndexFlatIP(dimension)

        for idx in indices:
            if idx < self.index.ntotal:
                # Получаем эмбеддинг для этого индекса
                embedding = self.index.reconstruct(idx).reshape(1, -1)
                temp_index.add(embedding.astype('float32'))
                valid_indices.append(idx)

        if temp_index.ntotal == 0:
            return []

        # Ищем в временном индексе
        scores, temp_indices = temp_index.search(query_embedding.astype('float32'), min(k, temp_index.ntotal))

        results = []
        for score, temp_idx in zip(scores[0], temp_indices[0]):
            if temp_idx < len(valid_indices):
                original_idx = valid_indices[temp_idx]
                results.append({
                    'index': original_idx,
                    'score': float(score),
                    'type': 'keyword'
                })

        return results

    def get_resume_details(self, index):
        """Получение деталей резюме по индексу"""
        if index >= len(self.df):
            return None

        row = self.df.iloc[index]

        details = {
            'id': row.get('id', ''),
            'position': row.get('positionName', 'Не указано'),
            'location': row.get('localityName', 'Не указано'),
            'age': row.get('age', 'Не указано'),
            'experience': row.get('experience', 'Не указано'),
            'education': self.extract_education(row.get('educationList', '')),
            'skills': self.extract_skills(row.get('hardSkills', '')),
            'schedule': row.get('scheduleType', 'Не указано'),
            'salary': row.get('salary', 'Не указано'),
            'relocation': row.get('relocation', 'Не указано')
        }

        return details

    def extract_education(self, education_data):
        """Извлечение информации об образовании"""
        if not education_data or education_data == '[]':
            return "Не указано"

        try:
            if isinstance(education_data, str) and 'instituteName' in education_data:
                # Упрощенное извлечение названия института
                match = re.search(r"'instituteName': '([^']*)'", str(education_data))
                if match:
                    return match.group(1)
            return str(education_data)[:100] + "..." if len(str(education_data)) > 100 else str(education_data)
        except:
            return "Информация об образовании"

    def extract_skills(self, skills_data):
        """Извлечение информации о навыках"""
        if not skills_data or skills_data == '[]':
            return "Не указано"

        try:
            if isinstance(skills_data, str) and 'hardSkillName' in skills_data:
                # Извлечение названий навыков
                skills = re.findall(r"'hardSkillName': '([^']*)'", str(skills_data))
                if skills:
                    return ", ".join(skills[:5])  # Ограничиваем количество навыков
            return str(skills_data)[:100] + "..." if len(str(skills_data)) > 100 else str(skills_data)
        except:
            return "Профессиональные навыки"


# Получение токена из переменных окружения
TELEGRAM_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')

if not TELEGRAM_TOKEN:
    raise ValueError("TELEGRAM_BOT_TOKEN не найден в .env файле")

# Инициализация поисковой системы
print("Инициализация поисковой системы...")
search_engine = ResumeSearchEngine('База данных профессий.xlsx')
search_engine.preprocess_data()
search_engine.create_faiss_index()

# Очистка устаревшего кэша (старше 30 дней)
search_engine.clear_old_cache(max_age_days=30)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /start"""
    welcome_text = """
🤖 Добро пожаловать в бот поиска резюме!

Для поиска отправьте сообщение в формате: водитель; Москва; 5
    """
    await update.message.reply_text(welcome_text)


async def handle_search(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик поисковых запросов"""
    try:
        user_input = update.message.text
        parts = user_input.split(';')

        if len(parts) < 3:
            await update.message.reply_text("❌ Неверный формат. Используйте: [должность]; [город]; [количество]")
            return

        keyword = parts[0].strip()
        location = parts[1].strip()

        try:
            k = int(parts[2].strip())
            k = min(k, 10)  # Ограничение максимум 10 результатов
        except:
            k = 5

        if not keyword or not location:
            await update.message.reply_text("❌ Укажите должность и местоположение")
            return

        # Выполнение поиска
        await update.message.reply_text("🔍 Ищу подходящие резюме...")

        results = search_engine.hybrid_search(keyword, location, k=k)

        if not results:
            await update.message.reply_text("❌ Подходящих резюме не найдено")
            return

        # Формирование ответа
        response = f"📊 Найдено резюме: {len(results)}\n\n"

        for i, result in enumerate(results, 1):
            details = search_engine.get_resume_details(result['index'])

            if details:
                response += f"🏆 **Резюме #{i}** (score: {result['final_score']:.2f})\n"
                response += f"💼 Должность: {details['position']}\n"
                response += f"📍 Местоположение: {details['location']}\n"
                response += f"👤 Возраст: {details['age']}\n"
                response += f"📅 Опыт: {details['experience']} лет\n"
                response += f"🎓 Образование: {details['education']}\n"
                response += f"💰 Зарплата: {details['salary']}\n"
                response += f"🕒 График: {details['schedule']}\n"
                response += f"🚗 Переезд: {details['relocation']}\n"

                if i < len(results):
                    response += "─" * 30 + "\n\n"

        # Разделение сообщения если слишком длинное
        if len(response) > 4096:
            parts = [response[i:i + 4096] for i in range(0, len(response), 4096)]
            for part in parts:
                await update.message.reply_text(part)
        else:
            await update.message.reply_text(response)

    except Exception as e:
        await update.message.reply_text(f"❌ Произошла ошибка: {str(e)}")


async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик ошибок"""
    print(f"Ошибка: {context.error}")
    await update.message.reply_text("❌ Произошла ошибка при обработке запроса")


def main():
    """Основная функция запуска бота"""
    # Создание приложения
    application = Application.builder().token(TELEGRAM_TOKEN).build()

    # Добавление обработчиков
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_search))
    application.add_error_handler(error_handler)

    # Запуск бота
    print("Бот запущен...")
    application.run_polling()


# Демонстрация работы поиска
if __name__ == "__main__":
    # Пример поиска
    print("Демонстрация работы поисковой системы:")

    # Пример 1: Поиск водителей в Москве
    results = search_engine.hybrid_search("водитель", "Москва", k=3)
    print(f"\nНайдено {len(results)} резюме водителей в Москве:")
    for i, result in enumerate(results, 1):
        details = search_engine.get_resume_details(result['index'])
        print(f"{i}. {details['position']} - {details['location']} (score: {result['final_score']:.2f})")

    # Пример 2: Поиск продавцов в Санкт-Петербурге
    results = search_engine.hybrid_search("продавец", "Санкт-Петербург", k=2)
    print(f"\nНайдено {len(results)} резюме продавцов в Санкт-Петербурге:")
    for i, result in enumerate(results, 1):
        details = search_engine.get_resume_details(result['index'])
        print(f"{i}. {details['position']} - {details['location']} (score: {result['final_score']:.2f})")

    # Запуск бота (раскомментируйте для использования)
    main()