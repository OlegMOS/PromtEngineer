# -*- coding: utf-8 -*-
"""
Виртуальный консультант для ювелирного магазина с использованием RAG
Поддерживает обе базы данных (ChromaDB и FAISS) и работает на CPU с ограниченной памятью
"""

import os
import json
import sys
import re
from typing import List, Dict, Any
from uuid import uuid4

# Установка совместимости с SQLite для ChromaDB (если нужно)
try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

# Импорт библиотек для векторных баз данных
import chromadb
import numpy as np
import faiss

# Исправленные импорты для новых версий LangChain
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

# Для работы с более простой модели
from transformers import pipeline


class JewelryConsultant:
    def __init__(self):
        # Модель для эмбеддингов, которая хорошо работает с русским языком
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="cointegrated/LaBSE-en-ru",
            model_kwargs={'device': 'cpu'}
        )

        # Данные для базы знаний
        self.jewelry_documents = [
            {
                "content": "Серебряные украшения требуют особого ухода. Храните их в сухом месте, избегайте контакта с водой, химикатами и косметикой. Для очистки используйте мягкую ткань и специальные средства для серебра.",
                "metadata": {"type": "уход", "материал": "серебро"}
            },
            {
                "content": "Золотые украшения следует хранить отдельно от других изделий. Для очистки используйте теплый мыльный раствор и мягкую щетку. Избегайте контакта с хлором и другими агрессивными химикатами.",
                "metadata": {"type": "уход", "материал": "золото"}
            }
        ]

        # Каталог товаров
        self.catalog = [
           {
                "name": "Золотое кольцо 'Элегантность'",
                "description": "Классическое золотое кольцо 585 пробы с гравировкой.",
                "usage": "Для повседневной носки.",
                "price": "25,000 рублей.",
                "url": "https://example.com/product/3",
                "material": "золото",
                "category": "кольцо"
            },
            {
                "name": "Серебряное кольцо 'Минимализм'",
                "description": "Простое и элегантное серебряное кольцо с минималистичным дизайном.",
                "usage": "Для повседневной носки.",
                "price": "3,500 рублей.",
                "url": "https://example.com/product/4",
                "material": "серебро",
                "category": "кольцо"
            }
            ,
            {
                "name": "Серебряные серьги 'Лунный свет'",
                "description": "Изящный дизайн.",
                "usage": "Для повседневной носки.",
                "price": "5,500 рублей.",
                "url": "https://example.com/product/4",
                "material": "серебро",
                "category": "серьги"
            }
        ]

        # Инициализация языковой модели
        self.llm = self.setup_llm()

        # Инициализация промпта
        self.prompt = self.setup_prompt()

        # Инициализация цепочки
        self.llm_chain = LLMChain(
            llm=self.llm,
            prompt=self.prompt,
            output_parser=StrOutputParser()
        )

    def setup_llm(self):
        """Настройка языковой модели для работы на CPU с ограниченной памятью"""
        # Используем небольшую модель, которая хорошо работает на CPU и с русским языком
        # Модель rugpt3small достаточно легкая для работы на CPU с 8 ГБ памяти
        text_generator = pipeline(
            "text-generation",
            model="sberbank-ai/rugpt3small_based_on_gpt2",
            tokenizer="sberbank-ai/rugpt3small_based_on_gpt2",
            device=-1,  # -1 для CPU, 0 для GPU
            max_new_tokens=1,  # Уменьшили количество токенов для более точных ответов
            temperature=0.1,  # Уменьшили температуру для более детерминированных ответов
            do_sample=True,
            pad_token_id=50256,  # Добавляем pad_token_id
            repetition_penalty=1.2  # Добавили штраф за повторения
        )

        # Создаем обертку для использования с LangChain
        from langchain_community.llms import HuggingFacePipeline
        return HuggingFacePipeline(pipeline=text_generator)

    def setup_prompt(self):
        """Настройка промпт-шаблона"""
        prompt_template = """
            Ты — виртуальный консультант ювелирного магазина. Отвечай на вопросы клиентов одним предложением, используя только предоставленную информацию.
            Будь точным и конкретным. Если информации для ответа недостаточно, вежливо скажи об этом.
            
            {care_context}
            
            {products_context}
            
            """

        return PromptTemplate(
            input_variables=["care_context", "products_context", "question"],
            template=prompt_template,
        )

    def clean_response(self, response, question):
        """Очистка ответа от промпта, контекста и повторения вопроса"""
        # Удаляем все, что может быть повторением промпта
        patterns_to_remove = [
            r"Ты — виртуальный консультант ювелирного магазина\.",
            r"Отвечай на вопросы клиентов одним предложением, используя только предоставленную информацию\.",
            r"Будь точным и конкретным\.",
            r"Если информации для ответа недостаточно, вежливо скажи об этом\.",
            r"Вопрос:",
            r"Ответ:",
            r"\(только факты из контекста, без повторения вопроса\)",
            # Удаляем повторение вопроса пользователя
            re.escape(question)
        ]

        if not response or not response.strip():
            return "Информации в базе данных недостаточно для ответа на заданный вопрос."

        for pattern in patterns_to_remove:
            response = re.sub(pattern, '', response, flags=re.IGNORECASE)

        # Удаляем лишние пробелы и переносы строк
        response = re.sub(r'\s+', ' ', response).strip()

        # Если ответ начинается с кавычек или других ненужных символов, удаляем их
        response = re.sub(r'^["\':\s]+', '', response)

        # Удаляем повторяющиеся фразы
        response = self.remove_repetitions(response)

        # Если после очистки ответ пустой, возвращаем сообщение о недостатке информации
        if not response or not response.strip():
            return "Информации в базе данных недостаточно для ответа на заданный вопрос."

        return response

    def remove_repetitions(self, text):
        """Удаление повторяющихся фраз из текста"""
        # Если текст пустой, возвращаем сообщение о недостатке информации
        if not text or not text.strip():
            return "Информации в базе данных недостаточно для ответа на заданный вопрос."

        # Разделяем текст на предложения
        sentences = re.split(r'[.!?]', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        # Удаляем дубликаты предложений
        unique_sentences = []
        seen_sentences = set()

        for sentence in sentences:
            # Нормализуем предложение для сравнения (приводим к нижнему регистру и удаляем лишние пробелы)
            normalized = re.sub(r'\s+', ' ', sentence.lower()).strip()

            if normalized not in seen_sentences:
                seen_sentences.add(normalized)
                unique_sentences.append(sentence)

        # Собираем текст обратно
        result = '. '.join(unique_sentences) + '.' if unique_sentences else text

        # Если результат пустой, возвращаем сообщение о недостатке информации
        if not result or not result.strip():
            return "Информации в базе данных недостаточно для ответа на заданный вопрос."

        return result

    def setup_vector_db(self, db_type="chroma"):
        """Настройка векторной базы данных (ChromaDB или FAISS)"""
        if db_type == "chroma":
            return self.setup_chromadb()
        else:
            return self.setup_faiss()

    def setup_chromadb(self):
        """Настройка ChromaDB базы данных"""
        # Клиент ChromaDB
        chroma_client = chromadb.PersistentClient(path="./chroma_db")

        # Коллекция для информации по уходу
        support_collection = chroma_client.get_or_create_collection(name="support_collection")

        # Добавление документов по уходу
        for i, doc in enumerate(self.jewelry_documents):
            support_collection.add(
                ids=[f"doc_{i}"],
                documents=[doc["content"]],
                metadatas=[doc["metadata"]]
            )

        # Коллекция для каталога товаров
        catalog_collection = chroma_client.get_or_create_collection(name="catalog_collection")

        # Добавление товаров в каталог
        for i, item in enumerate(self.catalog):
            text = f"{item['name']}: {item['description']} Назначение: {item['usage']} Цена: {item['price']}"
            catalog_collection.add(
                ids=[f"prod_{i}"],
                documents=[text],
                metadatas=[{
                    "name": item['name'],
                    "price": item['price'],
                    "url": item['url'],
                    "material": item['material'],
                    "category": item['category']
                }]
            )

        return {
            "type": "chroma",
            "support": support_collection,
            "catalog": catalog_collection
        }

    def setup_faiss(self):
        """Настройка FAISS базы данных"""
        # Создание индексов FAISS
        dimension = 768  # Размерность эмбеддингов LaBSE

        # Индекс для информации по уходу
        care_index = faiss.IndexFlatL2(dimension)
        care_documents = []
        care_metadatas = []

        # Индекс для каталога товаров
        catalog_index = faiss.IndexFlatL2(dimension)
        catalog_documents = []
        catalog_metadatas = []

        # Добавление документов по уходу
        for doc in self.jewelry_documents:
            embedding = self.embedding_model.embed_query(doc["content"])
            care_index.add(np.array([embedding]).astype('float32'))
            care_documents.append(doc["content"])
            care_metadatas.append(doc["metadata"])

        # Добавление товаров в каталог
        for item in self.catalog:
            text = f"{item['name']}: {item['description']} Назначение: {item['usage']} Цена: {item['price']}"
            embedding = self.embedding_model.embed_query(text)
            catalog_index.add(np.array([embedding]).astype('float32'))
            catalog_documents.append(text)
            catalog_metadatas.append({
                "name": item['name'],
                "price": item['price'],
                "url": item['url'],
                "material": item['material'],
                "category": item['category']
            })

        return {
            "type": "faiss",
            "care_index": care_index,
            "care_documents": care_documents,
            "care_metadatas": care_metadatas,
            "catalog_index": catalog_index,
            "catalog_documents": catalog_documents,
            "catalog_metadatas": catalog_metadatas
        }

    def query_database(self, db, query_text, n_results=3, search_type="care", material_filter=None,
                       category_filter=None):
        """Поиск релевантных документов в выбранной базе данных"""
        if db["type"] == "chroma":
            if search_type == "care":
                # Добавляем фильтр по материалу, если указан
                where_filter = {"материал": material_filter} if material_filter else None

                results = db["support"].query(
                    query_texts=[query_text],
                    n_results=n_results,
                    where=where_filter
                )
                return results["documents"][0] if results["documents"] else []
            else:
                # Для каталога собираем фильтры с использованием оператора $and
                where_filters = []
                if material_filter:
                    where_filters.append({"material": material_filter})
                if category_filter:
                    where_filters.append({"category": category_filter})

                # Если есть несколько фильтров, объединяем их с помощью $and
                if len(where_filters) > 1:
                    where_filter = {"$and": where_filters}
                elif len(where_filters) == 1:
                    where_filter = where_filters[0]
                else:
                    where_filter = None

                # Если есть фильтры, применяем их
                if where_filter:
                    results = db["catalog"].query(
                        query_texts=[query_text],
                        n_results=n_results,
                        where=where_filter
                    )
                else:
                    results = db["catalog"].query(
                        query_texts=[query_text],
                        n_results=n_results
                    )
                return results["documents"][0] if results["documents"] else []
        else:
            # Поиск в FAISS
            query_embedding = self.embedding_model.embed_query(query_text)

            if search_type == "care":
                distances, indices = db["care_index"].search(
                    np.array([query_embedding]).astype('float32'),
                    n_results
                )
                # Фильтруем документы по материалу, если указан фильтр
                filtered_docs = []
                for i in indices[0]:
                    if i < len(db["care_documents"]):
                        doc = db["care_documents"][i]
                        # Если указан фильтр по материалу, проверяем соответствие
                        if material_filter:
                            material_keywords = {
                                "серебро": "серебр",
                                "золото": "золот",
                                "жемчуг": "жемчуг",
                                "бриллиант": "брил"
                            }
                            if material_filter in material_keywords and material_keywords[
                                material_filter] in doc.lower():
                                filtered_docs.append(doc)
                        else:
                            filtered_docs.append(doc)
                return filtered_docs
            else:
                distances, indices = db["catalog_index"].search(
                    np.array([query_embedding]).astype('float32'),
                    n_results
                )
                # Фильтруем результаты по материалу и категории
                filtered_docs = []
                for i in indices[0]:
                    if i < len(db["catalog_documents"]):
                        doc = db["catalog_documents"][i]
                        metadata = db["catalog_metadatas"][i]

                        # Проверяем фильтры
                        material_match = True
                        category_match = True

                        if material_filter:
                            material_keywords = {
                                "серебро": "серебр",
                                "золото": "золот"
                            }
                            material_match = (material_filter in material_keywords and
                                              material_keywords[material_filter] in doc.lower())

                        if category_filter:
                            category_keywords = {
                                "кольцо": "кольцо",
                                "серьги": "серьги"
                            }
                            category_match = (category_filter in category_keywords and
                                              category_keywords[category_filter] in doc.lower())

                        if material_match and category_match:
                            filtered_docs.append(doc)

                return filtered_docs

    def get_response(self, db, question):
        """Получение ответа от консультанта"""
        # Определяем тип запроса
        care_keywords = ["уход", "ухаживать", "чистить", "хранить", "очистка"]
        product_keywords = ["посоветуй", "рекомендуй", "купить", "товар", "украшение", "стоимость", "цена",
                            "сколько стоит"]

        has_care_query = any(keyword in question.lower() for keyword in care_keywords)
        has_product_query = any(keyword in question.lower() for keyword in product_keywords)

        # Определяем материал, о котором идет речь
        material_filter = None
        if "серебр" in question.lower():
            material_filter = "серебро"
        elif "золот" in question.lower():
            material_filter = "золото"
        elif "жемчуг" in question.lower():
            material_filter = "жемчуг"
        elif "брил" in question.lower():
            material_filter = "бриллиант"

        # Определяем категорию, о которой идет речь
        category_filter = None
        if "кольц" in question.lower():
            category_filter = "кольцо"
        elif "серьг" in question.lower() or "серёж" in question.lower():
            category_filter = "серьги"
        elif "браслет" in question.lower():
            category_filter = "браслет"
        elif "ожерель" in question.lower() or "кулон" in question.lower():
            category_filter = "ожерелье"

        # Для вопросов про уход за конкретным материалом используем специальный фильтр
        if has_care_query and material_filter:
            # Поиск только информации о конкретном материале
            care_context = self.query_database(db, question, 3, "care", material_filter)
            products_context = []  # Не включаем информацию о товарах
        elif has_product_query and (material_filter or category_filter):
            # Для вопросов о товарах с указанием материала или категории применяем фильтры
            care_context = []  # Не включаем информацию об уходе
            products_context = self.query_database(
                db, question, 3, "catalog", material_filter, category_filter
            )
        else:
            # Обычный поиск
            care_context = self.query_database(db, question, 3, "care") if has_care_query else []
            products_context = self.query_database(db, question, 2, "catalog") if has_product_query else []

        # Если запрос не определен, ищем в обоих типах данных
        if not has_care_query and not has_product_query:
            care_context = self.query_database(db, question, 2, "care")
            products_context = self.query_database(db, question, 1, "catalog")

        # Форматирование контекста
        care_text = "\n".join(care_context) if care_context else ""
        products_text = "\n".join(products_context) if products_context else ""

        # Генерация ответа с использованием invoke вместо run
        response = self.llm_chain.invoke({
            "care_context": care_text,
            "products_context": products_text,
            "question": question
        })

        # Извлекаем текст ответа
        response_text = response["text"] if isinstance(response, dict) else str(response)

        # Очистка ответа от промпта и контекста
        cleaned_response = self.clean_response(response_text, question)

        # Проверка на пустой ответ
        if not cleaned_response or not cleaned_response.strip():
            return "Информации в базе данных недостаточно для ответа на заданный вопрос."

        return cleaned_response

    def interactive_mode(self):
        """Интерактивный режим работы с консультантом"""
        print("=== ВИРТУАЛЬНЫЙ КОНСУЛЬТАНТ ЮВЕЛИРНОГО МАГАЗИНА ===")
        print("Выберите тип базы данных:")
        print("1. ChromaDB")
        print("2. FAISS")

        db_choice = input("Введите номер (1 или 2): ").strip()
        db_type = "chroma" if db_choice == "1" else "faiss"

        # Инициализация базы данных
        print(f"Инициализация {db_type.upper()} базы данных...")
        db = self.setup_vector_db(db_type)

        print("База данных готова к работе!")
        print("Введите ваш вопрос (или 'выход' для завершения):")

        while True:
            user_input = input("\nВаш вопрос: ").strip()

            if user_input.lower() in ['выход', 'exit', 'quit']:
                print("Завершение работы виртуального консультанта.")
                break

            if user_input:
                try:
                    response = self.get_response(db, user_input)
                    print(f"\nКонсультант: {response}")
                except Exception as e:
                    print(f"Произошла ошибка: {e}")


# Запуск консультанта
if __name__ == "__main__":
    consultant = JewelryConsultant()
    consultant.interactive_mode()