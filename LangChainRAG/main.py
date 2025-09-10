# -*- coding: utf-8 -*-
"""
Виртуальный консультант для ювелирного магазина с использованием RAG
Поддерживает обе базы данных (FAISS и ChromaDB) и работает на CPU с ограниченной памятью
"""

import os
import json
import sys
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

# Для работы с более простой моделью
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
            },
            {
                "content": "Жемчужные украшения боятся сухости и прямых солнечных лучей. Храните их в отдельной шкатулке с мягкой тканью. Для очистки используйте только влажную мягкую ткань.",
                "metadata": {"type": "уход", "материал": "жемчуг"}
            }
        ]

        # Каталог товаров
        self.catalog = [
            {
                "name": "Кольцо с бриллиантом 'Нежность'",
                "description": "Изготовлено из белого золота 585 пробы, содержит бриллиант 0.5 карата.",
                "usage": "Идеально для помолвки",
                "price": "45,000 рублей",
                "url": "https://example.com/product/1"
            },
            {
                "name": "Серебряные серьги 'Лунный свет'",
                "description": "Изящный дизайн с фианитами. Подходят для повседневной носки.",
                "usage": "Для повседневного использования и особых случаев",
                "price": "5,500 рублей",
                "url": "https://example.com/product/2"
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
            max_new_tokens=150,
            temperature=0.7,
            do_sample=True,
            pad_token_id=50256  # Добавляем pad_token_id
        )

        # Создаем обертку для использования с LangChain
        from langchain.llms import HuggingFacePipeline
        return HuggingFacePipeline(pipeline=text_generator)

    def setup_prompt(self):
        """Настройка промпт-шаблона"""
        prompt_template = """
Ты — виртуальный консультант ювелирного магазина. Твои задачи:
1. Консультировать по уходу за ювелирными изделиями на основе предоставленного контекста.
2. Рекомендовать товары из ассортимента магазина на основе запросов клиентов.
3. Быть вежливым, дружелюбным и профессиональным.

Отвечай только на основе предоставленного контекста. Если в контексте нет информации для ответа, 
вежливо сообщи, что не можешь помочь с этим вопросом и предложи уточнить запрос или обратиться 
к менеджеру магазина.

Контекст для консультации по уходу:
{care_context}

Контекст с товарами:
{products_context}

Вопрос: {question}

Ответ:
"""

        return PromptTemplate(
            input_variables=["care_context", "products_context", "question"],
            template=prompt_template,
        )

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
                    "url": item['url']
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
                "url": item['url']
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

    def query_database(self, db, query_text, n_results=3, search_type="care"):
        """Поиск релевантных документов в выбранной базе данных"""
        if db["type"] == "chroma":
            if search_type == "care":
                results = db["support"].query(
                    query_texts=[query_text],
                    n_results=n_results
                )
                return results["documents"][0] if results["documents"] else []
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
                return [db["care_documents"][i] for i in indices[0] if i < len(db["care_documents"])]
            else:
                distances, indices = db["catalog_index"].search(
                    np.array([query_embedding]).astype('float32'),
                    n_results
                )
                return [db["catalog_documents"][i] for i in indices[0] if i < len(db["catalog_documents"])]

    def get_response(self, db, question):
        """Получение ответа от консультанта"""
        # Поиск релевантной информации
        care_context = self.query_database(db, question, 3, "care")
        products_context = self.query_database(db, question, 2, "catalog")

        # Форматирование контекста
        care_text = "\n".join(care_context) if care_context else "Информация по уходу отсутствует."
        products_text = "\n".join(products_context) if products_context else "Информация о товарах отсутствует."

        # Генерация ответа
        response = self.llm_chain.run({
            "care_context": care_text,
            "products_context": products_text,
            "question": question
        })

        return response

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