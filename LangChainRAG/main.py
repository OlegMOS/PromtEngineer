# -*- coding: utf-8 -*-
"""
Виртуальный консультант для ювелирного магазина с использованием RAG и LangChain
"""
import os
import sys
import json
import chromadb
import numpy as np
from typing import List, Dict, Any
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline

# Обновленные импорты для совместимости с новыми версиями LangChain
try:
    from langchain_community.llms import HuggingFacePipeline
    from langchain_community.embeddings import HuggingFaceEmbeddings
except ImportError:
    # Если langchain_community не установлен, попробуем старые импорты
    from langchain.llms import HuggingFacePipeline
    from langchain.embeddings import HuggingFaceEmbeddings

from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.chains import LLMChain
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import faiss
from sentence_transformers import SentenceTransformer

# --- 1. ПОДГОТОВКА ДОКУМЕНТОВ О ЮВЕЛИРНЫХ ИЗДЕЛИЯХ ---
# Предполагаем, что у нас есть данные о ювелирных изделиях
jewelry_documents = [
    {
        "content": "Серебряные украшения требуют особого ухода. Храните их в сухом месте, избегайте контакта с водой, химикатами и косметикой. Для очистки используйте мягкую ткань и специальные средства для серебра.",
        "metadata": {"type": "уход", "материал": "серебро"}
    },
    {
        "content": "Золотые украшения следует хранить отдельно от других изделий. Для очистки используйте теплый мыльный раствор и мягкую щетку. Избегайте контакта с хлором и другими агрессивными химикатами.",
        "metadata": {"type": "уход", "материал": "золото"}
    },
    {
        "content": "Кольцо с бриллиантом 'Нежность': изготовлено из белого золота 585 пробы, содержит бриллиант 0.5 карата. Идеально для помолвки. Цена: 45,000 рублей.",
        "metadata": {"type": "товар", "категория": "кольцо", "материал": "белое золото", "камень": "бриллиант"}
    },
    {
        "content": "Серебряные серьги 'Лунный свет': изящный дизайн с фианитами. Подходят для повседневной носки и особых occasions. Цена: 5,500 рублей.",
        "metadata": {"type": "товар", "категория": "серьги", "материал": "серебро", "камень": "фианит"}
    },
    {
        "content": "Жемчужное ожерелье 'Жемчужина': культивированный жемчуг высшего качества, длина 45 см. Элегантное украшение для вечерних мероприятий. Цена: 22,000 рублей.",
        "metadata": {"type": "товар", "категория": "ожерелье", "материал": "жемчуг"}
    }
]


# --- 2. НАСТРОЙКА БАЗ ДАННЫХ (FAISS и ChromaDB) ---
class VectorDatabase:
    def __init__(self, database_type="chroma"):
        self.database_type = database_type
        self.encoder = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

        if database_type == "faiss":
            self.setup_faiss()
        else:
            self.setup_chromadb()

    def setup_faiss(self):
        """Настройка FAISS базы данных"""
        dimension = 384  # Размерность векторов для выбранного энкодера
        self.faiss_index = faiss.IndexFlatL2(dimension)
        self.faiss_documents = []

    def setup_chromadb(self):
        """Настройка ChromaDB базы данных"""
        # Установка совместимости с SQLite для ChromaDB
        try:
            __import__('pysqlite3')
            sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
        except ImportError:
            print("pysqlite3 не установлен, используем стандартный sqlite3")

        self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
        self.collection = self.chroma_client.get_or_create_collection(name="jewelry_collection")

    def add_documents(self, documents: List[Dict]):
        """Добавление документов в выбранную базу данных"""
        texts = [doc["content"] for doc in documents]
        metadatas = [doc["metadata"] for doc in documents]
        ids = [f"id_{i}" for i in range(len(documents))]

        if self.database_type == "faiss":
            # Для FAISS
            embeddings = self.encoder.encode(texts)
            self.faiss_index.add(np.array(embeddings).astype('float32'))
            self.faiss_documents.extend(documents)
        else:
            # Для ChromaDB
            self.collection.add(
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )

    def query(self, query_text: str, k: int = 3):
        """Поиск релевантных документов"""
        if self.database_type == "faiss":
            query_embedding = self.encoder.encode([query_text])
            distances, indices = self.faiss_index.search(np.array(query_embedding).astype('float32'), k)

            results = []
            for i in indices[0]:
                if i < len(self.faiss_documents):
                    results.append(self.faiss_documents[i])
            return results
        else:
            # Для ChromaDB
            results = self.collection.query(
                query_texts=[query_text],
                n_results=k
            )

            documents = []
            for i in range(len(results['documents'][0])):
                doc_content = results['documents'][0][i]
                doc_metadata = results['metadatas'][0][i]
                documents.append({
                    "content": doc_content,
                    "metadata": doc_metadata
                })
            return documents


# --- 3. НАСТРОЙКА МОДЕЛИ SAIGA LLAMA3 ---
def setup_llm_model():
    """Настройка модели IlyaGusev/saiga_llama3_8b"""
    model_name = "IlyaGusev/saiga_llama3_8b"

    print(f"Загрузка модели: {model_name}")

    try:
        # Загрузка токенизатора
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=False,
            trust_remote_code=True
        )

        # Конфигурация для квантования (экономия памяти)
        try:
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )
            quantization_config = bnb_config
        except ImportError:
            print("bitsandbytes не доступен, загружаем без квантования")
            quantization_config = None

        # Загрузка модели
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16 if not quantization_config else None
        )

        # Saiga модели используют специальный формат промптов
        # Настройка пайплайна для генерации
        text_generation_pipeline = pipeline(
            model=model,
            tokenizer=tokenizer,
            task="text-generation",
            temperature=0.2,
            do_sample=True,
            repetition_penalty=1.1,
            return_full_text=False,
            max_new_tokens=500,
        )

        llm = HuggingFacePipeline(pipeline=text_generation_pipeline)
        return llm

    except Exception as e:
        print(f"Ошибка при загрузке модели: {e}")
        print("Проверьте:")
        print("1. Выполнили ли 'huggingface-cli login'")
        print("2. Есть ли доступ к модели на https://huggingface.co/IlyaGusev/saiga_llama3_8b")
        raise

# --- 4. СОЗДАНИЕ RAG-ЦЕПОЧКИ ---
def create_rag_chain(llm, retriever):
    """Создание RAG-цепочки для обработки запросов"""

    # Шаблон промпта для Saiga модели
    prompt_template = """<|im_start|>system
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
    {products_context}<|im_end|>
    <|im_start|>user
    {question}<|im_end|>
    <|im_start|>assistant
    """

    prompt = PromptTemplate(
        input_variables=["care_context", "products_context", "question"],
        template=prompt_template,
    )

    # Функция для разделения документов на контекст ухода и товары
    def split_contexts(docs):
        care_docs = []
        product_docs = []

        for doc in docs:
            if doc['metadata'].get('type') == 'уход':
                care_docs.append(doc['content'])
            elif doc['metadata'].get('type') == 'товар':
                product_docs.append(doc['content'])

        care_context = "\n".join(care_docs) if care_docs else "Информация по уходу отсутствует."
        products_context = "\n".join(product_docs) if product_docs else "Информация о товарах отсутствует."

        return {
            "care_context": care_context,
            "products_context": products_context
        }

    # Создание цепочки
    rag_chain = (
            {
                "docs": lambda x: retriever(x["question"]),
                "question": lambda x: x["question"]
            }
            | {
                "care_context": lambda x: split_contexts(x["docs"])["care_context"],
                "products_context": lambda x: split_contexts(x["docs"])["products_context"],
                "question": lambda x: x["question"]
            }
            | prompt
            | llm
            | StrOutputParser()
    )

    return rag_chain


# --- 5. ТЕСТИРОВАНИЕ СИСТЕМЫ ---
def test_consultant(rag_chain):
    """Тестирование виртуального консультанта"""
    test_questions = [
        "Как ухаживать за серебряными украшениями?",
        "Посоветуйте кольцо с бриллиантом для помолвки",
        "Что у вас есть из жемчужных украшений?",
        "Как чистить золотые украшения?",
        "Расскажите о ваших серебряных серьгах"
    ]

    print("=== ТЕСТИРОВАНИЕ ВИРТУАЛЬНОГО КОНСУЛЬТАНТА ===\n")

    for i, question in enumerate(test_questions, 1):
        print(f"{i}. Вопрос: {question}")
        try:
            response = rag_chain.invoke({"question": question})
            print(f"Ответ: {response}\n")
        except Exception as e:
            print(f"Ошибка при обработке вопроса: {e}\n")
        print("-" * 80 + "\n")


# --- ОСНОВНАЯ ПРОГРАММА ---
def main():
    """Основная функция программы"""
    print("Инициализация виртуального консультанта для ювелирного магазина...")

    # 1. Инициализация базы данных (можно выбрать "faiss" или "chroma")
    print("Настройка базы данных...")
    db = VectorDatabase(database_type="chroma")  # Можно изменить на "faiss"

    # 2. Добавление документов в базу данных
    print("Добавление документов о ювелирных изделиях...")
    db.add_documents(jewelry_documents)

    # 3. Создание функции для поиска
    def retriever(query):
        return db.query(query, k=5)

    # 4. Настройка модели Llama3
    print("Загрузка модели Llama3...")
    try:
        llm = setup_llm_model()
    except Exception as e:
        print(f"Ошибка при загрузке модели: {e}")
        print("Проверьте доступ к модели и наличие необходимых зависимостей.")
        return

    # 5. Создание RAG-цепочки
    print("Создание RAG-цепочки...")
    rag_chain = create_rag_chain(llm, retriever)

    # 6. Тестирование консультанта
    print("Запуск тестирования...")
    test_consultant(rag_chain)

    # 7. Пример интерактивного режима
    print("=== РЕЖИМ ВЗАИМОДЕЙСТВИЯ ===")
    print("Введите ваш вопрос (или 'выход' для завершения):")

    while True:
        user_input = input("\nВаш вопрос: ").strip()

        if user_input.lower() in ['выход', 'exit', 'quit']:
            print("Завершение работы виртуального консультанта.")
            break

        if user_input:
            try:
                response = rag_chain.invoke({"question": user_input})
                print(f"\nКонсультант: {response}")
            except Exception as e:
                print(f"Произошла ошибка: {e}")


if __name__ == "__main__":
    main()