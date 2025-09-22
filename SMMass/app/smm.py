from flask import Blueprint, render_template, request, flash, session, redirect, url_for, current_app, jsonify
from app.models import User
from app import db
from generators.text_gen import PostGenerator
from generators.image_gen import ImageGenerator
from social_publishers.vk_publisher import VKPublisher
from social_publishers.telegram_publisher import TelegramPublisher  # Новый импорт
from social_stats.vk_stats import VKStats
from social_stats.telegram_stats import TelegramStats  # Новый импорт
from config import openai_key, huggingface_key
from openai import OpenAI
import os
import certifi
import logging
import requests

# Инициализируем логгер
logger = logging.getLogger(__name__)

# Устанавливаем переменные окружения для SSL
os.environ['SSL_CERT_FILE'] = certifi.where()
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()

# Инициализация клиента OpenAI с прокси
client = OpenAI(
    api_key=openai_key,
    base_url="https://api.proxyapi.ru/openai/v1",
)

# Инициализация генератора изображений с Hugging Face API
image_generator = ImageGenerator(huggingface_key)

smm_bp = Blueprint('smm', __name__)


@smm_bp.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('auth.login'))
    return render_template('dashboard.html')


@smm_bp.route('/settings', methods=['GET', 'POST'])
def settings():
    if 'user_id' not in session:
        return redirect(url_for('auth.login'))

    user = User.query.get(session['user_id'])

    if request.method == 'POST':
        user.vk_api_id = request.form['vk_api_id']
        user.vk_group_id = request.form['vk_group_id']
        user.telegram_bot_token = request.form.get('telegram_bot_token', '')
        user.telegram_channel_id = request.form.get('telegram_channel_id', '')
        db.session.commit()
        flash('Settings saved!', 'success')

    return render_template('settings.html', user=user)


@smm_bp.route('/post-generator', methods=['GET', 'POST'])
def post_generator():
    if 'user_id' not in session:
        return redirect(url_for('auth.login'))

    if request.method == 'POST':
        tone = request.form['tone']
        topic = request.form['topic']
        generate_image = 'generate_image' in request.form
        auto_post_vk = 'auto_post_vk' in request.form
        auto_post_telegram = 'auto_post_telegram' in request.form  # Новый флаг
        custom_prompt = request.form.get('custom_prompt', '')

        user = User.query.get(session['user_id'])

        try:
            # Используем глобальный клиент OpenAI с прокси
            post_gen = PostGenerator(client, tone, topic)
            post_content = post_gen.generate_post()

            image_url = None

            if generate_image:
                try:
                    if custom_prompt:
                        image_prompt = custom_prompt
                    else:
                        image_prompt = post_gen.generate_post_image_description()

                    logger.info(f"Image generation prompt: {image_prompt}")

                    if not huggingface_key or huggingface_key.strip() == "":
                        flash('Error: Hugging Face API key is not configured. Please contact administrator.', 'error')
                        image_url = None
                    else:
                        image_url = image_generator.generate_image(image_prompt)
                        flash('Image generated successfully!', 'success')

                except Exception as e:
                    error_msg = str(e)
                    logger.error(f"Image generation error: {error_msg}")

                    if "payment method" in error_msg.lower():
                        flash(
                            'Error: Hugging Face requires a payment method for image generation. Please add a payment method to your Hugging Face account settings.',
                            'error')
                    elif "quota" in error_msg.lower() or "limit" in error_msg.lower():
                        flash('Error: API quota exceeded. Please try again later or upgrade your plan.', 'error')
                    elif "temporary" in error_msg.lower() or "unavailability" in error_msg.lower():
                        flash(
                            'Error: Image generation service is temporarily unavailable. Please try again in a few minutes.',
                            'error')
                    elif "API key" in error_msg or "authentication" in error_msg.lower():
                        flash('Error: Invalid Hugging Face API key. Please check your configuration.', 'error')
                    else:
                        flash(f'Error generating image: {error_msg}', 'error')
                    image_url = None

            # Публикация в VK
            if auto_post_vk and user.vk_api_id and user.vk_group_id:
                try:
                    vk_publisher = VKPublisher(user.vk_api_id, user.vk_group_id)
                    vk_publisher.publish_post(post_content, image_url)
                    flash('Post published to VK successfully!', 'success')
                except Exception as e:
                    flash(f'Error publishing to VK: {str(e)}', 'error')
            elif auto_post_vk:
                flash('Для автоматической публикации в VK необходимо настроить VK API в настройках', 'error')

            # Публикация в Telegram
            # Публикация в Telegram
            if auto_post_telegram and user.telegram_bot_token and user.telegram_channel_id:
                try:
                    telegram_publisher = TelegramPublisher(user.telegram_bot_token, user.telegram_channel_id)

                    # Получаем последнее сгенерированное изображение из папки
                    image_path = None
                    if generate_image:
                        image_dir = r"C:\Users\o.muravickiy\PycharmProjects\PromtEngineer\SMMass\app\static\generated"
                        if os.path.exists(image_dir):
                            # Ищем все PNG файлы и берем самый новый
                            png_files = [f for f in os.listdir(image_dir) if f.endswith('.png')]
                            if png_files:
                                # Сортируем по времени создания (последний первый)
                                png_files.sort(key=lambda x: os.path.getctime(os.path.join(image_dir, x)), reverse=True)
                                latest_image = png_files[0]
                                image_path = os.path.join(image_dir, latest_image)
                                flash(f'Using latest generated image: {latest_image}', 'success')
                            else:
                                flash('No PNG images found in generated folder', 'error')
                        else:
                            flash('Generated images folder not found', 'error')

                    telegram_publisher.publish_post(post_content, image_path)
                    flash('Post published to Telegram successfully!', 'success')
                except Exception as e:
                    flash(f'Error publishing to Telegram: {str(e)}', 'error')
            elif auto_post_telegram:
                flash(
                    'Для автоматической публикации в Telegram необходимо настроить Telegram бота и канал в настройках',
                    'error')
            return render_template('post_generator.html', post_content=post_content, image_url=image_url)

        except Exception as e:
            flash(f'Error generating content: {str(e)}', 'error')
            return render_template('post_generator.html')

    return render_template('post_generator.html')


@smm_bp.route('/vk-stats', methods=['GET'])
def vk_stats():
    if 'user_id' not in session:
        return redirect(url_for('auth.login'))

    user = User.query.get(session['user_id'])

    if not user.vk_api_id or not user.vk_group_id:
        flash('Настройте VK API в настройках для просмотра статистики', 'error')
        return redirect(url_for('smm.settings'))

    try:
        vk_stats = VKStats(user.vk_api_id, user.vk_group_id)
        followers_count = vk_stats.get_followers()

        stats = {
            "Подписчики": followers_count,
            "Лайки": "N/A",
            "Комментарии": "N/A",
            "Репосты": "N/A"
        }

        return render_template('vk_stats.html', stats=stats)
    except Exception as e:
        flash(f'Ошибка получения статистики VK: {str(e)}', 'error')
        return render_template('vk_stats.html', stats={})


@smm_bp.route('/telegram-stats', methods=['GET'])
def telegram_stats():
    if 'user_id' not in session:
        return redirect(url_for('auth.login'))

    user = User.query.get(session['user_id'])

    if not user.telegram_bot_token or not user.telegram_channel_id:
        flash('Настройте Telegram бота и канал в настройках для просмотра статистики', 'error')
        return redirect(url_for('smm.settings'))

    try:
        telegram_stats = TelegramStats(user.telegram_bot_token, user.telegram_channel_id)
        stats = telegram_stats.get_channel_stats()

        return render_template('telegram_stats.html', stats=stats)
    except Exception as e:
        flash(f'Ошибка получения статистики Telegram: {str(e)}', 'error')
        return render_template('telegram_stats.html', stats={})


@smm_bp.route('/check-models', methods=['GET'])
def check_models():
    """Эндпоинт для проверки доступности моделей Hugging Face"""
    results = {}

    models_to_check = [
        "runwayml/stable-diffusion-v1-5",
        "stabilityai/stable-diffusion-2-1",
        "CompVis/stable-diffusion-v1-4"
    ]

    headers = {"Authorization": f"Bearer {huggingface_key}"}

    for model_name in models_to_check:
        api_url = f"https://api-inference.huggingface.co/models/{model_name}"
        try:
            response = requests.get(api_url, headers=headers, timeout=10)
            results[model_name] = {
                "status": response.status_code,
                "available": response.status_code == 200
            }
        except Exception as e:
            results[model_name] = {
                "status": "error",
                "error": str(e)
            }

    return jsonify(results)