import os
import logging
import aiogram
import asyncio
import openai
from aiogram import executor
import requests
import aiohttp
from aiogram import types
from dotenv import load_dotenv


load_dotenv()

# Установка ключа API для OpenAI
openai.api_key = os.getenv('OPENAI_API')

elevenlabs_key = os.getenv('ELEVENLABS_API')

# Logger
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

# History Init
user_history = {}

# Bot and Dispatcher Init
aiogram_key = os.getenv('AIOGRAM_API')
bot = aiogram.Bot(token=aiogram_key)
dp = aiogram.Dispatcher(bot)


# Message Handler
@dp.message_handler(commands=['start'])
async def start(message: aiogram.types.Message):
    """Обрабатывает команду /start."""
    await message.reply('Привет! Я не бот, который может поддерживать диалог с помощью искусственного интеллекта.')


@dp.message_handler(commands=['help'])
async def help(message: aiogram.types.Message):
    """Обрабатывает команду /help."""
    await message.reply('Просто начни cо мной говорить и я постараюсь дать тебе ответ!')


async def generate_response(chat_id):
    """Генерирует ответ на основе входных данных пользователя."""
    global user_history
    prompt = f"You: {user_history[chat_id]}\nFriend:"
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        temperature=0.5,
        max_tokens=1024,
        top_p=1.0,
        frequency_penalty=0.5,
        presence_penalty=0.0,
        stop=["You:", "\nFriend:"]
    )
    message = response.choices[0].text.strip()
    user_history[chat_id] += f"\nFriend: {message}"
    return message


async def text_to_speech_async(data):
    url = 'https://api.elevenlabs.io/v1/text-to-speech/TxGEqnHWrfWFTfGW9XjX'
    headers = {
        'accept': 'audio/mpeg',
        'xi-api-key': f'${elevenlabs_key}',
        'Content-Type': 'application/json',
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, json=data) as response:
            if response.status == 200:
                return await response.read()
            else:
                raise Exception(
                    f'Request failed with status code {response.status}')


@dp.message_handler(content_types=['text'])
async def reply(message: aiogram.types.Message):
    """Отвечает на сообщение пользователя."""
    global user_history
    chat_id = message.chat.id
    print(chat_id)
    if chat_id not in user_history:
        user_history[chat_id] = ""
    user_history[chat_id] += f"\nYou: {message.text}"
    response = await generate_response(chat_id)
    # audio_data = await text_to_speech_async({'text': response,  "stability": 0.40,"similarity_boost": 0.60})
    # with open('output.mp3', 'wb') as f:
    #     f.write(audio_data)
    # audio = open("output.mp3", "rb")
    # audio_message = types.InputFile(audio, filename='output.mp3')
    # await bot.send_voice(chat_id, audio_message)
    try:
        audio_data = await text_to_speech_async({'text': response, "stability": 0.40, "similarity_boost": 0.60})
        await bot.send_voice(chat_id, audio_data)
    except Exception as e:
        print(f"Error sending voice message: {e}")
        await bot.send_message(chat_id, response)


# Loop
if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    executor.start_polling(dp, loop=loop, skip_updates=True)
