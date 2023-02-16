import os
import logging
import aiogram
import asyncio
import openai
import requests
import aiohttp
from aiogram import types
from dotenv import load_dotenv
from aiogram.utils import executor

load_dotenv()


class OpenAIService:
    """
    A class that encapsulates the OpenAI API and provides a method to generate
    responses to prompts.
    """

    def __init__(self, api_key):
        self.api_key = api_key
        openai.api_key = self.api_key

    async def generate_response(self, prompt):
        response = openai.Completion.create(
            model="text-curie-001",
            prompt=prompt,
            temperature=0.5,
            max_tokens=1024,
            top_p=1.0,
            frequency_penalty=0.5,
            presence_penalty=0.0,
            stop=["You:", "\nFriend:"]
        )
        message = response.choices[0].text.strip()
        return message


class ElevenLabsService:
    """
    # A class that encapsulates the Eleven Labs text-to-speech API and provides a
    # method to convert text to speech.
    """

    def __init__(self, api_key):
        self.api_key = api_key

    async def text_to_speech_async(self, data):
        url = 'https://api.elevenlabs.io/v1/text-to-speech/TxGEqnHWrfWFTfGW9XjX'
        headers = {
            'accept': 'audio/mpeg',
            'xi-api-key': f'{self.api_key}',
            'Content-Type': 'application/json',
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=data) as response:
                if response.status == 200:
                    return await response.read()
                else:
                    raise Exception(f'Request failed {response.status}')


class ChatBot:

    # A class that implements the chatbot functionality using the aiogram library.
    def __init__(self, aiogram_key, openai_service, elevenlabs_service):
        self.aiogram_key = aiogram_key
        self.openai_service = openai_service
        self.elevenlabs_service = elevenlabs_service

        self.user_history = {}

        self.bot = aiogram.Bot(token=self.aiogram_key)
        self.dp = aiogram.Dispatcher(self.bot)

        self.dp.register_message_handler(self.welcome, commands=['start'])
        self.dp.register_message_handler(self.reply, content_types=['text'])

    def start(self):
        executor.start_polling(self.dp, skip_updates=True)

    async def welcome(self, message):
        await message.reply('Привет!')

    async def reply(self, message):
        chat_id = message.chat.id
        if chat_id not in self.user_history:
            self.user_history[chat_id] = ""

        self.user_history[chat_id] += f"\nYou: {message.text}"
        prompt = f"You: {self.user_history[chat_id]}\nFriend:"
        response = await self.openai_service.generate_response(prompt)
        self.user_history[chat_id] += f"\nFriend: {response}"
        print(self.user_history[chat_id])
        try:
            audio_data = await self.elevenlabs_service.text_to_speech_async({'text': response, "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.6
            }})
            await self.bot.send_voice(chat_id, audio_data)
        except Exception as e:
            await self.bot.send_message(chat_id, response)
            logging.error(f"Failed to process message: {e}")


if __name__ == '__main__':
    openai_service = OpenAIService(api_key=os.getenv('OPENAI_API'))
    elevenlabs_service = ElevenLabsService(api_key=os.getenv('ELEVENLABS_API'))
    chat_bot = ChatBot(aiogram_key=os.getenv(
        'AIOGRAM_API'), openai_service=openai_service, elevenlabs_service=elevenlabs_service)
    chat_bot.start()
