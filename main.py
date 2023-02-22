from aiogram.utils import executor
from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton
from aiogram import Bot, Dispatcher
from aiogram.types import Message
from aiogram.types import ChatActions
from googletrans import Translator
import logging
import os
import aiogram
import asyncio
import openai
import aiohttp
from aiogram import types
from dotenv import load_dotenv
import aiomysql
import re
load_dotenv()


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


class OpenAIService:
    """
    A class that encapsulates the OpenAI API and provides a method to generate
    responses to prompts.
    """

    def __init__(self, api_key):
        self.api_key = api_key
        openai.api_key = self.api_key

    async def generate_response(self, prompt, temp, top_p, fr_pen, pr_pen, stop):
        response = openai.Completion.create(
            model="text-davinci-003",
            prompt=prompt,
            temperature=temp,
            # Какая температура выборки будет использоваться, от 0 до 2.
            # Более высокие значения, например 0,8, сделают вывод более случайным, а более низкие, например 0,2,
            # сделают его более целенаправленным и детерминированным.
            # Обычно мы рекомендуем изменять это значение или top_p, но не оба.
            max_tokens=2048,

            top_p=top_p,
            # Альтернатива выборке с температурой, называемой выборкой ядра, когда модель рассматривает результаты токенов с массой вероятности top_p.
            # Таким образом, 0,1 означает, что рассматриваются только лексемы, составляющие верхние 10% вероятностной массы.

            # Разумные значения коэффициентов штрафа составляют от 0,1 до 1, если цель состоит в том, чтобы просто несколько уменьшить количество повторяющихся образцов.
            # Если целью является сильное подавление повторов, то можно увеличить коэффициенты до 2, но это может заметно ухудшить качество образцов.
            # Отрицательные значения можно использовать для увеличения вероятности повторения.
            frequency_penalty=fr_pen,
            # Число между -2,0 и 2,0. Положительные значения штрафуют новые лексемы на основе их существующей частоты в тексте,
            # уменьшая вероятность того, что модель повторит ту же строку дословно.
            presence_penalty=pr_pen,
            # frequency_penalty=0.7,
            # presence_penalty=0.6,
            # Число между -2,0 и 2,0. Положительные значения штрафуют новые лексемы на основе того, появляются ли они в тексте до сих пор,
            # увеличивая вероятность того, что модель будет говорить о новых темах.
            stop=stop
            # temperature=0.5,
            # max_tokens=60,
            # top_p=1.0,
            # frequency_penalty=0.5,
            # presence_penalty=0.0,
        )
        message = response.choices[0].text.strip()
        return message


class MessageHistory:
    def __init__(self, db_host, db_user, db_password, db_name, ):
        self.db_host = db_host
        self.db_user = db_user
        self.db_password = db_password
        self.db_name = db_name
        self.state_summary = False

    async def get_db_connection(self):
        if not hasattr(self, '_db_pool'):
            self._db_pool = await aiomysql.create_pool(
                host=self.db_host,
                user=self.db_user,
                password=self.db_password,
                db=self.db_name,
                cursorclass=aiomysql.cursors.DictCursor)
        return self._db_pool

    async def summarize_messages(self, chat_id, conn, cursor):
        await cursor.execute(
            "SELECT message FROM message_history WHERE chat_id = %s ORDER BY id ASC LIMIT 10", (chat_id,))
        messages = [row['message'] for row in await cursor.fetchall()]

        # Check if there is a synopsis in the summary table
        await cursor.execute("SELECT synopsis FROM summary WHERE chat_id = %s", (chat_id,))
        result = await cursor.fetchone()

        if result is not None and result['synopsis'] != '':
            messages = [result['synopsis']] + messages
        summary = (''.join(messages))

        message = await self.summarize_text(summary)

        await cursor.execute(
            "DELETE FROM message_history WHERE chat_id = %s ORDER BY id ASC LIMIT 10", (chat_id,))
        await cursor.execute(
            "INSERT INTO summary (chat_id, synopsis) VALUES (%s, %s) ON DUPLICATE KEY UPDATE synopsis = %s",
            (chat_id, message, message))
        await conn.commit()

    async def summarize_text(self, summary):
        openai.api_key = os.getenv('OPENAI_API')
        req = "Prompt: Emulate human memory by composing a brief sinopsis of dialog"
        # "Make an outline, highlighting the main points, and be brief."
        # "Make a summary, highlighting the main things, and keep it short"
        # Prompt: Emulate human memory by composing a brief sinopsis of dialog
        prompt = f"{req}\n{summary}"
        print(prompt)
        response = openai.Completion.create(
            model="text-davinci-003",
            prompt=prompt,
            temperature=0.7,
            max_tokens=2048,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
        )
        return response.choices[0].text.strip()

    async def save_message(self, chat_id, message):
        db = await self.get_db_connection()
        async with db.acquire() as conn:
            async with conn.cursor() as cursor:
                # Insert the message into the database
                await cursor.execute(
                    "INSERT INTO message_history (chat_id, message) VALUES (%s, %s)", (chat_id, message))
                # Check the number of messages from the current user
                await cursor.execute(
                    "SELECT COUNT(*) as message_count FROM message_history WHERE chat_id = %s", (chat_id,))
                result = await cursor.fetchone()
                message_count = result['message_count']

                # If the number of messages exceeds 12, summarize the messages and insert the result back into the database
                if message_count >= 12:
                    self.state_summary = True
                    await self.summarize_messages(chat_id, conn, cursor)

                await conn.commit()

    async def get_message_history(self, chat_id):
        db = await self.get_db_connection()
        async with db.acquire() as conn:
            async with conn.cursor() as cursor:
                await cursor.execute(
                    "SELECT message FROM message_history WHERE chat_id = %s", (chat_id,))
                message_history = [row['message'] for row in await cursor.fetchall()]
                if self.state_summary:
                    # print("tyt")
                    await cursor.execute(
                        "SELECT synopsis FROM summary WHERE chat_id = %s", (chat_id,))
                    result = await cursor.fetchone()
                    summary = result['synopsis']
                    message_history.append(summary)
                return message_history


class ChatBot:
    def __init__(
            self, aiogram_key: str, openai_service: OpenAIService,
            elevenlabs_service: ElevenLabsService, db_host: str,
            db_user: str, db_password: str, db_name: str) -> None:
        self.aiogram_key = aiogram_key
        self.openai_service = openai_service
        self.elevenlabs_service = elevenlabs_service
        self.message_history = MessageHistory(
            db_host, db_user, db_password, db_name)
        self.bot = Bot(token=self.aiogram_key)
        self.dp = Dispatcher(self.bot)
        self.translator = Translator()

        self.dp.register_message_handler(self.welcome, commands=['start'])
        self.dp.register_message_handler(
            self.chat_mode_menu, commands=['mode'])
        self.dp.register_message_handler(
            self.translate_message, commands=['translate'])
        self.dp.register_message_handler(
            self.transcript_message, commands=['transcribe'])
        self.dp.register_message_handler(
            self.check_grammar, commands=['grammar'])
        self.dp.register_message_handler(self.reply, content_types=['text'])
        self.dp.register_callback_query_handler(
            self.handle_callback_query)

        self.current_mode = 'text'  # initialize the user's current mode as text
        self.last_message_bot = None  # initialize the last message as None
        self.last_message_user = None

    def start(self):
        executor.start_polling(self.dp, skip_updates=True)

    async def check_grammar(self, message: Message) -> None:
        await self.bot.send_chat_action(message.chat.id, ChatActions.TYPING)
        if not self.last_message_user:
            await self.bot.send_message(message.chat.id, "Sorry, no message for the grammar check.")
        prompt = f"Fix grammar and typos: \"{self.last_message_user}\"\nAnd give specific advice on what rule to learn so as not to make this mistake again, and show example. In the form of:\nCorrect sentence:\nExplanation:\nExample rule:"
        try:
            response = await self.openai_service.generate_response(prompt, 1, 1, 0, 0, None)
            await self.bot.send_message(message.chat.id, response)
        except Exception as e:
            logging.error(f"Failed to check_grammar message: {e}")
            await self.bot.send_message(message.chat.id,
                                        "Sorry, I couldn't run the grammar check on the message. Please try again later.")

    async def translate_message(self, message: Message) -> None:
        await self.bot.send_chat_action(message.chat.id, ChatActions.TYPING)
        if not self.last_message_bot:
            await self.bot.send_message(message.chat.id, "Sorry, there is no message to translate.")
            return

        try:
            translation = self.translator.translate(
                self.last_message_bot, dest='ru')
            await self.bot.send_message(message.chat.id, f"Translated message: {translation.text}")
        except Exception as e:
            logging.error(f"Failed to translate message: {e}")
            await self.bot.send_message(message.chat.id, "Sorry, I couldn't translate the message. Please try again later.")

    async def transcript_message(self, message: Message) -> None:
        if self.current_mode != 'voice':
            await self.bot.send_message(message.chat.id, "Sorry, the transcribe command is only available in voice mode.")
        elif not self.last_message_bot:
            await self.bot.send_message(message.chat.id, "Sorry, there is no message to transcribe.")
        else:
            await self.bot.send_message(message.chat.id,  self.last_message_bot)

    async def chat_mode_menu(self, message) -> None:
        keyboard = InlineKeyboardMarkup(row_width=2)
        text_button = InlineKeyboardButton(
            'Text', callback_data='text')
        voice_button = InlineKeyboardButton(
            'Voice', callback_data='voice')
        keyboard.add(text_button, voice_button)
        await self.bot.send_message(
            message.chat.id, 'Please choose a chat mode:',
            reply_markup=keyboard)

    async def welcome(self, message) -> None:
        await self.bot.send_message(message.chat.id, 'Welcome!')

    async def reply(self, message: Message) -> None:
        chat_id: int = message.chat.id
        self.last_message_user = message.text  # save the user's last message

        await self.message_history.save_message(chat_id, f"\nNick:{message.text}")
        message_history = await self.message_history.get_message_history(chat_id)
        summary: str = ''.join(message_history)
        # custom_settings: str = "The friend is responsive, creative, smart, inquisitive and very sociable"
        custom_settings: str = "The friend works as an actor, has been in many movies, dumb, creative, recently won an award for best actor, sarcastic."
        prompt: str = f"{custom_settings}\n{summary}\nFriend:"

        await self.bot.send_chat_action(chat_id, ChatActions.TYPING if self.current_mode == 'text' else ChatActions.RECORD_AUDIO)

        try:
            # (prompt,temperature,top_p,frequency_penalty,presence_penalty,stop)
            response: str = await self.openai_service.generate_response(prompt, 0.5, 1, 0.7, 0.6, ["Nick:"])
        except Exception as e:
            await self.bot.send_message(chat_id, "Sorry, I couldn't process your message. Please try again later.")
            logging.error(f"Failed to process message: {e}")
            return

        if response:

            self.last_message_bot = response  # save the bot's last response

            await self.message_history.save_message(chat_id, f"\nFriend:{response}")

            if self.current_mode == 'text':
                await self.bot.send_message(chat_id, response)
            else:
                try:
                    audio_data = await self.elevenlabs_service.text_to_speech_async({'text': response, "voice_settings": {
                        "stability": 0.8,
                        "similarity_boost": 0.3
                    }})

                    await self.bot.send_voice(chat_id, audio_data)
                except Exception as e:
                    await self.bot.send_message(chat_id, response)
                    logging.error(f"Failed to process message: {e}")

    async def handle_callback_query(self, callback_query) -> None:
        await callback_query.answer()
        self.current_mode = 'text' if callback_query.data == 'text' else 'voice'
        await self.bot.delete_message(chat_id=callback_query.message.chat.id, message_id=callback_query.message.message_id)
        await self.bot.send_message(callback_query.message.chat.id, f'You have selected {self.current_mode} mode')


if __name__ == '__main__':
    aiogram_key = os.getenv('AIOGRAM_API', '0000')
    openai_service = OpenAIService(api_key=os.getenv('OPENAI_API'))
    elevenlabs_service = ElevenLabsService(api_key=os.getenv('ELEVENLABS_API'))
    db_host = "localhost"
    db_user = "root"
    db_password = "Spleenter1q1"
    db_name = "chat_db"

    chat_bot = ChatBot(aiogram_key=aiogram_key, openai_service=openai_service,
                       elevenlabs_service=elevenlabs_service, db_host=db_host, db_user=db_user, db_password=db_password, db_name=db_name)
    chat_bot.start()
