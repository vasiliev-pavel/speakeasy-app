async def translate_message(self, message: Message) -> None:
    await self.bot.send_chat_action(message.chat.id, ChatActions.TYPING)
    if not self.last_message_bot:
        await self.bot.send_message(message.chat.id, "Sorry, there is no message to translate.")
        return
    body = {
        "targetLanguageCode": 'ru',
        "sourceLanguageCode": 'en',
        "texts": [self.last_message_bot],
        "folderId": os.getenv('FOLDER_ID'),
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.getenv('IAM_TOKEN')}"
    }

    async with aiohttp.ClientSession(headers=headers) as session:
        await self.bot.send_chat_action(message.chat.id, ChatActions.TYPING)

        try:
            async with session.post('https://translate.api.cloud.yandex.net/translate/v2/translate',
                                    json=body) as response:
                response_dict = json.loads(await response.text())
                await self.bot.send_message(message.chat.id, f"Translated message: {response_dict['translations'][0]['text']}")
        except Exception as e:
            logging.error(f"Failed to translate message: {e}")
            await self.bot.send_message(message.chat.id, "Sorry, I couldn't translate the message. Please try again later.")
