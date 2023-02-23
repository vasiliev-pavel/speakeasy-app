class ChatBot:
    def __init__(
        # Init
        self.last_message_bot=None  # initialize the last message as None
        self.last_message_user=None

        self.dp.register_message_handler(self.welcome, commands=['start'])
        self.dp.register_message_handler(
            self.chat_mode_menu, commands=['mode', 'language'])  # change to accept both commands
        self.dp.register_message_handler(
            self.translate_message(self.last_message_bot, "ru", "eng"), commands=['translate'])
        self.dp.register_message_handler(
            self.transcript_message, commands=['transcribe'])
        self.dp.register_message_handler(
            self.check_grammar, commands=['grammar'])
        self.dp.register_message_handler(self.reply, content_types=['text'])
        self.dp.register_callback_query_handler(
            self.handle_callback_query)
