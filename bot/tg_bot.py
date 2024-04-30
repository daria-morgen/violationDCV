import os.path

import telebot
from credentials import bot_token
import logging
import validators

from app.handlers.violatation_dcv import ViolatationDCV

project_predicts = "/Users/Daria/projects/PycharmProjects/violationDCV/bot/project_predicts"

token = bot_token
bot = telebot.TeleBot(token)

logger = telebot.logger
telebot.logger.setLevel(logging.INFO)  # Outputs debug messages to console.

vdcv = ViolatationDCV()


@bot.message_handler(commands=['start', 'help'])
def send_welcome(message):
    bot.reply_to(message, "Отправь мне фото или ссылку на фото/видео")


@bot.message_handler(content_types='text')
def message_reply(message):
    if validators.url(message.text):
        if ".jpg" in message.text:
            save_path = os.path.join(project_predicts, str(message.chat.id))
            try:
                result = vdcv.detect(message.text[0:message.text.index(".jpg") + len(".jpg")], save_path)
                for res in result:
                    bot.send_message(message.chat.id, res.get('label'))
            except:
                bot.send_message(message.chat.id, 'Во время работы со ссылкой произошла ошибка')
        else:
            bot.send_message(message.chat.id, 'Ваша ссылка не является картинкой формата jpeg или пр:')
    else:
        bot.send_message(message.chat.id, 'Ваше сообщение не похоже на ссылку на картинку или видео')

# @bot.message_handler(content_types='photo')
# def get_broadcast_picture(message):
#     logger.info('photo update')
#
#     file_path = bot.get_file(message.photo[0].file_id).file_path
#     file = bot.download_file(file_path)
#
#     with open("python1.png", "wb") as code:
#         code.write(file)


bot.infinity_polling()
