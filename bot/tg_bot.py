import os.path
import shutil

import telebot
from credentials import bot_token
import logging
import validators

import requests
from PIL import Image

from app.handlers.violatation_dcv import ViolatationDCV

project_predicts = "/Users/Daria/projects/PycharmProjects/violationDCV/project_predicts"

token = bot_token
bot = telebot.TeleBot(token)

logger = telebot.logger
telebot.logger.setLevel(logging.INFO)  # Outputs debug messages to console.

vdcv = ViolatationDCV()


@bot.message_handler(commands=['start', 'help'])
def send_welcome(message):
    bot.reply_to(message, "Привет! Отправь мне ссылку на картинку")


@bot.message_handler(content_types='text')
def message_reply(message):
    if validators.url(message.text):

        url = message.text

        try:
            data = requests.get(url).content
            img = str(message.message_id)+'.jpg'
            with open(img, 'wb') as file:
                file.write(data)

            try:
                result = vdcv.detect(img, project_predicts)
                for res in result:
                    label = res.get('label')
                    if label=='okay':
                        bot.send_message(message.chat.id, 'Рабочий экипирован')
                    elif label=='bad':
                        bot.send_message(message.chat.id, 'Рабочий не экипирован')
                    elif label=='unknown':
                        bot.send_message(message.chat.id, 'Сложно распознать экипирован или нет')
                    else:
                        bot.send_message(message.chat.id, label)


            except Exception as e:
                bot.send_message(message.chat.id, str(e))

            os.remove(img)
        except Exception as e:
            bot.send_message(message.chat.id, str(e))
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
