import os
import gdown
import zipfile
import telebot
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import torch.nn.functional as F
import PyPDF2
import requests

oauth_token = os.environ.get("OAUTH_TOKEN")
catalog_id = os.environ.get("CATALOG_ID")
tg_token = os.environ.get("BOT_TOKEN")

MODEL_DIR = "/model"
MODEL_WORK_DIR = "/model/ModernBERT"
MODEL_ZIP = os.path.join(MODEL_DIR, "model_weights.zip")

def pdf_to_text(pdf_path):
    # Open the PDF file in read-binary mode
    with open(pdf_path, 'rb') as pdf_file:
        # Create a PdfReader object instead of PdfFileReader
        pdf_reader = PyPDF2.PdfReader(pdf_file)

        # Initialize an empty string to store the text
        text = ''

        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text()

    return text

def generate_resume_recommendation(probabilities):
    """
    Генерирует расширенную рекомендацию по улучшению резюме на основе массива вероятностей классов.

    Args:
        probabilities (list): Список вероятностей классов (7 штук).

    Returns:
        str: Строка с рекомендациями по улучшению резюме.
    """

    class_names = [
        "Класс 0", "Класс 1", "Класс 2", "Класс 3", "Класс 4", "Класс 5", "Класс 6"
    ]

    list1_recommendations = {
        0: "в определенной степени резюме составлено сравнительно корректно с точки зрения работодателя (но уверенность сервиса в этом составляет всего лишь ...%)",
        1: "возможно стоит ещё чуть подробнее перечислить основные технологии и инструменты, с которыми умеет работать автор резюме (уверенность сервиса в этом составляет ...%)",
        2: "можно дополнить раздел с информацией о предыдущих местах работы (уверенность сервиса в этом составляет ...%)",
        3: "можно чуть полнее расписать технические навыки, опыт и достижения (уверенность сервиса в этом составляет ...%)",
        4: "в резюме имеется информация об образовании, в том числе дополнительном (онлайн курсы, повышение квалификации, сертификаты и т.д.), но можно еще немного полнее раскрыть данный пункт резюме (уверенность сервиса в этом составляет ...%)",
        5: "резюме стоит сделать чуть более кратким и лаконичным (уверенность сервиса в этом составляет ...%)",
        6: "в резюме стоит упомянуть, что соискатель готов к постоянному обучению и развитию (саморазвитию), готов работать в команде, обладает аналитическими навыками и так далее, т.е. добавить чуть больше информации о софт скиллах (уверенность сервиса в этом составляет ...%)"
    }

    list2_recommendations = {
        0: "В целом представленное резюме составлено достаточно правильно и корректно с точки зрения работодателя (уверенность сервиса в этом составляет ...%), но есть несколько замечаний (в порядке убывания важности):",
        1: "На наш взгляд в представленном резюме недостаточно подробно перечислены основные технологии и инструменты, с которыми умеет работать соискатель (уверенность сервиса в этом составляет ...%) и кроме этого есть еще несколько замечаний (в порядке убывания важности):",
        2: "На наш взгляд в представленном резюме недостаточно информации о предыдущих местах работы (уверенность сервиса в этом составляет ...%), хорошо бы дополнить этот раздел резюме. Кроме этого есть еще несколько замечаний (в порядке убывания важности):",
        3: "На наш взгляд в представленном резюме недостаточно полно указаны технические навыки, опыт и достижения (уверенность сервиса в этом составляет ...%) и кроме этого есть еще несколько замечаний (в порядке убывания важности):",
        4: "На наш взгляд в представленном резюме можно еще немного полнее раскрыть пункт про образование, в том числе и про дополнительное (онлайн курсы, повышение квалификации, полученные сертификаты и так далее) (уверенность сервиса в этом составляет ...%). Кроме этого есть еще несколько замечаний (в порядке убывания важности):",
        5: "На наш взгляд представленное резюме стоит сделать чуть более кратким и лаконичным (уверенность сервиса в этом составляет ...%). Кроме этого есть еще несколько замечаний (в порядке убывания важности):",
        6: "На наш взгляд в представленном резюме нужно полнее раскрыть информацию о софт скиллах соискателя (уверенность сервиса в этом составляет ...%). Кроме этого есть еще несколько замечаний (в порядке убывания важности):"
    }

    # 1. Связываем вероятности с индексами классов и сортируем по вероятности
    indexed_probabilities = sorted(enumerate(probabilities), key=lambda x: x[1], reverse=True)

    # 2. Берем топ 4 класса
    top_4_classes = indexed_probabilities[:4]

    # 3. Определяем класс с наибольшей вероятностью
    highest_prob_class_index = top_4_classes[0][0]
    highest_probability = top_4_classes[0][1]

    recommendations_output = []

    # 4. Формируем вступительную фразу из СПИСКА 2 для класса с наибольшей вероятностью
    opening_recommendation = list2_recommendations[highest_prob_class_index].replace("...%", f"{highest_probability*100:.0f}%")
    recommendations_output.append(opening_recommendation)

    # 5. Формируем остальные рекомендации из СПИСКА 1 для остальных классов из топ-4
    general_recommendations = []
    for i, (class_index, probability) in enumerate(top_4_classes):
        if i == 0: # Пропускаем класс с наибольшей вероятностью, так как для него уже есть вступительная фраза
            continue
        recommendation_text = list1_recommendations[class_index].replace("...%", f"{probability*100:.0f}%")
        general_recommendations.append(f"{len(general_recommendations) + 1}. {recommendation_text}") # Нумеруем рекомендации

    recommendations_output.extend(general_recommendations)

    return "\n".join(recommendations_output)

def download_and_extract_model():
    """Скачивает и распаковывает веса модели, если их нет в Volume"""
    if not os.path.exists(MODEL_WORK_DIR):
        # Скачиваем архив с Google Диска
        url = 'https://drive.google.com/uc?export=download&confirm=no_antivirus&id=1FQ9-NzozILQroq2cDZzSV82pBTXrHUxL'
        gdown.download(url, MODEL_ZIP, quiet=False)
        
        # Распаковываем архив
        with zipfile.ZipFile(MODEL_ZIP, 'r') as zip_ref:
            zip_ref.extractall(MODEL_DIR)
        
        # Удаляем архив после распаковки (опционально)
        os.remove(MODEL_ZIP)
        print(f"Модель скачана и распакована в {MODEL_WORK_DIR}")
    else:
        print(f"Модель уже существует в {MODEL_DIR}")


bot = telebot.TeleBot(tg_token)

# Скачиваем и распаковываем веса при запуске
download_and_extract_model()
# Загружаем модель и токенизатор из сохраненных файлов
tokenizer = AutoTokenizer.from_pretrained(MODEL_WORK_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_WORK_DIR)

# Переводим модель в evaluation mode
model.eval()

@bot.message_handler(commands=["start"])
def send_welcome(message):
    bot.send_message(message.chat.id, "Привет! Отправь мне своё резюме в формате pdf, и я что-нибудь расскажу тебе о нём)")

@bot.message_handler(content_types=["document"])
def document(message):
    # Загружаем резюме
    file_name = message.document.file_name
    file_info = bot.get_file(message.document.file_id)
    downloaded_file = bot.download_file(file_info.file_path)
    with open('/model/resume.pdf', 'wb') as new_file:
        new_file.write(downloaded_file)
    # Сохраняем резюме в файл
    resume_ru = pdf_to_text('/model/resume.pdf').replace('\n',' ')

    # Выполняем перевод резюме на английский
    IAM_TOKEN = oauth_token
    folder_id = catalog_id
    target_language = 'en'
    texts = [resume_ru]

    body = {
        "targetLanguageCode": target_language,
        "texts": texts,
        "folderId": folder_id,
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": "Api-Key {}".format(IAM_TOKEN)
    }

    response = requests.post('https://translate.api.cloud.yandex.net/translate/v2/translate',
        json = body,
        headers = headers
    )
    print(response.text)

    # Переведенное резюме
    resume = response.text

    # Токенизируем резюме
    encoded_input = tokenizer(resume, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        output = model(**encoded_input)

    # Получаем логиты (logits) - выходные значения модели до применения softmax
    logits = output.logits

    # Применяем softmax для получения вероятностей классов
    probabilities = F.softmax(logits, dim=1)

    # Выводим вероятности для каждого класса
    for i, prob in enumerate(probabilities[0]):
        print(f"Вероятность класса {i}: {prob.item()}")

    # Получаем индекс класса с максимальной вероятностью
    predicted_class = torch.argmax(probabilities, dim=1).item()
    print(f"Предсказанный класс: {predicted_class}")

    print(probabilities[0])

    recommendations = generate_resume_recommendation(np.array(probabilities[0]))
    print(recommendations)

    bot.send_message(message.chat.id, recommendations)

def main():

    print('waiting for message...')
    bot.polling(none_stop=True)

if __name__ == "__main__":
    main()
