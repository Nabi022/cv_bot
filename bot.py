import os
import numpy as np
from ultralytics import YOLO
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, CallbackContext
import cv2
from sort import Sort  # Для трекинга объектов

CONFIDENCE_THRESHOLD = 0.6  # Минимальная уверенность для фильтрации объектов
FRAME_STEP = 5  # Обрабатывать каждый 5-й кадр для повышения производительности
IOU_THRESHOLD = 0.5  # Порог IoU для удаления пересекающихся объектов

# Загрузка модели YOLO
model = YOLO("yolov8n.pt")  # Используем YOLOv8n (самая быстрая версия)

# Словарь переводов
TRANSLATIONS = {
    "person": "человек",
    "car": "машина",
    "dog": "собака",
    "cat": "кот",
    "bicycle": "велосипед",
    "truck": "грузовик",
    "bus": "автобус",
    "chair": "стул",
    "handbag": "сумка",
}

# Функция перевода
def translate_label(label: str) -> str:
    return TRANSLATIONS.get(label, label)  # Возвращаем перевод, если он есть

# Обработчик команды /start
async def start(update: Update, context: CallbackContext) -> None:
    await update.message.reply_text("Привет! Отправьте мне видео, и я найду все объекты в нем.")

# Обработчик видео
async def process_video(update: Update, context: CallbackContext) -> None:
    video = update.message.video
    if not video:
        await update.message.reply_text("Пожалуйста, отправьте видео.")
        return

    video_path = "temp_video.mp4"
    try:
        # Скачиваем видео
        file = await context.bot.get_file(video.file_id)
        await file.download_to_drive(video_path)

        # Анализируем видео
        detected_objects = analyze_video(video_path)

        if detected_objects:
            # Переводим названия объектов на русский
            object_list = ", ".join([f"{count} {translate_label(obj)}" for obj, count in detected_objects.items()])
            await update.message.reply_text(f"Обнаруженные объекты: {object_list}.")
        else:
            await update.message.reply_text("В видео не обнаружено объектов. Пожалуйста, попробуйте другое видео.")
    except Exception as e:
        await update.message.reply_text(f"Произошла ошибка: {str(e)}")
    finally:
        # Удаляем временный файл
        if os.path.exists(video_path):
            os.remove(video_path)

# Функция анализа видео
def analyze_video(video_path: str) -> dict:
    """
    Анализирует видео и возвращает словарь с уникальными объектами.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {}

    tracker = Sort()  # Инициализация трекера
    object_counter = {}
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % FRAME_STEP != 0:
            continue  # Обрабатываем только каждый 5-й кадр

        # Прогоняем кадр через YOLO
        results = model(frame, conf=CONFIDENCE_THRESHOLD, iou=IOU_THRESHOLD)
        detections = []

        # Извлекаем детекции из результатов YOLO
        for result in results:
            for obj in result.boxes:
                x1, y1, x2, y2 = obj.xyxy[0]  # Координаты объекта
                score = obj.conf[0]  # Уверенность модели
                class_id = int(obj.cls)  # ID класса
                label = result.names[class_id]  # Название объекта

                # Фильтруем по уверенности и минимальному размеру объекта
                if score > CONFIDENCE_THRESHOLD and (x2 - x1) * (y2 - y1) > 1000:
                    detections.append([x1, y1, x2, y2, score, class_id])

        if len(detections) == 0:
            continue  # Если нет детекций, переходим к следующему кадру

        # Передаем детекции в трекер
        tracked_objects = tracker.update(np.array(detections)[:, :5])  # Передаем только координаты и уверенность

        # Обрабатываем результаты трекера
        for track in tracked_objects:
            x1, y1, x2, y2, track_id = track
            for detection in detections:
                dx1, dy1, dx2, dy2, score, class_id = detection
                if abs(x1 - dx1) < 5 and abs(y1 - dy1) < 5:  # Проверяем совпадение координат
                    label = result.names[class_id]  # Получаем название объекта
                    if label not in object_counter:
                        object_counter[label] = set()
                    object_counter[label].add(int(track_id))
                    break  # Останавливаем поиск совпадений для текущего объекта

    cap.release()

    # Возвращаем количество уникальных объектов
    return {label: len(ids) for label, ids in object_counter.items()}

# Основная функция
def main() -> None:
   
    TOKEN = "7639621644:AAFKoDTtMEWFYm8Z1D2wDTTYIz1vhEaZGVw"

    # Создание приложения
    application = Application.builder().token(TOKEN).build()

    # Регистрация обработчиков
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.VIDEO, process_video))

    # Запуск бота
    application.run_polling()

if __name__ == "__main__":
    main()
