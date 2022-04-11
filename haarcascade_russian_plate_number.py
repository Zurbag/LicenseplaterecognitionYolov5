import cv2
import numpy

# Получаем изображение
img = cv2.imread('images/2.jpeg')
# Преобразуем к серому
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Получаем в программу faces.xml
# CascadeClassifier() - берет файл и вытягивает его как наренированную модель
plate = cv2.CascadeClassifier('haarcascade_russian_plate_number.xml')

# Получаем координаты найденых номеров
# detectMultiScale() - мы получаем координаты всех найденых объектов
# img - Изображение на которм ищем
# caleFactor=2 - жтот метод позволяет находить те
# лица которые были в два раза больше чем в натренированной модели
#               это нужно для того что можель
#               могли тренировать на маленьких картинках
#               могли тренировать на маленьких картинках
# minNeighbors = Как много может быть сосдних лиц
results = plate.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=4)

# Все найденные лица выделяем в квадраты
# Цикорм переьераем координаты ширину и высоту

for (x, y, w, h) in results:
    # rectangle    
    # img Квадраты рисуем на цветных изображениях    
    # (x, y) - начальные координаты    
    # (x + w, y + h) - конечные координаты x+ найденная ширина y + найденная высота    
    # (0, 0, 255) - цвет - тут красный    
    # thickness=1 Рамка    
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), thickness=2)


cv2.imshow("Face - 1", img)
cv2.waitKey(0)