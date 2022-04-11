import cv2
import matplotlib.pyplot as plt

import pytesseract  # This is the TesseractOCR Python library

plate = cv2.CascadeClassifier('haarcascade_russian_plate_number.xml')
# pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files (x86)\\Tesseract-OCR\\tesseract.exe'
cap = cv2.VideoCapture('video/AM-STO-vorota_20220401-094050--20220401-094115.avi')

count = 0
path = "result/image"

while True:
    success, img = cap.read()
    # show_img = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2))
    # cv2.imshow("Videos", show_img)
    # cv2.waitKey(1)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


    # Извлекаем номер в файл
    def carplate_extract(image):
        carplate_rects = plate.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5)

        for x, y, w, h in carplate_rects:
            # Скорректировано для извлечения определенной интересующей области, например номерного знака автомобиля.
            carplate_img = image[y + 15:y + h - 10,
                           x + 15:x + w - 20]
        return carplate_img


    # На нужном изображении рисуем квадрат
    def carplate_detect(image):
        carplate_overlay = image.copy()
        carplate_rects = plate.detectMultiScale(carplate_overlay, scaleFactor=1.1, minNeighbors=3)
        for x, y, w, h in carplate_rects:
            cv2.rectangle(carplate_overlay, (x, y), (x + w, y + h), (255, 255, 255), 5)

        return carplate_overlay

    detected_carplate_img = carplate_detect(img)
    net_img = cv2.resize(detected_carplate_img, (img.shape[1] // 2, img.shape[0] // 2))

    show_img = cv2.resize(net_img, (net_img.shape[1] // 2, img.shape[0] // 2))
    cv2.imshow("Videos", show_img)
    cv2.waitKey(1)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


    def text_recognised(image):
        custom_config = '--oem 3 --psm 9'
        text = pytesseract.image_to_string(image, lang='rus+eng', config=custom_config)
        print(text)


    try:
        # Извлекаем номер
        img = carplate_extract(img)

        # Улучшаем качество изображения
        img = cv2.resize(img, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        # blur = cv2.GaussianBlur(img, (5, 5), 0)
        img = cv2.medianBlur(img, 3)
        # ret, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
        # rect_kern = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        # dilation = cv2.dilate(thresh, rect_kern, iterations=1)
        #
        # try:
        #     contours, hierarchy = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # except:
        #     ret_img, contours, hierarchy = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # sorted_contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])
        im2 = img.copy()
        # Запускаем распознование
        # text_recognised(img)

        cv2.imwrite(f"result/image{count}.jpg", img)
        # cv2.imshow('Result', im2)
        # cv2.waitKey(1)

    except:
        pass

# ВОЗМОЖНО ИСПОЛЬЗУЮ ПОЗЖЕ

# На нужном изображении рисуем квадрат
# def carplate_detect(image):
#     carplate_overlay = image.copy()
#     carplate_rects = plate.detectMultiScale(carplate_overlay, scaleFactor=1.1, minNeighbors=3)
#     for x, y, w, h in carplate_rects:
#         cv2.rectangle(carplate_overlay, (x, y), (x + w, y + h), (255, 255, 255), 5)
#
#     return carplate_overlay

# detected_carplate_img = carplate_detect(img)
# net_img = cv2.resize(detected_carplate_img, (img.shape[1] // 2, img.shape[0] // 2))


# Просто другое распознование
#         im2 = img.copy()
#         text = pytesseract.image_to_string(im2,
#                                         config='-c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ --psm 12 --oem 3')
#
#         print(text)

# def carplate_extract(image):
#     carplate_rects = plate.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5)
#
#     for x, y, w, h in carplate_rects:
#         # Скорректировано для извлечения определенной интересующей области, например номерного знака автомобиля.
#         carplate_img = image[y + 15:y + h - 10,
#                        x + 15:x + w - 20]
#     return carplate_img
