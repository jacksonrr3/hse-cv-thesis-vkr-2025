import os
import re
from pathlib import Path

from collections import defaultdict

import cv2
import easyocr
import torch
import torch.nn as nn
from ultralytics import YOLO
from PIL import Image
import numpy as np
from torchvision import datasets, models, transforms



PATH_TO_PLATE_DETECTOR_MODEL = 'models/license_plate_detector.pt'
PATH_TO_CAR_CLASSIFICATION_MODEL_WEIGHTS = 'models/car_classifier.pth'

CONF_TRESHOLD = 80

car_models = ['volvo xc70', 'vw teramont'] 

class VehicleProcessor:
    def __init__(self, device='cuda:0' if torch.cuda.is_available() else 'cpu'):
        # Initialize models
        self._device = device
        self._vehicles_data = defaultdict(dict)

        # YOLOv8 custom model for plate detection
        self._plate_detector = YOLO(PATH_TO_PLATE_DETECTOR_MODEL).to(
            device)  # Your custom trained model

        # OCR for plate recognition
        # self._plate_reader = easyocr.Reader(
            # ['en', 'ru'], gpu=torch.cuda.is_available())
        self._ocr = easyocr.Reader(['ru', 'en'], gpu=torch.cuda.is_available())  # Только русский язык
        
        # Параметры предобработки
        self._plate_size = (520, 112)  # Стандартный размер номера для предобработки
        self._char_whitelist = 'ABEKMHOPCTYXАВЕКМНОРСТУХ0123456789'  # Разрешенные символы

        # Vehicle make/model classifier
        self._classifier = self._init_classifier()

        # Image transforms
        self._transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def _init_classifier(self):
        """Initialize vehicle classifier"""
        # model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
        # model.fc = torch.nn.Linear(512, len(self.make_model_labels))  # Modify last layer
        # model.load_state_dict(torch.load('vehicle_classifier.pt'))  # Your trained weights
        # return model.eval().to(self.device)
        model = models.resnet50(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, len(car_models))
        )

        model.load_state_dict(torch.load(PATH_TO_CAR_CLASSIFICATION_MODEL_WEIGHTS, map_location=self._device))
        model.eval()
        model.to(self._device)

        return model
        

    def process_frame(self, frame, detections):
        """Обработка одного кадра с детекциями"""
        results = []
        
        # Проверяем что есть детекции и треки
        if not hasattr(detections[0], 'boxes') or detections[0].boxes.id is None:
            return results
            
        for box, track_id in zip(detections[0].boxes, detections[0].boxes.id.int().tolist()):
            # Работаем только с автомобилями (class 2)
            if box.cls != 2:
                continue
                
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            vehicle_roi = frame[y1:y2, x1:x2]
            
            if vehicle_roi.size == 0:
                continue
                
            # Получаем или создаем запись о машине
            vehicle = self._vehicles_data.get(track_id, {})
                       
            # # Классификация (если еще не делали)
            if 'class' not in vehicle:
                car_make, ok = self._classify_vehicle(vehicle_roi)
                if ok:
                    vehicle['class'] = car_make    
                
            # # Чтение номера (если еще не делали)
            if 'plate' not in vehicle and 'class' in vehicle:
                vehicle['plates'] = self._detect_plates(vehicle_roi)
                for plate in vehicle['plates']:
                    plate_text = self._recognize_plate(plate['roi'])
                    if plate_text:
                        vehicle['plate'] = plate_text   
            
            # Сохраняем обновленные данные
            self._vehicles_data[track_id] = vehicle

            # Формируем результат для этого авто

            results.append({
                'track_id': track_id,
                'bbox': [x1, y1, x2, y2],
                'class': vehicle.get('class', ""),
                # 'class_conf': vehicle.get('class_conf', 0),
                'plate': vehicle.get('plate', ''),
                # 'plates': vehicle.get('plates', []),
                'plates': [{
                    'plate_bbox': [plate['x1'], plate['y1'], plate['x2'], plate['y2']],
                    # 'plate_text': text
                } for plate in vehicle.get('plates', [])],
            })
            
        return results
    

    def _detect_plates(self, vehicle_roi):
        """YOLOv8 plate detection with relative→absolute coordinates conversion"""
        plate_results = self._plate_detector(vehicle_roi)[0]
        plates = []

        for det in plate_results.boxes:
            x1, y1, x2, y2 = map(int, det.xyxy[0].tolist())
            plates.append({
                'x1': x1,
                'y1': y1,
                'x2': x2,
                'y2': y2,
                'roi': vehicle_roi[y1:y2, x1:x2]
            })
            return plates

        return plates

    def _recognize_plate(self, plate_roi):
        # """Enhanced OCR with preprocessing"""
        # try:
        #     # Preprocessing
        #     gray = cv2.cvtColor(plate_roi, cv2.COLOR_BGR2GRAY)
        #     gray = cv2.GaussianBlur(gray, (3, 3), 0)
        #     _, thresh = cv2.threshold(
        #         gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        #     # OCR with confidence threshold
        #     results = self.plate_reader.readtext(
        #         thresh, detail=0, paragraph=True)
        #     return results[0] if results else None

        # except Exception as e:
        #     print(f"OCR Error: {e}")
        #     return None
                # processed_plate = self._preprocess_plate(plate_roi)

        # processed_plate = self._preprocess_plate(plate_roi)

        plate_text = self._ocr_plate(plate_roi)
        
        validated_text = self._validate_russian_plate(plate_text)
        
        return validated_text

    def _preprocess_plate(self, plate_roi):
        """Предобработка изображения номера"""
        # Конвертация в grayscale
        gray = cv2.cvtColor(plate_roi, cv2.COLOR_BGR2GRAY)
        # cv2.imwrite('images/plate_1.jpg', gray)
        # Увеличение резкости
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(gray, -1, kernel)
        # cv2.imwrite('images/plate_2.jpg', sharpened)
        
        # Адаптивная бинаризация
        binary = cv2.adaptiveThreshold(
            sharpened, 255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        # cv2.imwrite('images/plate_3.jpg', binary)
        
        # # Морфологические операции для удаления шума
        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        # cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # cv2.imwrite('images/plate_4.jpg', cleaned)

        # Приведение к стандартному размеру
        resized = cv2.resize(binary, self._plate_size, interpolation=cv2.INTER_CUBIC)
        
        return resized
    
    def _ocr_plate(self, plate_image):
        """Распознавание текста на номере"""
        # Конвертация обратно в BGR для EasyOCR
        # plate_bgr = cv2.cvtColor(plate_image, cv2.COLOR_GRAY2BGR)
        license_plate_crop_gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY) 
        cv2.imwrite('images/plate.jpg', plate_image)
        # Распознавание с жесткими параметрами
        results = self._ocr.readtext(
            license_plate_crop_gray,
            decoder='beamsearch',  # Для последовательностей
            beamWidth=10,
            # batch_size=1,
            # allowlist=self._char_whitelist,
            detail=0
        )
        
        results.reverse()
        return "".join(results) if results else ''
    
    def _validate_russian_plate(self, text):
        """Проверка соответствия российскому стандарту"""
        if not text:
            return None
            
        # Удаляем все пробелы и неразрешенные символы
        cleaned = ''.join(c for c in text.upper() if c in self._char_whitelist)
        
        # Проверяем основные форматы:
        # Х999ХХ99 или Х999ХХ999
        if len(cleaned) > 7 and len(cleaned) < 10:
            # Проверяем формат буква+3 цифры+2 буквы+2-3 цифры
            pattern = (
                r'^[ABEKMHOPCTYXАВЕКМНОРСТУХ]{1}\d{3}[ABEKMHOPCTYXАВЕКМНОРСТУХ]{2}\d{2,3}$'
            )
            if re.match(pattern, cleaned):
                # Форматируем номер с пробелами: Х 999 ХХ 99
                return f"{cleaned[0]} {cleaned[1:4]} {cleaned[4:6]} {cleaned[6:]}"
                
        return None

    def _classify_vehicle(self, vehicle_roi):
        # """Vehicle classification with enhanced preprocessing"""
        try:
            # img = Image.fromarray(cv2.cvtColor(vehicle_roi, cv2.COLOR_BGR2RGB))
            vehicle_pil = Image.fromarray(vehicle_roi)
            # img_t = self.transform(img).unsqueeze(0).to(self.device)
            image_tensor = self._transform(vehicle_pil).unsqueeze(0).to(self._device)
            with torch.no_grad():
                outputs = self._classifier(image_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)[0] * 100
                _, preds = torch.max(outputs, 1)
                pred = preds[0]
                if probabilities.cpu().detach().numpy()[pred] > CONF_TRESHOLD:
                    return car_models[pred], True  

            return "Unknown", False

        except Exception as e:
            print(f"Classification Error: {e}")
            return "Unknown", False



# Example Usage
if __name__ == "__main__":
    processor = VehicleProcessor()
    detection_model = YOLO("yolo11n.pt") 

    img = cv2.imread('/home/ezaborshchikov/hse_mc/Masters_tethis/images/ter_2.png')
    detections = detection_model.track(img, persist=True, classes=[2])

    results = processor.process_frame(img, detections)


    video_path = '/home/ezaborshchikov/hse_mc/Masters_tethis/videos/parking_video_2.mp4'
    # input_path = Path(input_path)
    output_path = '/home/ezaborshchikov/hse_mc/Masters_tethis/videos/out_parking_video_2.mp4'
    # Process video stream
    cap = cv2.VideoCapture(video_path)

    # Get video properties for output
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    original_fps = cap.get(cv2.CAP_PROP_FPS)

    # Video writer setup (output at half the original FPS since we're skipping frames)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, original_fps/2, (frame_width, frame_height))

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        detections = detection_model.track(frame, persist=True, classes=[2])

        results = processor.process_frame(frame, detections)

        # Visualization
        for vehicle in results:
            x1, y1, x2, y2 = vehicle['bbox']
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            for plate in vehicle['plates']:
                px1, py1, px2, py2 = plate['plate_bbox']
                cv2.rectangle(frame, (px1+x1, py1+y1),
                              (px2+x1, py2+y1), (255, 0, 0), 2)

                if vehicle['plate']:
                    cv2.putText(frame, vehicle['plate'], (px1+x1, py1+y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                    

            color = (0, 255, 0)
            # label = f"ID: {vehicle.track_id} Class: {info['class']} Conf: {info['confidence']:.2f}"
            label = f"ID: {vehicle['track_id']} Class: {vehicle['class']}"
            cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        out.write(frame)
        frame_count += 1
 
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print(f"Processing complete. Frames: {frame_count}. Results saved to {output_path}")
