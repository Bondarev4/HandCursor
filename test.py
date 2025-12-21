"""
Hand Cursor Control
"""

import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import math
import time
from collections import deque

class HandCursor:
    def __init__(self):
        # MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=1,
            static_image_mode=False,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
            model_complexity=1
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Размеры экрана
        self.screen_width, self.screen_height = pyautogui.size()
        
        # Область отслеживания
        self.tracking_zone = {
            'x_min': 0.1,
            'x_max': 0.9,
            'y_min': 0.1,
            'y_max': 0.9
        }
        
        # Сглаживание
        self.mapping_smoothing = 0.7
        self.prev_x, self.prev_y = self.screen_width // 2, self.screen_height // 2
        self.position_history = deque(maxlen=3)
        
        # ОКРУЖНОСТЬ АКТИВАЦИИ
        self.palm_radius = 0.10  # Радиус вокруг ладони
        self.finger_extended_threshold = 0.20  # Палец считается поднятым
        self.finger_retracted_threshold = 0.14  # Палец считается прижатым
        
        # # Защита от ложных срабатываний
        # self.last_gesture_time = 0
        # self.gesture_cooldown = 0.1
        # self.gesture_buffer = deque(maxlen=3) 

        # Статистика
        self.frame_count = 0
        self.fps = 0
        self.prev_time = time.time()
        self.current_gesture = "Нет жеста"
        self.debug_info = ""
        
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # ВАЖНО: КАМЕРА НЕ ЗЕРКАЛЬНАЯ для одинакового движения
        self.mirror_view = False  # КАМЕРА НЕ зеркальная
        self.show_debug = True
        self.show_tracking_zone = True
        self.show_activation_circle = True
        self.mirror_control = True
        
        # Цвета
        self.colors = {
            'cursor': (0, 255, 0),      # Зеленый - курсор
            'lkm': (255, 0, 0),         # Синий - ЛКМ
            'pkm': (0, 0, 255),         # Красный - ПКМ
            'scroll_up': (255, 255, 0), # Желтый - скролл вверх
            'scroll_down': (0, 255, 255), # Голубой - скролл вниз
            'circle': (0, 200, 255),    # Голубой - окружность активации
            'circle_threshold': (0, 255, 0), # Зеленый - порог поднятия
            'palm_center': (255, 255, 0) # Желтый - центр ладони
        }
        
        print("✅ Система инициализирована - все двигается одинаково!")
    
    def calculate_palm_center(self, landmarks):
        """Вычисление центра ладони"""
        wrist = landmarks[self.mp_hands.HandLandmark.WRIST]
        middle_mcp = landmarks[self.mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
        
        return (wrist.x + middle_mcp.x) / 2, (wrist.y + middle_mcp.y) / 2
    
    def get_finger_tip_distance(self, finger_tip, palm_center):
        """Расстояние от кончика пальца до центра ладони"""
        dx = finger_tip.x - palm_center[0]
        dy = finger_tip.y - palm_center[1]
        return math.sqrt(dx*dx + dy*dy)
    
    def is_finger_raised(self, distance):
        """Определяет, поднят ли палец"""
        if distance > self.finger_extended_threshold:
            return "raised"
        elif distance < self.finger_retracted_threshold:
            return "retracted"
        else:
            return "neutral"
    
    def detect_gestures(self, landmarks, frame_width, frame_height):
        """Определение жестов на основе окружности"""
        gestures = {
            'cursor_move': False,
            'left_click': False,
            'right_click': False,
            'scroll_up': False,
            'scroll_down': False,
            'hand_detected': True
        }
        
        try:
            # Кончики пальцев
            thumb_tip = landmarks[self.mp_hands.HandLandmark.THUMB_TIP]
            index_tip = landmarks[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
            middle_tip = landmarks[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            pinky_tip = landmarks[self.mp_hands.HandLandmark.PINKY_TIP]
            
            # Центр ладони
            palm_center = self.calculate_palm_center(landmarks)
            palm_x = int(palm_center[0] * frame_width)
            palm_y = int(palm_center[1] * frame_height)
            
            # Расстояния до центра ладони
            thumb_dist = self.get_finger_tip_distance(thumb_tip, palm_center)
            index_dist = self.get_finger_tip_distance(index_tip, palm_center)
            middle_dist = self.get_finger_tip_distance(middle_tip, palm_center)
            pinky_dist = self.get_finger_tip_distance(pinky_tip, palm_center)
            
            # Статусы пальцев
            thumb_status = self.is_finger_raised(thumb_dist)
            index_status = self.is_finger_raised(index_dist)
            middle_status = self.is_finger_raised(middle_dist)
            pinky_status = self.is_finger_raised(pinky_dist)
            
            # Отладочная информация о расстояниях
            self.debug_info = f"У:{index_dist:.2f} Б:{thumb_dist:.2f} С:{middle_dist:.2f} М:{pinky_dist:.2f}"
            
            # Проверяем жесты
            index_raised = index_status == "raised"
            thumb_raised = thumb_status == "raised"
            middle_raised = middle_status == "raised"
            pinky_raised = pinky_status == "raised"
            
            # ЖЕСТЫ:
            
            # 1. ТОЛЬКО УКАЗАТЕЛЬНЫЙ → КУРСОР
            if index_raised and not thumb_raised and not middle_raised and not pinky_raised:
                gestures['cursor_move'] = True
                self.current_gesture = "Курсор"
                color = self.colors['cursor']
            
            # 2. УКАЗАТЕЛЬНЫЙ + БОЛЬШОЙ → ЛКМ
            elif index_raised and thumb_raised and not middle_raised and not pinky_raised:
                gestures['left_click'] = True
                self.current_gesture = "ЛКМ"
                color = self.colors['lkm']
            
            # 3. УКАЗАТЕЛЬНЫЙ + МИЗИНЕЦ → ПКМ
            elif index_raised and pinky_raised and not thumb_raised and not middle_raised:
                gestures['right_click'] = True
                self.current_gesture = "ПКМ"
                color = self.colors['pkm']
            
            # 4. УКАЗАТЕЛЬНЫЙ + СРЕДНИЙ → СКРОЛЛ
            elif index_raised and middle_raised and not thumb_raised and not pinky_raised:
                if index_tip.y < middle_tip.y:
                    gestures['scroll_up'] = True
                    self.current_gesture = "Скролл ↑"
                    color = self.colors['scroll_up']
                else:
                    gestures['scroll_down'] = True
                    self.current_gesture = "Скролл ↓"
                    color = self.colors['scroll_down']
            
            else:
                gestures['cursor_move'] = True
                self.current_gesture = "Курсор"
                color = self.colors['cursor']
            
            # Сохраняем информацию о пальцах для визуализации
            finger_info = {
                'thumb': (thumb_dist, thumb_status),
                'index': (index_dist, index_status),
                'middle': (middle_dist, middle_status),
                'pinky': (pinky_dist, pinky_status)
            }
            
            return gestures, palm_x, palm_y, color, self.current_gesture, palm_center, finger_info
            
        except Exception as e:
            gestures['hand_detected'] = False
            self.current_gesture = "Ошибка"
            return gestures, 0, 0, (255, 255, 255), self.current_gesture, (0, 0), {}
    
    def map_hand_to_screen(self, hand_x, hand_y, frame_width, frame_height):
        """Преобразование позиции руки в позицию курсора"""
        # Нормализованные координаты
        norm_x = hand_x / frame_width
        norm_y = hand_y / frame_height
        
        # ВАЖНО: Для управления курсором МЫ ЗЕРКАЛИМ!
        # Рука вправа на экране → Курсор вправа на экране
        # Это нужно для интуитивного управления
        norm_x = 1.0 - norm_x  # Зеркалим по X
        
        # Ограничиваем зоной отслеживания
        zone_x = np.clip(norm_x, self.tracking_zone['x_min'], self.tracking_zone['x_max'])
        zone_y = np.clip(norm_y, self.tracking_zone['y_min'], self.tracking_zone['y_max'])
        
        # Нормализуем относительно зоны
        zone_norm_x = (zone_x - self.tracking_zone['x_min']) / (self.tracking_zone['x_max'] - self.tracking_zone['x_min'])
        zone_norm_y = (zone_y - self.tracking_zone['y_min']) / (self.tracking_zone['y_max'] - self.tracking_zone['y_min'])
        
        # Преобразуем в экранные координаты
        screen_x = zone_norm_x * self.screen_width
        screen_y = zone_norm_y * self.screen_height
        
        # Сглаживание
        self.position_history.append((screen_x, screen_y))
        
        if len(self.position_history) > 0:
            avg_x = np.mean([p[0] for p in self.position_history])
            avg_y = np.mean([p[1] for p in self.position_history])
            
            smooth_x = self.prev_x * (1 - self.mapping_smoothing) + avg_x * self.mapping_smoothing
            smooth_y = self.prev_y * (1 - self.mapping_smoothing) + avg_y * self.mapping_smoothing
        else:
            smooth_x, smooth_y = screen_x, screen_y
        
        # Ограничиваем экраном
        smooth_x = np.clip(smooth_x, 10, self.screen_width - 10)
        smooth_y = np.clip(smooth_y, 10, self.screen_height - 10)
        
        self.prev_x, self.prev_y = smooth_x, smooth_y
        
        return int(smooth_x), int(smooth_y)
    
    def execute_commands(self, gestures, screen_x, screen_y):
        """Выполнение команд"""
        try:
            if gestures['cursor_move'] and "Курсор" in self.current_gesture:
                pyautogui.moveTo(screen_x, screen_y, duration=0.03)
            
            elif gestures['left_click']:
                pyautogui.click()
                print(f"[ЛКМ]")
                time.sleep(0.3)
            
            elif gestures['right_click']:
                pyautogui.rightClick()
                print(f"[ПКМ]")
                time.sleep(1.5)
            
            elif gestures['scroll_up']:
                pyautogui.scroll(10)
                print(f"[СКРОЛЛ ↑]")
                time.sleep(0.01)
            
            elif gestures['scroll_down']:
                pyautogui.scroll(-10)
                print(f"[СКРОЛЛ ↓]")
                time.sleep(0.01)
                
        except Exception as e:
            pass
    
    def draw_activation_circle(self, frame, palm_center_px, finger_info):
        """Отрисовка окружности активации ВОКРУГ РУКИ"""
        if not self.show_activation_circle:
            return
        
        frame_height, frame_width = frame.shape[:2]
        
        # Центр ладони в пикселях
        palm_x, palm_y = palm_center_px
        
        # Радиусы в пикселях
        palm_radius_px = int(self.palm_radius * min(frame_width, frame_height))
        threshold_radius_px = int(self.finger_extended_threshold * min(frame_width, frame_height))
        
        # 1. Круг ладони (внутренний, маленький)
        cv2.circle(frame, (palm_x, palm_y), palm_radius_px, self.colors['circle'], 1)
        
        # 2. Центр ладони (точка)
        cv2.circle(frame, (palm_x, palm_y), 5, self.colors['palm_center'], -1)
        cv2.circle(frame, (palm_x, palm_y), 7, self.colors['palm_center'], 1)
        
        # 3. Пороговый круг (зеленый) - где палец считается поднятым
        cv2.circle(frame, (palm_x, palm_y), threshold_radius_px, self.colors['circle_threshold'], 1)
        
        # 4. Визуализация пальцев
        finger_offsets = {
            'thumb': (-40, -30),
            'index': (0, -50),
            'middle': (0, -70),
            'pinky': (40, -30)
        }
        
        for finger_name, (dist, status) in finger_info.items():
            offset_x, offset_y = finger_offsets[finger_name]
            finger_x = palm_x + offset_x
            finger_y = palm_y + offset_y
            
            # Цвет в зависимости от статуса
            if status == "raised":
                color = (0, 255, 0)  # Зеленый - поднят
            elif status == "retracted":
                color = (255, 0, 0)  # Красный - прижат
            else:
                color = (200, 200, 200)  # Серый - нейтрально
            
            # Кружок пальца
            cv2.circle(frame, (finger_x, finger_y), 8, color, -1)
            
            # Буква обозначения
            letters = {'thumb': 'Б', 'index': 'У', 'middle': 'С', 'pinky': 'М'}
            cv2.putText(frame, letters[finger_name], (finger_x - 5, finger_y + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # Линия от центра к пальцу (если поднят)
            if status == "raised":
                cv2.line(frame, (palm_x, palm_y), (finger_x, finger_y), color, 1)
    
    def draw_tracking_zone(self, frame):
        """Отрисовка зоны отслеживания"""
        if not self.show_tracking_zone:
            return
        
        height, width = frame.shape[:2]
        
        # Координаты зоны
        x1 = int(self.tracking_zone['x_min'] * width)
        y1 = int(self.tracking_zone['y_min'] * height)
        x2 = int(self.tracking_zone['x_max'] * width)
        y2 = int(self.tracking_zone['y_max'] * height)
        
        # Полупрозрачный прямоугольник
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (100, 100, 255, 100), -1)
        frame = cv2.addWeighted(overlay, 0.1, frame, 0.9, 0)
        
        # Граница
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 255), 1)
    
    def draw_interface(self, frame, palm_x, palm_y, color, gesture_text, finger_info):
        """Отрисовка интерфейса"""
        if not self.show_debug:
            return
        
        # Зона отслеживания
        self.draw_tracking_zone(frame)
        
        # Окружность активации
        self.draw_activation_circle(frame, (palm_x, palm_y), finger_info)
        
        # Информационный блок
        info = [
            f"FPS: {self.fps}",
            f"Жест: {gesture_text}",
            f"Порог: {self.finger_extended_threshold:.2f}",
            f"Радиус: {self.palm_radius:.2f}",
            "",
            "🎮 Жесты:",
            "• Указательный = Курсор",
            "• Указательный+Большой = ЛКМ",
            "• Указательный+Мизинец = ПКМ",
            "• Указательный+Средний = Скролл"
        ]
        
        # Фон для информации
        overlay = frame.copy()
        cv2.rectangle(overlay, (5, 5), (300, 200), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
        
        # Текст
        y_pos = 25
        for i, line in enumerate(info):
            if i == 0: text_color = (255, 255, 0)
            elif i == 1: text_color = color
            elif i in [2, 3]: text_color = (100, 255, 255)
            elif i == 5: text_color = (255, 255, 255)
            else: text_color = (200, 200, 200)
            
            cv2.putText(frame, line, (10, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, text_color, 1)
            y_pos += 18
        
        # Отладочная информация внизу
        if self.debug_info:
            cv2.putText(frame, self.debug_info, (10, frame.shape[0] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 255, 100), 1)
        
        # Визуализация жеста (курсор)
        if "Курсор" in gesture_text:
            # Большой курсор для видимости
            cv2.circle(frame, (palm_x, palm_y), 20, color, 2)
            cv2.circle(frame, (palm_x, palm_y), 10, color, -1)
            cv2.circle(frame, (palm_x, palm_y), 5, (255, 255, 255), -1)
        else:
            # Просто точка для жестов
            cv2.circle(frame, (palm_x, palm_y), 12, color, -1)
        
        # Название жеста
        cv2.putText(frame, gesture_text, (palm_x + 25, palm_y - 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    def run(self):
        """Основной цикл"""
        print("=" * 70)
        print("🎯 УПРАВЛЕНИЕ - ВСЕ ДВИГАЕТСЯ ОДИНАКОВО")
        print("=" * 70)
        print("⚙️  ПРИНЦИП РАБОТЫ:")
        print("  • Камера: НЕ зеркальная")
        print("  • Движение: РУКА ВПРАВО → ВСЕ ВПРАВО")
        print("  • Круги: всегда вокруг руки")
        print("  • Курсор: следует за рукой")
        print("")
        print("🔵 СИНИЙ КРУГ: Зона ладони")
        print("🟢 ЗЕЛЕНЫЙ КРУГ: Порог поднятия пальца")
        print("🟡 ЖЕЛТАЯ ТОЧКА: Центр ладони")
        print("")
        print("🎮 ЖЕСТЫ:")
        print("  Поднимите палец ЗА зеленый круг:")
        print("  • Указательный = Курсор")
        print("  • Указательный+Большой = ЛКМ")
        print("  • Указательный+Мизинец = ПКМ")
        print("  • Указательный+Средний = Скролл")
        print("")
        print("⚙️  НАСТРОЙКА:")
        print("  +/- - изменить порог поднятия")
        print("  Z/X - изменить радиус ладони")
        print("  M - переключить зеркальность камеры")
        print("  H - скрыть/показать интерфейс")
        print("  Q - выход")
        print("=" * 70)
        
        try:
            while self.cap.isOpened():
                success, frame = self.cap.read()
                if not success:
                    continue
                
                frame_height, frame_width = frame.shape[:2]
                
                # Применяем зеркальность камеры (False = не зеркалим)
                if self.mirror_view:
                    frame = cv2.flip(frame, 1)
                
                # Обработка
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb_frame.flags.writeable = False
                results = self.hands.process(rgb_frame)
                rgb_frame.flags.writeable = True
                
                if results.multi_hand_landmarks:
                    hand_landmarks = results.multi_hand_landmarks[0]
                    
                    # Минимальная отрисовка скелета
                    self.mp_drawing.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing.DrawingSpec(color=(30, 30, 30), thickness=1),
                        self.mp_drawing.DrawingSpec(color=(100, 100, 100), thickness=1)
                    )
                    
                    # Определение жестов
                    gestures, palm_x, palm_y, color, gesture_text, palm_center, finger_info = \
                        self.detect_gestures(
                            hand_landmarks.landmark, frame_width, frame_height)
                    
                    # ВАЖНО: НЕ зеркалим координаты для отображения!
                    # Круги должны отображаться там, где реально находится рука
                    # Рука вправо → Круги справа от руки
                    
                    # Управление курсором
                    if gestures['cursor_move'] and "Курсор" in gesture_text:
                        screen_x, screen_y = self.map_hand_to_screen(
                            palm_x, palm_y, frame_width, frame_height)
                        self.execute_commands(gestures, screen_x, screen_y)
                    
                    elif any([gestures['left_click'], gestures['right_click'], 
                             gestures['scroll_up'], gestures['scroll_down']]):
                        self.execute_commands(gestures, 0, 0)
                    
                    # Отрисовка интерфейса
                    self.draw_interface(frame, palm_x, palm_y, color, 
                                       gesture_text, finger_info)
                
                # FPS
                self.frame_count += 1
                current_time = time.time()
                if current_time - self.prev_time >= 1.0:
                    self.fps = self.frame_count
                    self.frame_count = 0
                    self.prev_time = current_time
                
                # Отображение режима камеры
                mode_text = "Камера: ЗЕРКАЛЬНАЯ" if self.mirror_view else "Камера: НЕ зеркальная"
                mode_color = (255, 100, 100) if self.mirror_view else (100, 255, 100)
                cv2.putText(frame, mode_text, (frame_width - 200, 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, mode_color, 1)
                
                # Отображение
                cv2.imshow('Hand Control - Все двигается одинаково', frame)
                
                # Обработка клавиш
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('h'):
                    self.show_debug = not self.show_debug
                    self.show_tracking_zone = self.show_debug
                    self.show_activation_circle = self.show_debug
                elif key == ord('m'):
                    self.mirror_view = not self.mirror_view
                    status = "ЗЕРКАЛЬНАЯ" if self.mirror_view else "НЕ зеркальная"
                    print(f"📷 Режим камеры: {status}")
                elif key == ord('+'):  # Увеличить порог поднятия
                    self.finger_extended_threshold = min(0.4, self.finger_extended_threshold + 0.01)
                    print(f"📈 Порог: {self.finger_extended_threshold:.2f}")
                elif key == ord('-'):  # Уменьшить порог поднятия
                    self.finger_extended_threshold = max(0.15, self.finger_extended_threshold - 0.01)
                    print(f"📉 Порог: {self.finger_extended_threshold:.2f}")
                elif key == ord('z'):  # Уменьшить радиус ладони
                    self.palm_radius = max(0.05, self.palm_radius - 0.01)
                    print(f"📉 Радиус: {self.palm_radius:.2f}")
                elif key == ord('x'):  # Увеличить радиус ладони
                    self.palm_radius = min(0.25, self.palm_radius + 0.01)
                    print(f"📈 Радиус: {self.palm_radius:.2f}")
        
        except KeyboardInterrupt:
            print("\n⏹️ Прервано")
        except Exception as e:
            print(f"\n❌ Ошибка: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Очистка"""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        self.hands.close()
        print("✅ Система завершена")

if __name__ == "__main__":
    cursor = HandCursor()
    cursor.run()