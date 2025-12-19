import sys
import os
import cv2
import numpy as np
from PIL import Image, ImageQt
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import warnings
warnings.filterwarnings('ignore')

import os
import sys

if sys.platform == 'win32':
    qt_plugin_path = os.path.join(
        sys.prefix, 'Lib', 'site-packages', 'PyQt5', 'Qt5', 'plugins'
    )
    if os.path.exists(qt_plugin_path):
        os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = qt_plugin_path
    else:
        qt_plugin_path = os.path.join(
            os.path.dirname(sys.executable), '..', 'Lib', 'site-packages', 'PyQt5', 'Qt5', 'plugins'
        )
        if os.path.exists(qt_plugin_path):
            os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = qt_plugin_path

print(f"QT plugin path: {os.environ.get('QT_QPA_PLATFORM_PLUGIN_PATH', 'not set')}")

class PhotoValidator:
    def __init__(self):
        self.rules = {
            'resolution': (420, 525),
            'aspect_ratio': 4/3,
            'background_color_tolerance': 40,
            'min_face_height_ratio': 0.3,
            'max_face_height_ratio': 0.4,
            'min_brightness': 50,
            'max_brightness': 200,
            'uniform_background_threshold': 15
        }
        
        self.face_cascade = None
        self.eye_cascade = None
        self.smile_cascade = None
        
        self._load_cascades()
    
    def _load_cascades(self):
        possible_paths = [
            "cascades",
            cv2.data.haarcascades,
        ]
    
        cascades_loaded = False
        
        for base_path in possible_paths:
            try:
                print(f"Пробуем загрузить каскады из: {base_path}")
                
                face_path = os.path.join(base_path, 'haarcascade_frontalface_default.xml')
                if os.path.exists(face_path):
                    self.face_cascade = cv2.CascadeClassifier(face_path)
                    if not self.face_cascade.empty():
                        print(f"✓ Загружен face каскад: {face_path}")
                        cascades_loaded = True
                else:
                    print(f"  Файл не найден: {face_path}")
                
                eye_path = os.path.join(base_path, 'haarcascade_eye.xml')
                if os.path.exists(eye_path):
                    self.eye_cascade = cv2.CascadeClassifier(eye_path)
                    if not self.eye_cascade.empty():
                        print(f"✓ Загружен eye каскад: {eye_path}")
                
                smile_path = os.path.join(base_path, 'haarcascade_smile.xml')
                if os.path.exists(smile_path):
                    self.smile_cascade = cv2.CascadeClassifier(smile_path)
                    if not self.smile_cascade.empty():
                        print(f"✓ Загружен smile каскад: {smile_path}")
                
                if cascades_loaded:
                    break
                    
            except Exception as e:
                print(f"  Ошибка при загрузке из {base_path}: {e}")
                continue
        
        if not cascades_loaded:
            print("ВНИМАНИЕ: Не удалось загрузить каскады для детекции лиц!")
            print("Будет использована упрощенная проверка по яркости и контрасту.")
        else:
            print("✓ Все каскады успешно загружены!")
    
    def detect_faces_simple(self, gray_img):
        """Упрощенная детекция лиц на основе анализа изображения"""
        h, w = gray_img.shape
        
        grid_size = 10
        cell_h = h // grid_size
        cell_w = w // grid_size
        
        max_contrast = 0
        best_cell = (0, 0, cell_w, cell_h)
        
        for i in range(grid_size - 1):
            for j in range(grid_size - 1):
                y1 = i * cell_h
                y2 = y1 + cell_h * 2
                x1 = j * cell_w
                x2 = x1 + cell_w * 2
                
                if y2 < h and x2 < w:
                    cell = gray_img[y1:y2, x1:x2]
                    if cell.size > 0:
                        contrast = np.std(cell)
                        if contrast > max_contrast:
                            max_contrast = contrast
                            best_cell = (x1, y1, x2-x1, y2-y1)
        
        if max_contrast > 20:
            return [best_cell]
        return []
    
    def validate_passport_photo(self, image_path):
        """Основная функция проверки фото на соответствие требованиям"""
        results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'face_detected': False,
            'face_position': None,
            'background_score': 0,
            'lighting_score': 0,
            'attributes': {}
        }
        
        img = cv2.imread(image_path)
        if img is None:
            results['errors'].append("Не удалось загрузить изображение")
            results['is_valid'] = False
            return results
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        
        if abs(w/h - self.rules['aspect_ratio']) > 0.05:
            results['warnings'].append(f"Соотношение сторон должно быть 4:3 (получено {w}:{h})")
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)
        
        if brightness < self.rules['min_brightness']:
            results['errors'].append("Изображение слишком темное")
            results['is_valid'] = False
        elif brightness > self.rules['max_brightness']:
            results['errors'].append("Изображение пересвечено")
            results['is_valid'] = False
        
        results['lighting_score'] = 100 - abs(brightness - 125) / 125 * 100
        
        contrast = np.std(gray)
        if contrast < 20:
            results['warnings'].append("Изображение недостаточно контрастное")
        
        faces = []
        if self.face_cascade and not self.face_cascade.empty():
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(int(w*0.1), int(h*0.1))
            )
        else:
            faces = self.detect_faces_simple(gray)
            if len(faces) > 0:
                results['warnings'].append("Используется упрощенная детекция лица")
        
        if len(faces) == 0:
            results['errors'].append("Лицо не обнаружено")
            results['is_valid'] = False
            face_box = [int(w*0.3), int(h*0.3), int(w*0.7), int(h*0.7)]
        else:
            results['face_detected'] = True
            x, y, face_w, face_h = faces[0]
            face_box = [x, y, x + face_w, y + face_h]
            results['face_position'] = face_box
            
            face_ratio = face_h / h
            
            if face_ratio < self.rules['min_face_height_ratio']:
                results['errors'].append("Лицо слишком маленькое")
                results['is_valid'] = False
            elif face_ratio > self.rules['max_face_height_ratio']:
                results['errors'].append("Лицо слишком большое")
                results['is_valid'] = False
            
            face_center_x = x + face_w/2
            face_center_y = y + face_h/2
            
            if abs(face_center_x - w/2) > 0.1 * w:
                results['errors'].append("Лицо не по центру по горизонтали")
                results['is_valid'] = False
            
            if abs(face_center_y - h/2) > 0.15 * h:
                results['errors'].append("Лицо не по центру по вертикали")
                results['is_valid'] = False
        
        background_score = self._check_background(img, face_box)
        results['background_score'] = background_score
        
        if background_score < 70:
            results['errors'].append("Фон не однородный или недостаточно яркий")
            results['is_valid'] = False
        
        if results['face_detected']:
            attributes = self._check_face_attributes(img, face_box)
            results['attributes'] = attributes
            
            if not attributes.get('eyes_detected', False):
                results['warnings'].append("Глаза не обнаружены (возможно закрыты)")
            
            if attributes.get('smile_detected', False):
                results['errors'].append("Обнаружена улыбка")
                results['is_valid'] = False
        else:
            results['attributes'] = {'eyes_detected': False, 'smile_detected': False}
        
        return results
    
    def _check_background(self, img, face_box):
        """Проверка фона на однородность и цвет"""
        h, w = img.shape[:2]
        
        mask = np.ones((h, w), dtype=np.uint8) * 255
        
        x1, y1, x2, y2 = map(int, face_box)
        margin = 20
        mask[max(0, y1-margin):min(h, y2+margin), 
             max(0, x1-margin):min(w, x2+margin)] = 0
        
        background_pixels = img[mask == 255]
        
        if len(background_pixels) == 0:
            return 50
        
        std_bgr = np.std(background_pixels, axis=0)
        uniformity_score = max(0, 100 - np.mean(std_bgr) / 2)
        
        brightness = np.mean(cv2.cvtColor(background_pixels.reshape(-1, 1, 3), 
                                         cv2.COLOR_BGR2GRAY))
        
        brightness_score = min(100, brightness / 255 * 100)
        
        final_score = (uniformity_score * 0.6) + (brightness_score * 0.4)
        
        return min(100, final_score)
    
    def _check_face_attributes(self, img, face_box):
        """Проверка атрибутов лица с использованием OpenCV"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        x1, y1, x2, y2 = map(int, face_box)
        face_roi_gray = gray[y1:y2, x1:x2]
        face_roi_color = img[y1:y2, x1:x2]
        
        attributes = {}
        
        if self.eye_cascade and not self.eye_cascade.empty() and face_roi_gray.size > 0:
            eyes = self.eye_cascade.detectMultiScale(
                face_roi_gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(int((x2-x1)*0.1), int((y2-y1)*0.1))
            )
            attributes['eyes_detected'] = len(eyes) >= 2
            attributes['eyes_count'] = len(eyes)
        else:
            if face_roi_gray.size > 0:
                eye_region = face_roi_gray[:int((y2-y1)*0.5), :]
                if eye_region.size > 0:
                    eye_contrast = np.std(eye_region)
                    attributes['eyes_detected'] = eye_contrast > 25
                    attributes['eye_contrast'] = eye_contrast
        
        if self.smile_cascade and not self.smile_cascade.empty() and face_roi_gray.size > 0:
            smile_zone = face_roi_gray[int((y2-y1)*0.6):, :]
            if smile_zone.size > 0:
                smiles = self.smile_cascade.detectMultiScale(
                    smile_zone,
                    scaleFactor=1.8,
                    minNeighbors=20,
                    minSize=(int((x2-x1)*0.2), int((y2-y1)*0.1))
                )
                attributes['smile_detected'] = len(smiles) > 0
                attributes['smile_count'] = len(smiles)
        else:
            if face_roi_gray.size > 0:
                mouth_region = face_roi_gray[int((y2-y1)*0.6):, :]
                if mouth_region.size > 0:
                    mouth_contrast = np.std(mouth_region)
                    attributes['smile_detected'] = mouth_contrast > 40
                    attributes['mouth_contrast'] = mouth_contrast
        
        if 'eyes_detected' in attributes and face_roi_gray.size > 0:
            eye_region = gray[max(0, y1-int((y2-y1)*0.3)):min(gray.shape[0], y2), 
                            max(0, x1-int((x2-x1)*0.2)):min(gray.shape[1], x2+int((x2-x1)*0.2))]
            if eye_region.size > 0:
                eye_contrast = np.std(eye_region)
                attributes['has_glasses'] = eye_contrast < 25
                attributes['eye_region_contrast'] = eye_contrast
        
        return attributes

class ImageEditor(QWidget):
    """Виджет для редактирования фотографии"""
    def __init__(self):
        super().__init__()
        self.original_image = None
        self.display_image = None
        self.offset = QPoint(0, 0)
        self.scale = 1.0
        self.rotation = 0
        self.setMouseTracking(True)
        self.dragging = False
        self.last_mouse_pos = None
        self.image_loaded = False
        
        self.template_width = 400
        self.template_height = 300
        self.passport_template = QRect(0, 0, self.template_width, self.template_height)
        
        self.template_color = QColor(0, 120, 215)
        self.grid_color = QColor(255, 0, 0, 100)
        self.bg_color = QColor(240, 240, 240)
    
    def load_image(self, image_path):
        """Загрузка изображения"""
        try:
            self.original_image = QImage(image_path)
            if self.original_image.isNull():
                print(f"Не удалось загрузить изображение: {image_path}")
                return False
            
            self.image_loaded = True
            
            self.display_image = self.original_image.copy()
            
            self._calculate_initial_scale()
            
            self.offset = QPoint(0, 0)
            self.rotation = 0
            
            self.update()
            return True
            
        except Exception as e:
            print(f"Ошибка при загрузке изображения: {e}")
            return False
    
    def _calculate_initial_scale(self):
        """Вычисление начального масштаба для вписывания в шаблон"""
        if not self.display_image:
            return
        
        img_width = self.display_image.width()
        img_height = self.display_image.height()
        template_width = self.template_width
        template_height = self.template_height
        
        scale_width = template_width / img_width
        scale_height = template_height / img_height
        
        self.scale = min(scale_width, scale_height) * 0.9
        
        scaled_width = img_width * self.scale
        scaled_height = img_height * self.scale
        
        offset_x = int((template_width - scaled_width) // 2)
        offset_y = int((template_height - scaled_height) // 2)
        
        self.offset = QPoint(offset_x, offset_y)
    
    def paintEvent(self, event):
        """Отрисовка виджета"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        painter.fillRect(self.rect(), self.bg_color)
        
        painter.setPen(QPen(self.template_color, 2))
        painter.setBrush(Qt.NoBrush)
        painter.drawRect(self.passport_template)
        
        painter.setPen(QPen(self.grid_color, 1, Qt.DashLine))
        
        center_x = self.template_width // 2
        painter.drawLine(center_x, 0, center_x, self.template_height)
        
        center_y = self.template_height // 2
        painter.drawLine(0, center_y, self.template_width, center_y)
        
        third_x = self.template_width // 3
        two_third_x = third_x * 2
        third_y = self.template_height // 3
        two_third_y = third_y * 2
        
        painter.setPen(QPen(self.grid_color, 1, Qt.DotLine))
        painter.drawLine(third_x, 0, third_x, self.template_height)
        painter.drawLine(two_third_x, 0, two_third_x, self.template_height)
        painter.drawLine(0, third_y, self.template_width, third_y)
        painter.drawLine(0, two_third_y, self.template_width, two_third_y)
        
        painter.setPen(QPen(QColor(255, 0, 0, 200), 2))
        cross_size = 20
        painter.drawLine(center_x - cross_size, center_y, center_x + cross_size, center_y)
        painter.drawLine(center_x, center_y - cross_size, center_x, center_y + cross_size)
        
        if self.image_loaded and self.display_image:
            painter.save()
            
            painter.translate(self.template_width // 2, self.template_height // 2)
            
            painter.rotate(self.rotation)
            
            painter.scale(self.scale, self.scale)
            
            img_width = self.display_image.width()
            img_height = self.display_image.height()
            
            translate_x = -img_width // 2 + int(self.offset.x() / self.scale)
            translate_y = -img_height // 2 + int(self.offset.y() / self.scale)
            
            painter.translate(translate_x, translate_y)
            
            painter.drawImage(0, 0, self.display_image)
            
            painter.restore()
    
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self.image_loaded:
            self.dragging = True
            self.last_mouse_pos = event.pos()
    
    def mouseMoveEvent(self, event):
        if self.dragging and self.last_mouse_pos and self.image_loaded:
            delta = event.pos() - self.last_mouse_pos
            self.offset += delta
            self.last_mouse_pos = event.pos()
            self.update()
    
    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.dragging = False
    
    def wheelEvent(self, event):
        """Масштабирование колесиком мыши"""
        if not self.image_loaded:
            return
        
        mouse_pos = event.pos()
        if not self.passport_template.contains(mouse_pos):
            return
        
        delta = event.angleDelta().y()
        
        old_scale = self.scale
        old_offset = self.offset
        
        img_center_x = self.template_width // 2
        img_center_y = self.template_height // 2
        
        scale_factor = 1.1 if delta > 0 else 0.9
        self.scale *= scale_factor
        
        self.scale = max(0.1, min(5.0, self.scale))
        
        scale_ratio = self.scale / old_scale
        
        new_offset_x = int((old_offset.x() - img_center_x) * scale_ratio + img_center_x)
        new_offset_y = int((old_offset.y() - img_center_y) * scale_ratio + img_center_y)
        
        self.offset = QPoint(new_offset_x, new_offset_y)
        
        self.update()
    
    def rotate(self, angle):
        """Поворот изображения"""
        if not self.image_loaded:
            return
        
        self.rotation += angle
        self.rotation %= 360
        
        transform = QTransform()
        transform.rotate(angle)
        self.display_image = self.original_image.transformed(transform, Qt.SmoothTransformation)
        
        self.update()
    
    def reset(self):
        """Сброс трансформаций"""
        if not self.image_loaded:
            return
        
        self.display_image = self.original_image.copy()
        self._calculate_initial_scale()
        self.rotation = 0
        self.update()
    
    def get_processed_image(self):
        """Получение обработанного изображения (обрезанного под шаблон 4:3)"""
        if not self.image_loaded or not self.display_image:
            return None
        
        result_image = QImage(self.template_width, self.template_height, QImage.Format_RGB32)
        result_image.fill(Qt.white)
        
        painter = QPainter(result_image)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)
        
        painter.translate(self.template_width // 2, self.template_height // 2)
        painter.rotate(self.rotation)
        painter.scale(self.scale, self.scale)
        
        img_width = self.display_image.width()
        img_height = self.display_image.height()
        
        translate_x = -img_width // 2 + int(self.offset.x() / self.scale)
        translate_y = -img_height // 2 + int(self.offset.y() / self.scale)
        
        painter.translate(translate_x, translate_y)
        
        painter.drawImage(0, 0, self.display_image)
        painter.end()
        
        return result_image
    
    def get_cropped_image(self):
        """Получение изображения, обрезанного по шаблону"""
        processed = self.get_processed_image()
        if processed:
            cropped = QImage(self.template_width, self.template_height, QImage.Format_RGB32)
            cropped.fill(Qt.white)
            
            painter = QPainter(cropped)
            painter.drawImage(0, 0, processed)
            painter.end()
            
            return cropped
        return None


class PassportPhotoApp(QMainWindow):
    """Главное окно приложения"""
    def __init__(self):
        super().__init__()
        self.validator = PhotoValidator()
        self.current_image_path = None
        self.init_ui()
        
    def init_ui(self):
        """Инициализация интерфейса"""
        self.setWindowTitle('Проверка фотографии для паспорта РФ')
        self.setGeometry(100, 100, 1000, 600)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QHBoxLayout(central_widget)
        
        left_panel = QFrame()
        left_panel.setFrameStyle(QFrame.StyledPanel)
        left_layout = QVBoxLayout(left_panel)
        
        self.image_editor = ImageEditor()
        self.image_editor.setMinimumSize(400, 300)
        left_layout.addWidget(self.image_editor)
        
        control_panel = QWidget()
        control_layout = QHBoxLayout(control_panel)
        
        btn_load = QPushButton('Загрузить фото')
        btn_load.clicked.connect(self.load_image)
        
        btn_rotate_left = QPushButton('↺')
        btn_rotate_left.clicked.connect(lambda: self.image_editor.rotate(-90))
        
        btn_rotate_right = QPushButton('↻')
        btn_rotate_right.clicked.connect(lambda: self.image_editor.rotate(90))
        
        btn_zoom_in = QPushButton('+')
        btn_zoom_in.clicked.connect(self.zoom_in)
        
        btn_zoom_out = QPushButton('-')
        btn_zoom_out.clicked.connect(self.zoom_out)
        
        btn_reset = QPushButton('Сброс')
        btn_reset.clicked.connect(self.image_editor.reset)
        
        control_layout.addWidget(btn_load)
        control_layout.addWidget(btn_rotate_left)
        control_layout.addWidget(btn_rotate_right)
        control_layout.addWidget(btn_zoom_in)
        control_layout.addWidget(btn_zoom_out)
        control_layout.addWidget(btn_reset)
        
        left_layout.addWidget(control_panel)
        
        right_panel = QFrame()
        right_panel.setFrameStyle(QFrame.StyledPanel)
        right_layout = QVBoxLayout(right_panel)
        
        title = QLabel('Результаты проверки')
        title.setStyleSheet('font-size: 18px; font-weight: bold; margin: 10px;')
        right_layout.addWidget(title)
        
        self.status_label = QLabel('Статус: Не проверено')
        self.status_label.setStyleSheet('font-size: 16px; margin: 10px;')
        right_layout.addWidget(self.status_label)
        
        self.details_text = QTextEdit()
        self.details_text.setReadOnly(True)
        self.details_text.setMaximumHeight(250)
        right_layout.addWidget(QLabel('Детали:'))
        right_layout.addWidget(self.details_text)
        
        btn_validate = QPushButton('Проверить фотографию')
        btn_validate.clicked.connect(self.validate_photo)
        btn_validate.setStyleSheet('''
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-size: 16px;
                padding: 12px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        ''')
        right_layout.addWidget(btn_validate)
        
        instruction = QLabel('''
        <b>Требования к фото для паспорта РФ:</b><br>
        • Размер: 35×45 мм, соотношение сторон 4:3<br>
        • Цветное, на светлом однородном фоне<br>
        • Лицо занимает 70-80% фото<br>
        • Голова расположена по центру<br>
        • Нейтральное выражение лица, без улыбки<br>
        • Глаза открыты, смотрят прямо<br>
        • Хорошее равномерное освещение
        ''')
        instruction.setWordWrap(True)
        instruction.setStyleSheet('background-color: #f0f0f0; padding: 10px; border-radius: 5px;')
        right_layout.addWidget(instruction)
        
        layout.addWidget(left_panel, 60)
        layout.addWidget(right_panel, 40)
        
        self.statusBar().showMessage('Готово к загрузке фотографии')
    
    def zoom_in(self):
        """Увеличение масштаба"""
        self.image_editor.scale *= 1.1
        self.image_editor.scale = min(5.0, self.image_editor.scale)
        self.image_editor.update()
    
    def zoom_out(self):
        """Уменьшение масштаба"""
        self.image_editor.scale *= 0.9
        self.image_editor.scale = max(0.1, self.image_editor.scale)
        self.image_editor.update()
    
    def load_image(self):
        """Загрузка изображения"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, 'Выберите фотографию', '',
            'Images (*.jpg *.jpeg *.png *.bmp)'
        )
        
        if file_path:
            self.current_image_path = file_path
            if self.image_editor.load_image(file_path):
                self.statusBar().showMessage(f'Загружено: {os.path.basename(file_path)}')
                self.status_label.setText('Статус: Не проверено')
                self.status_label.setStyleSheet('font-size: 16px; margin: 10px; color: black;')
                self.details_text.clear()
            else:
                QMessageBox.warning(self, 'Ошибка', 'Не удалось загрузить изображение')
    
    def validate_photo(self):
        """Проверка фотографии"""
        if not self.current_image_path:
            QMessageBox.warning(self, 'Предупреждение', 'Сначала загрузите фотографию')
            return
        
        temp_image = self.image_editor.get_processed_image()
        if temp_image:
            temp_path = 'temp_processed_photo.jpg'
            temp_image.save(temp_path, 'JPEG', 90)
            image_to_validate = temp_path
        else:
            image_to_validate = self.current_image_path
        
        self.statusBar().showMessage('Проверка фотографии...')
        QApplication.processEvents()
        
        results = self.validator.validate_passport_photo(image_to_validate)
        
        self.display_results(results)
        
        if os.path.exists('temp_processed_photo.jpg'):
            try:
                os.remove('temp_processed_photo.jpg')
            except:
                pass
    
    def display_results(self, results):
        """Отображение результатов проверки"""
        details = []
        
        if results['is_valid']:
            self.status_label.setText('✅ Фотография соответствует требованиям')
            self.status_label.setStyleSheet('font-size: 16px; margin: 10px; color: green;')
        else:
            self.status_label.setText('❌ Фотография не соответствует требованиям')
            self.status_label.setStyleSheet('font-size: 16px; margin: 10px; color: red;')
        
        details.append(f"<b>Общий результат:</b> {'✅ ПРОШЛО' if results['is_valid'] else '❌ НЕ ПРОШЛО'}")
        details.append(f"<b>Оценка освещения:</b> {results['lighting_score']:.1f}/100")
        details.append(f"<b>Оценка фона:</b> {results['background_score']:.1f}/100")
        
        if results['attributes']:
            details.append("<br><b>Анализ лица:</b>")
            if 'eyes_detected' in results['attributes']:
                details.append(f"• Глаза обнаружены: {'да' if results['attributes']['eyes_detected'] else 'нет'}")
            if 'smile_detected' in results['attributes']:
                details.append(f"• Улыбка: {'да' if results['attributes']['smile_detected'] else 'нет'}")
            if 'has_glasses' in results['attributes']:
                details.append(f"• Очки: {'да' if results['attributes']['has_glasses'] else 'нет'}")
        
        if results['errors']:
            details.append("<br><b>Обнаруженные проблемы:</b>")
            for error in results['errors']:
                details.append(f"• ❌ {error}")
        
        if results['warnings']:
            details.append("<br><b>Предупреждения:</b>")
            for warning in results['warnings']:
                details.append(f"• ⚠ {warning}")
        
        self.details_text.setHtml('<br>'.join(details))
        self.statusBar().showMessage('Проверка завершена')


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(240, 240, 240))
    palette.setColor(QPalette.WindowText, QColor(0, 0, 0))
    palette.setColor(QPalette.Base, QColor(255, 255, 255))
    palette.setColor(QPalette.AlternateBase, QColor(240, 240, 240))
    palette.setColor(QPalette.ToolTipBase, QColor(255, 255, 255))
    palette.setColor(QPalette.ToolTipText, QColor(0, 0, 0))
    palette.setColor(QPalette.Text, QColor(0, 0, 0))
    palette.setColor(QPalette.Button, QColor(240, 240, 240))
    palette.setColor(QPalette.ButtonText, QColor(0, 0, 0))
    palette.setColor(QPalette.BrightText, QColor(255, 0, 0))
    palette.setColor(QPalette.Link, QColor(42, 130, 218))
    palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
    palette.setColor(QPalette.HighlightedText, QColor(255, 255, 255))
    app.setPalette(palette)
    
    window = PassportPhotoApp()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()