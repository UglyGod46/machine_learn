class PassportValidationRules:
    """Класс с правилами проверки фотографий для паспорта РФ"""
    
    RULES = {
        'general': {
            'format': 'JPEG',
            'color_mode': 'RGB',
            'min_resolution': (420, 525),
            'max_resolution': (1200, 1500),
            'aspect_ratio': 4/3,
            'aspect_ratio_tolerance': 0.05,
            'file_size_max_mb': 5
        },
        
        'technical': {
            'min_brightness': 50,
            'max_brightness': 200,
            'min_contrast': 20,
            'max_noise': 10,
            'sharpness_threshold': 20,
            'color_cast_threshold': 15
        },
        
        'composition': {
            'face_height_min_ratio': 0.7,
            'face_height_max_ratio': 0.8,
            'face_center_tolerance_x': 0.1,
            'face_center_tolerance_y': 0.05,
            'eye_line_height_ratio': 0.4,
            'background_uniformity_threshold': 15
        },
        
        'pose': {
            'max_yaw_angle': 5,
            'max_pitch_angle': 5,
            'max_roll_angle': 5,
            'eyes_open_threshold': 0.25,
            'mouth_open_threshold': 0.3,
            'smile_threshold': 0.5
        },
        
        'appearance': {
            'no_glasses': True,
            'no_hat': True,
            'no_headphones': True,
            'no_uniform': True,
            'neutral_expression': True,
            'eyes_looking_forward': True,
            'hair_not_covering_eyes': True,
            'ears_visible': True
        },
        
        'background': {
            'color': 'light',
            'uniformity': 'high',
            'no_patterns': True,
            'no_shadows': True,
            'no_other_people': True,
            'no_objects': True
        }
    }
    
    @classmethod
    def check_resolution(cls, width, height):
        """Проверка разрешения"""
        min_w, min_h = cls.RULES['general']['min_resolution']
        max_w, max_h = cls.RULES['general']['max_resolution']
        
        return min_w <= width <= max_w and min_h <= height <= max_h
    
    @classmethod
    def check_aspect_ratio(cls, width, height):
        """Проверка соотношения сторон"""
        actual_ratio = width / height
        target_ratio = cls.RULES['general']['aspect_ratio']
        tolerance = cls.RULES['general']['aspect_ratio_tolerance']
        
        return abs(actual_ratio - target_ratio) <= tolerance
    
    @classmethod
    def get_all_rules(cls):
        """Получение всех правил в текстовом формате"""
        rules_text = []
        
        for category, rules in cls.RULES.items():
            rules_text.append(f"\n{category.upper()}:")
            for rule, value in rules.items():
                rules_text.append(f"  • {rule}: {value}")
        
        return '\n'.join(rules_text)