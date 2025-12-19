import torch
import torch.nn as nn
import torchvision.models as models

class FaceAttributeClassifier(nn.Module):
    """Классификатор атрибутов лица"""
    
    def __init__(self, num_attributes=6):
        super().__init__()
        
        # Используем предобученный ResNet
        backbone = models.resnet50(pretrained=True)
        
        # Заменяем последний слой
        num_features = backbone.fc.in_features
        
        # Основные слои
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        
        # Атрибуты для классификации
        self.attributes = nn.ModuleDict({
            'glasses': nn.Sequential(
                nn.Linear(num_features, 256),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(256, 2)
            ),
            'hat': nn.Sequential(
                nn.Linear(num_features, 256),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(256, 2)
            ),
            'smile': nn.Sequential(
                nn.Linear(num_features, 256),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(256, 2)
            ),
            'eyes_open': nn.Sequential(
                nn.Linear(num_features, 256),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(256, 2)
            ),
            'gender': nn.Sequential(
                nn.Linear(num_features, 256),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(256, 2)
            ),
            'age': nn.Sequential(
                nn.Linear(num_features, 256),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(256, 4)  # возрастные группы
            )
        })
    
    def forward(self, x):
        features = self.backbone(x)
        features = features.view(features.size(0), -1)
        
        outputs = {}
        for attr_name, attr_classifier in self.attributes.items():
            outputs[attr_name] = attr_classifier(features)
        
        return outputs