"""
NVIDIA Hydra-MDP Implementation for CARLA
Multi-teacher knowledge distillation architecture
Based on CVPR 2024 E2E Driving at Scale Challenge Winner
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import math

class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism for feature fusion"""
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Linear transformations
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(attention_scores, dim=-1)
        context = torch.matmul(attention_weights, V)
        
        # Concatenate heads
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        return self.W_o(context)

class SafetyAwareHead(nn.Module):
    """Safety-aware driving score prediction head"""
    def __init__(self, input_dim):
        super().__init__()
        self.safety_net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()  # Safety score [0, 1]
        )
        
        self.collision_risk = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        self.traffic_compliance = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return {
            'safety_score': self.safety_net(x),
            'collision_risk': self.collision_risk(x),
            'traffic_compliance': self.traffic_compliance(x)
        }

class HumanTeacher(nn.Module):
    """Human behavior teacher network"""
    def __init__(self, input_dim):
        super().__init__()
        self.human_encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        self.steering_head = nn.Linear(128, 1)
        self.speed_head = nn.Linear(128, 1)
        self.comfort_head = nn.Linear(128, 1)  # Driving comfort score
    
    def forward(self, x):
        features = self.human_encoder(x)
        return {
            'steering': self.steering_head(features),
            'speed': self.speed_head(features),
            'comfort': torch.sigmoid(self.comfort_head(features))
        }

class RuleBasedTeacher(nn.Module):
    """Rule-based planning teacher network"""
    def __init__(self, input_dim):
        super().__init__()
        self.rule_encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        self.lane_keeping = nn.Linear(128, 1)
        self.speed_limit = nn.Linear(128, 1)
        self.following_distance = nn.Linear(128, 1)
        self.traffic_signal = nn.Linear(128, 3)  # Red, Yellow, Green
    
    def forward(self, x):
        features = self.rule_encoder(x)
        return {
            'lane_keeping': torch.tanh(self.lane_keeping(features)),
            'speed_limit': torch.sigmoid(self.speed_limit(features)),
            'following_distance': torch.sigmoid(self.following_distance(features)),
            'traffic_signal': F.softmax(self.traffic_signal(features), dim=-1)
        }

class HydraMDP(nn.Module):
    """
    Hydra-MDP: Multi-teacher Knowledge Distillation for End-to-End Autonomous Driving
    CVPR 2024 E2E Driving at Scale Challenge Winner
    """
    def __init__(self):
        super().__init__()
        
        # Vision backbone - EfficientNet-like architecture
        self.vision_backbone = nn.Sequential(
            # Stage 1: 224x224 -> 112x112
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            # Stage 2: 112x112 -> 56x56
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            # Stage 3: 56x56 -> 28x28
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            # Stage 4: 28x28 -> 14x14
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            
            # Stage 5: 14x14 -> 7x7
            nn.Conv2d(256, 512, 3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            
            # Global average pooling
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Calculate feature dimension after vision backbone
        self.feature_dim = 512
        
        # Multi-teacher architecture
        self.human_teacher = HumanTeacher(self.feature_dim)
        self.rule_teacher = RuleBasedTeacher(self.feature_dim)
        
        # Multi-head attention for teacher fusion
        self.teacher_attention = MultiHeadAttention(self.feature_dim, num_heads=8)
        
        # Student network (main driving policy)
        self.student_encoder = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.4)
        )
        
        # Driving action heads
        self.steering_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        self.throttle_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        self.brake_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Safety-aware components
        self.safety_head = SafetyAwareHead(256)
        
        # Waypoint prediction for planning
        self.waypoint_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 20)  # 10 waypoints (x, y)
        )
        
        # Speed prediction
        self.speed_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.ReLU()  # Non-negative speed
        )
    
    def forward(self, image, training=True):
        # Extract visual features
        batch_size = image.size(0)
        
        # Vision backbone
        visual_features = self.vision_backbone(image / 255.0)
        visual_features = visual_features.view(batch_size, -1)
        
        outputs = {}
        
        if training:
            # Multi-teacher knowledge distillation
            human_outputs = self.human_teacher(visual_features)
            rule_outputs = self.rule_teacher(visual_features)
            
            # Teacher attention fusion
            teacher_features = visual_features.unsqueeze(1)  # Add sequence dimension
            attended_features = self.teacher_attention(
                teacher_features, teacher_features, teacher_features
            ).squeeze(1)
            
            outputs['teacher'] = {
                'human': human_outputs,
                'rule': rule_outputs
            }
        else:
            attended_features = visual_features
        
        # Student network (main policy)
        student_features = self.student_encoder(attended_features)
        
        # Driving actions
        steering = self.steering_head(student_features)
        throttle = self.throttle_head(student_features)
        brake = self.brake_head(student_features)
        
        # Safety predictions
        safety_outputs = self.safety_head(student_features)
        
        # Planning outputs
        waypoints = self.waypoint_head(student_features)
        waypoints = waypoints.view(batch_size, 10, 2)  # Reshape to (batch, 10, 2)
        
        speed = self.speed_head(student_features)
        
        outputs.update({
            'steering': steering,
            'throttle': throttle,
            'brake': brake,
            'waypoints': waypoints,
            'speed': speed,
            'safety': safety_outputs
        })
        
        return outputs

class HydraMDPLoss(nn.Module):
    """Multi-task loss for Hydra-MDP training"""
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        self.l1_loss = nn.L1Loss()
    
    def forward(self, predictions, targets, training=True):
        total_loss = 0.0
        loss_dict = {}
        
        # Main driving actions
        if 'steering' in targets:
            steering_loss = self.mse_loss(predictions['steering'], targets['steering'])
            total_loss += 2.0 * steering_loss  # Steering is critical
            loss_dict['steering'] = steering_loss.item()
        
        if 'throttle' in targets:
            throttle_loss = self.mse_loss(predictions['throttle'], targets['throttle'])
            total_loss += 1.0 * throttle_loss
            loss_dict['throttle'] = throttle_loss.item()
        
        if 'brake' in targets:
            brake_loss = self.mse_loss(predictions['brake'], targets['brake'])
            total_loss += 1.0 * brake_loss
            loss_dict['brake'] = brake_loss.item()
        
        # Waypoint planning
        if 'waypoints' in targets:
            waypoint_loss = self.l1_loss(predictions['waypoints'], targets['waypoints'])
            total_loss += 1.5 * waypoint_loss
            loss_dict['waypoint'] = waypoint_loss.item()
        
        # Speed prediction
        if 'speed' in targets:
            speed_loss = self.mse_loss(predictions['speed'], targets['speed'])
            total_loss += 0.5 * speed_loss
            loss_dict['speed'] = speed_loss.item()
        
        # Safety components
        if 'safety_score' in targets:
            safety_loss = self.bce_loss(
                predictions['safety']['safety_score'], 
                targets['safety_score']
            )
            total_loss += 1.0 * safety_loss
            loss_dict['safety'] = safety_loss.item()
        
        # Teacher distillation losses (during training)
        if training and 'teacher' in predictions:
            # Human teacher distillation
            if 'human_steering' in targets:
                human_steering_loss = self.mse_loss(
                    predictions['teacher']['human']['steering'],
                    targets['human_steering']
                )
                total_loss += 0.5 * human_steering_loss
                loss_dict['human_steering'] = human_steering_loss.item()
            
            # Rule teacher distillation
            if 'rule_lane_keeping' in targets:
                rule_lane_loss = self.mse_loss(
                    predictions['teacher']['rule']['lane_keeping'],
                    targets['rule_lane_keeping']
                )
                total_loss += 0.5 * rule_lane_loss
                loss_dict['rule_lane'] = rule_lane_loss.item()
        
        loss_dict['total'] = total_loss.item()
        return total_loss, loss_dict

def count_parameters(model):
    """Count total trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def test_hydra_mdp():
    """Test Hydra-MDP architecture"""
    print("=" * 60)
    print("NVIDIA HYDRA-MDP ARCHITECTURE TEST")
    print("=" * 60)
    
    # Create model
    model = HydraMDP()
    total_params = count_parameters(model)
    print(f"Model parameters: {total_params:,}")
    
    # Test input
    batch_size = 4
    test_input = torch.randn(batch_size, 3, 224, 224)
    
    print(f"Input shape: {test_input.shape}")
    
    # Test forward pass (training mode)
    print("\n1. Testing training mode:")
    model.train()
    with torch.no_grad():
        outputs_train = model(test_input, training=True)
    
    print("Training outputs:")
    for key, value in outputs_train.items():
        if key == 'teacher':
            print(f"  {key}:")
            for teacher_type, teacher_outputs in value.items():
                print(f"    {teacher_type}:")
                for output_name, output_tensor in teacher_outputs.items():
                    print(f"      {output_name}: {output_tensor.shape}")
        elif key == 'safety':
            print(f"  {key}:")
            for safety_name, safety_tensor in value.items():
                print(f"    {safety_name}: {safety_tensor.shape}")
        else:
            print(f"  {key}: {value.shape}")
    
    # Test forward pass (inference mode)
    print("\n2. Testing inference mode:")
    model.eval()
    with torch.no_grad():
        outputs_inference = model(test_input, training=False)
    
    print("Inference outputs:")
    for key, value in outputs_inference.items():
        if key == 'safety':
            print(f"  {key}:")
            for safety_name, safety_tensor in value.items():
                print(f"    {safety_name}: {safety_tensor.shape}")
        else:
            print(f"  {key}: {value.shape}")
    
    # Test loss function
    print("\n3. Testing loss function:")
    criterion = HydraMDPLoss()
    
    # Create dummy targets
    targets = {
        'steering': torch.randn(batch_size, 1),
        'throttle': torch.rand(batch_size, 1),
        'brake': torch.rand(batch_size, 1),
        'waypoints': torch.randn(batch_size, 10, 2),
        'speed': torch.rand(batch_size, 1) * 50,  # Speed in km/h
        'safety_score': torch.rand(batch_size, 1),
        'human_steering': torch.randn(batch_size, 1),
        'rule_lane_keeping': torch.randn(batch_size, 1)
    }
    
    loss, loss_dict = criterion(outputs_train, targets, training=True)
    print(f"Total loss: {loss.item():.4f}")
    print("Loss breakdown:")
    for loss_name, loss_value in loss_dict.items():
        print(f"  {loss_name}: {loss_value:.4f}")
    
    print("\n? Hydra-MDP architecture test completed successfully!")
    
    # Memory usage estimation
    model_size_mb = total_params * 4 / (1024 * 1024)  # Assuming float32
    print(f"\n? Model Statistics:")
    print(f"Parameters: {total_params:,}")
    print(f"Model size: {model_size_mb:.1f} MB")
    print(f"GPU memory (batch=4): ~{model_size_mb * 3:.1f} MB")

if __name__ == "__main__":
    test_hydra_mdp()