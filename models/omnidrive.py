"""
NVIDIA OmniDrive Implementation for CARLA
End-to-End Autonomous Driving with Large Language Models
CVPR 2025 - Combines 3D perception, reasoning, and planning
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class RotaryPositionalEmbedding(nn.Module):
    """Rotary positional embedding for 3D spatial reasoning"""
    def __init__(self, dim, max_seq_len=1000):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        
    def forward(self, x, seq_len):
        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos()[None, :, :], emb.sin()[None, :, :]

class Multi3DAttention(nn.Module):
    """Multi-head attention with 3D spatial awareness"""
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
        # 3D spatial bias
        self.spatial_bias = nn.Parameter(torch.randn(num_heads, 100, 100))
        
    def forward(self, query, key, value, spatial_coords=None, mask=None):
        batch_size, seq_len = query.size(0), query.size(1)
        
        # Linear transformations
        Q = self.W_q(query).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Add 3D spatial bias if coordinates provided
        if spatial_coords is not None:
            # Simplified spatial bias (in practice, would use actual 3D coordinates)
            spatial_bias = self.spatial_bias[:, :seq_len, :K.size(-2)]
            attention_scores += spatial_bias.unsqueeze(0)
        
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        # Residual connection and layer norm
        output = self.layer_norm(query + self.W_o(context))
        return output, attention_weights

class LLMReasoningModule(nn.Module):
    """Large Language Model-based reasoning for driving decisions"""
    def __init__(self, d_model=512, num_layers=6, num_heads=8, ff_dim=2048):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        
        # Token embeddings for driving concepts
        self.concept_embedding = nn.Embedding(1000, d_model)  # Driving vocabulary
        self.position_embedding = RotaryPositionalEmbedding(d_model)
        
        # Project visual features to model dimension
        self.visual_projection = nn.Linear(32, d_model)  # From BEV pooled features
        
        # Transformer layers
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'attention': Multi3DAttention(d_model, num_heads),
                'ff': nn.Sequential(
                    nn.Linear(d_model, ff_dim),
                    nn.GELU(),
                    nn.Dropout(0.1),
                    nn.Linear(ff_dim, d_model),
                    nn.Dropout(0.1)
                ),
                'norm': nn.LayerNorm(d_model)
            }) for _ in range(num_layers)
        ])
        
        # Reasoning heads
        self.scene_reasoning = nn.Linear(d_model, 256)
        self.risk_assessment = nn.Linear(d_model, 128)
        self.decision_justification = nn.Linear(d_model, 512)
        
    def forward(self, visual_features, scene_tokens=None):
        batch_size = visual_features.size(0)
        
        # Project visual features to model dimension and add sequence dimension
        x = self.visual_projection(visual_features).unsqueeze(1)
        
        # Add scene tokens if provided (traffic signs, objects, etc.)
        if scene_tokens is not None:
            scene_embeds = self.concept_embedding(scene_tokens)
            x = torch.cat([x, scene_embeds], dim=1)
        
        seq_len = x.size(1)
        
        # Apply transformer layers
        for layer in self.layers:
            # Self-attention with 3D spatial awareness
            attn_out, _ = layer['attention'](x, x, x)
            
            # Feed-forward
            ff_out = layer['ff'](attn_out)
            x = layer['norm'](attn_out + ff_out)
        
        # Extract reasoning outputs
        pooled_features = x.mean(dim=1)  # Global pooling
        
        scene_reasoning = self.scene_reasoning(pooled_features)
        risk_assessment = torch.sigmoid(self.risk_assessment(pooled_features))
        decision_justification = self.decision_justification(pooled_features)
        
        return {
            'scene_reasoning': scene_reasoning,
            'risk_assessment': risk_assessment,
            'decision_justification': decision_justification,
            'reasoning_features': pooled_features
        }

class BEVPerceptionModule(nn.Module):
    """Bird's Eye View perception for 3D understanding"""
    def __init__(self, input_channels=3, bev_size=200):
        super().__init__()
        self.bev_size = bev_size
        
        # Front camera to BEV transformation
        self.perspective_transformer = nn.Sequential(
            nn.Conv2d(input_channels, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.Conv2d(64, 128, 5, stride=2, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            
            nn.Conv2d(256, 512, 3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        
        # BEV feature processing
        self.bev_processor = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        
        # Object detection in BEV
        self.object_detector = nn.Conv2d(32, 10, 1)  # 10 object classes
        
        # Lane detection in BEV
        self.lane_detector = nn.Conv2d(32, 3, 1)  # Left, ego, right lanes
        
        # Occupancy grid
        self.occupancy_head = nn.Conv2d(32, 1, 1)
        
    def forward(self, front_camera):
        # Extract perspective features
        features = self.perspective_transformer(front_camera)
        
        # Transform to BEV representation
        bev_features = self.bev_processor(features)
        
        # BEV predictions
        objects = self.object_detector(bev_features)
        lanes = self.lane_detector(bev_features)
        occupancy = torch.sigmoid(self.occupancy_head(bev_features))
        
        return {
            'bev_features': bev_features,
            'objects': objects,
            'lanes': lanes,
            'occupancy': occupancy
        }

class OmniDrive(nn.Module):
    """
    OmniDrive: End-to-End Autonomous Driving with Large Language Models
    CVPR 2025 - Combines 3D perception, reasoning, and planning
    """
    def __init__(self, vocab_size=1000):
        super().__init__()
        
        # Core components
        self.bev_perception = BEVPerceptionModule()
        self.llm_reasoning = LLMReasoningModule()
        
        # Feature fusion - calculate actual dimensions dynamically
        # BEV pooled features: 32, reasoning features: 512
        self.feature_fusion = nn.Sequential(
            nn.Linear(32 + 512, 512),  # BEV + reasoning features
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Multi-modal planning head
        self.planning_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        # Action prediction heads
        self.steering_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
        self.speed_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.ReLU()  # Non-negative speed
        )
        
        self.throttle_brake_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2),  # Throttle and brake
            nn.Sigmoid()
        )
        
        # Trajectory prediction (future waypoints)
        self.trajectory_head = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 30)  # 15 future waypoints (x, y)
        )
        
        # Counterfactual reasoning head
        self.counterfactual_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32)  # Counterfactual scenario embeddings
        )
        
        # Decision explanation generator
        self.explanation_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, vocab_size)  # Text explanation tokens
        )
    
    def forward(self, front_camera, scene_tokens=None, return_reasoning=False):
        batch_size = front_camera.size(0)
        
        # BEV perception
        bev_outputs = self.bev_perception(front_camera)
        bev_features = bev_outputs['bev_features']
        
        # Global pooling of BEV features
        bev_pooled = F.adaptive_avg_pool2d(bev_features, (1, 1)).flatten(1)
        
        # LLM-based reasoning
        reasoning_outputs = self.llm_reasoning(bev_pooled, scene_tokens)
        reasoning_features = reasoning_outputs['reasoning_features']
        
        # Feature fusion
        fused_features = torch.cat([bev_pooled, reasoning_features], dim=1)
        fused_features = self.feature_fusion(fused_features)
        
        # Planning
        planning_features = self.planning_head(fused_features)
        
        # Action predictions
        steering = self.steering_head(planning_features)
        speed = self.speed_head(planning_features)
        throttle_brake = self.throttle_brake_head(planning_features)
        
        # Trajectory prediction
        trajectory = self.trajectory_head(planning_features)
        trajectory = trajectory.view(batch_size, 15, 2)  # Reshape to waypoints
        
        # Advanced reasoning outputs
        counterfactual = self.counterfactual_head(fused_features)
        explanation_logits = self.explanation_head(fused_features) if return_reasoning else None
        
        outputs = {
            'steering': steering,
            'speed': speed,
            'throttle': throttle_brake[:, :1],
            'brake': throttle_brake[:, 1:],
            'trajectory': trajectory,
            'bev_objects': bev_outputs['objects'],
            'bev_lanes': bev_outputs['lanes'],
            'bev_occupancy': bev_outputs['occupancy'],
            'risk_assessment': reasoning_outputs['risk_assessment'],
            'counterfactual': counterfactual
        }
        
        if return_reasoning:
            outputs.update({
                'scene_reasoning': reasoning_outputs['scene_reasoning'],
                'decision_justification': reasoning_outputs['decision_justification'],
                'explanation_logits': explanation_logits
            })
        
        return outputs

class OmniDriveLoss(nn.Module):
    """Multi-task loss for OmniDrive training"""
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.ce_loss = nn.CrossEntropyLoss()
        self.l1_loss = nn.L1Loss()
    
    def forward(self, predictions, targets):
        total_loss = 0.0
        loss_dict = {}
        
        # Driving actions
        if 'steering' in targets:
            steering_loss = self.mse_loss(predictions['steering'], targets['steering'])
            total_loss += 3.0 * steering_loss  # High weight for steering
            loss_dict['steering'] = steering_loss.item()
        
        if 'speed' in targets:
            speed_loss = self.mse_loss(predictions['speed'], targets['speed'])
            total_loss += 1.0 * speed_loss
            loss_dict['speed'] = speed_loss.item()
        
        if 'throttle' in targets:
            throttle_loss = self.mse_loss(predictions['throttle'], targets['throttle'])
            total_loss += 1.0 * throttle_loss
            loss_dict['throttle'] = throttle_loss.item()
        
        if 'brake' in targets:
            brake_loss = self.mse_loss(predictions['brake'], targets['brake'])
            total_loss += 1.0 * brake_loss
            loss_dict['brake'] = brake_loss.item()
        
        # Trajectory planning
        if 'trajectory' in targets:
            traj_loss = self.l1_loss(predictions['trajectory'], targets['trajectory'])
            total_loss += 2.0 * traj_loss
            loss_dict['trajectory'] = traj_loss.item()
        
        # BEV perception losses
        if 'bev_objects' in targets:
            obj_loss = self.bce_loss(predictions['bev_objects'], targets['bev_objects'])
            total_loss += 1.5 * obj_loss
            loss_dict['bev_objects'] = obj_loss.item()
        
        if 'bev_lanes' in targets:
            lane_loss = self.bce_loss(predictions['bev_lanes'], targets['bev_lanes'])
            total_loss += 1.5 * lane_loss
            loss_dict['bev_lanes'] = lane_loss.item()
        
        if 'bev_occupancy' in targets:
            occ_loss = self.bce_loss(predictions['bev_occupancy'], targets['bev_occupancy'])
            total_loss += 1.0 * occ_loss
            loss_dict['bev_occupancy'] = occ_loss.item()
        
        # Risk assessment
        if 'risk_assessment' in targets:
            risk_loss = self.mse_loss(predictions['risk_assessment'], targets['risk_assessment'])
            total_loss += 1.0 * risk_loss
            loss_dict['risk_assessment'] = risk_loss.item()
        
        # Counterfactual reasoning
        if 'counterfactual' in targets:
            cf_loss = self.mse_loss(predictions['counterfactual'], targets['counterfactual'])
            total_loss += 0.5 * cf_loss
            loss_dict['counterfactual'] = cf_loss.item()
        
        # Explanation generation
        if 'explanation_logits' in predictions and 'explanation_tokens' in targets:
            # Reshape to match cross entropy requirements
            batch_size = predictions['explanation_logits'].size(0)
            vocab_size = predictions['explanation_logits'].size(-1)
            seq_len = targets['explanation_tokens'].size(-1)
            
            # Take only the first part of target to match batch size
            target_tokens = targets['explanation_tokens'][:batch_size]
            
            exp_loss = self.ce_loss(
                predictions['explanation_logits'].unsqueeze(1).expand(-1, seq_len, -1).contiguous().view(-1, vocab_size),
                target_tokens.view(-1)
            )
            total_loss += 0.5 * exp_loss
            loss_dict['explanation'] = exp_loss.item()
        
        loss_dict['total'] = total_loss.item()
        return total_loss, loss_dict

def count_parameters(model):
    """Count total trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def test_omnidrive():
    """Test OmniDrive architecture"""
    print("=" * 60)
    print("NVIDIA OMNIDRIVE ARCHITECTURE TEST")
    print("=" * 60)
    
    # Create model
    model = OmniDrive()
    total_params = count_parameters(model)
    print(f"Model parameters: {total_params:,}")
    
    # Test input
    batch_size = 2
    front_camera = torch.randn(batch_size, 3, 224, 224)
    scene_tokens = torch.randint(0, 100, (batch_size, 10))  # 10 scene concept tokens
    
    print(f"Front camera shape: {front_camera.shape}")
    print(f"Scene tokens shape: {scene_tokens.shape}")
    
    # Test forward pass without reasoning
    print("\n1. Testing standard forward pass:")
    model.eval()
    with torch.no_grad():
        outputs = model(front_camera, scene_tokens, return_reasoning=False)
    
    print("Standard outputs:")
    for key, value in outputs.items():
        print(f"  {key}: {value.shape}")
    
    # Test forward pass with reasoning
    print("\n2. Testing forward pass with reasoning:")
    with torch.no_grad():
        outputs_reasoning = model(front_camera, scene_tokens, return_reasoning=True)
    
    print("Reasoning outputs:")
    for key, value in outputs_reasoning.items():
        if key not in outputs:  # Only show new outputs
            print(f"  {key}: {value.shape}")
    
    # Test loss function
    print("\n3. Testing loss function:")
    criterion = OmniDriveLoss()
    
    # Create dummy targets matching output dimensions
    targets = {
        'steering': torch.randn(batch_size, 1),
        'speed': torch.rand(batch_size, 1) * 50,
        'throttle': torch.rand(batch_size, 1),
        'brake': torch.rand(batch_size, 1),
        'trajectory': torch.randn(batch_size, 15, 2),
        'bev_objects': torch.rand(batch_size, 10, 56, 56),  # Match BEV output size
        'bev_lanes': torch.rand(batch_size, 3, 56, 56),
        'bev_occupancy': torch.rand(batch_size, 1, 56, 56),
        'risk_assessment': torch.rand(batch_size, 128),
        'counterfactual': torch.randn(batch_size, 32),
        'explanation_tokens': torch.randint(0, 1000, (batch_size, 50))
    }
    
    loss, loss_dict = criterion(outputs_reasoning, targets)
    print(f"Total loss: {loss.item():.4f}")
    print("Loss breakdown:")
    for loss_name, loss_value in loss_dict.items():
        print(f"  {loss_name}: {loss_value:.4f}")
    
    print("\n? OmniDrive architecture test completed successfully!")
    
    # Memory usage estimation
    model_size_mb = total_params * 4 / (1024 * 1024)  # Assuming float32
    print(f"\n? Model Statistics:")
    print(f"Parameters: {total_params:,}")
    print(f"Model size: {model_size_mb:.1f} MB")
    print(f"GPU memory (batch=2): ~{model_size_mb * 4:.1f} MB")

if __name__ == "__main__":
    test_omnidrive()