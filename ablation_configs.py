"""
全アブレーションスタディのパターン設定
"""

# 全アブレーションパターンの設定
ABLATION_CONFIGS = {
    'moving_clap': {
        'fusion_type': 'cross_attn',
        'pooling_type': 'adaptive',
        'use_text_attention': True,
        'spatial_bias_value': 5.0,
        'hidden_dim': 768,
        'spatial_contrastive_weight': 0.01,
        'swap_weight': 0.3,
        'swap_loss_text_margin': 0.3,
        'swap_loss_audio_margin': 0.2,
        'description': 'MovingCLAP',
        'checkpoint_path': 'output_moving_clap/phase1/ckpt/last_model.pt'
    },
}


def get_config(pattern_name):
    """指定されたパターンの設定を取得"""
    if pattern_name not in ABLATION_CONFIGS:
        raise ValueError(f"Unknown pattern: {pattern_name}. Available: {list(ABLATION_CONFIGS.keys())}")
    config = ABLATION_CONFIGS[pattern_name].copy()
    return config


def get_all_patterns():
    """全パターン名のリストを取得"""
    return list(ABLATION_CONFIGS.keys())