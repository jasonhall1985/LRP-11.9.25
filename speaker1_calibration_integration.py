
# 🎯 SPEAKER 1 CALIBRATION DATA - Generated 2025-09-24T16:47:20.033508
# Training accuracy: 0.550
# Total samples: 80

SPEAKER1_CALIBRATION_DATA = {
    'doctor': {
        'bias': -0.5000,
        'target_logit': -0.6364,
        'sample_count': 20
    },
    'i_need_to_move': {
        'bias': 0.5000,
        'target_logit': 1.0071,
        'sample_count': 20
    },
    'my_mouth_is_dry': {
        'bias': 0.5000,
        'target_logit': 1.0626,
        'sample_count': 20
    },
    'pillow': {
        'bias': -0.2019,
        'target_logit': -0.1625,
        'sample_count': 20
    },
}

def apply_speaker1_calibration(raw_logits):
    """Apply Speaker 1 specific calibration to raw model logits."""
    calibrated_logits = raw_logits.copy()

    class_names = ['doctor', 'i_need_to_move', 'my_mouth_is_dry', 'pillow']
    for i, class_name in enumerate(class_names):
        if class_name in SPEAKER1_CALIBRATION_DATA:
            bias = SPEAKER1_CALIBRATION_DATA[class_name]['bias']
            calibrated_logits[i] += bias

    return calibrated_logits
