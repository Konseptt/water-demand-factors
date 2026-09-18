"""Feature-group ablation utilities."""
from .features import FEATURE_GROUPS

def feature_columns(drop_group=None):
    return [column for group, columns in FEATURE_GROUPS.items() if group != drop_group for column in columns]

def ablation_labels():
    return ["none"] + [f"drop_{group}" for group in FEATURE_GROUPS]
