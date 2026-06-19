import os
import base64
import pandas as pd
from PIL import Image
from typing import List, Dict, Optional

def load_claims(file_path: str = "dataset/claims.csv") -> pd.DataFrame:
    """Load the claims dataset."""
    return pd.read_csv(file_path)

def load_sample_claims(file_path: str = "dataset/sample_claims.csv") -> pd.DataFrame:
    """Load the sample labeled claims."""
    return pd.read_csv(file_path)

def load_user_history(file_path: str = "dataset/user_history.csv") -> pd.DataFrame:
    """Load the user claim history."""
    return pd.read_csv(file_path)

def load_evidence_requirements(file_path: str = "dataset/evidence_requirements.csv") -> pd.DataFrame:
    """Load the minimum evidence requirements."""
    return pd.read_csv(file_path)

def get_user_history(user_id: str, history_df: pd.DataFrame) -> Dict:
    """Retrieve history record for a specific user ID."""
    row = history_df[history_df["user_id"] == user_id]
    if not row.empty:
        return row.iloc[0].to_dict()
    return {
        "past_claim_count": 0,
        "accept_claim": 0,
        "manual_review_claim": 0,
        "rejected_claim": 0,
        "last_90_days_claim_count": 0,
        "history_flags": "none",
        "history_summary": "No history available."
    }

def get_evidence_requirements(claim_object: str, req_df: pd.DataFrame) -> List[Dict]:
    """Retrieve minimum evidence requirements for a claim object type."""
    filtered = req_df[(req_df["claim_object"] == claim_object) | (req_df["claim_object"] == "all")]
    return filtered.to_dict(orient="records")

def encode_image_to_base64(image_path: str) -> Optional[str]:
    """Encode a local image file to a base64 string."""
    try:
        if os.path.exists(image_path):
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode("utf-8")
    except Exception as e:
        print(f"Error encoding image to base64: {e}")
    return None

def encode_multiple_images(image_paths: List[str]) -> List[str]:
    """Encode multiple image files to base64 strings."""
    encoded = []
    for path in image_paths:
        base64_str = encode_image_to_base64(path)
        if base64_str:
            encoded.append(base64_str)
    return encoded
