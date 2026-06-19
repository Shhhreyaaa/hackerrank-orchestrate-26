from PIL import TiffImagePlugin
from PIL import TiffImagePlugin
from PIL import TiffImagePlugin
import os
import sys
import json
import hashlib
import logging
from PIL import Image
from typing import List, Dict, Optional
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from dotenv import load_dotenv
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Load dotenv
load_dotenv()

# Prevent conflicts with valid GEMINI_API_KEY
if os.environ.get("GOOGLE_API_KEY") == "your_key_here":
    del os.environ["GOOGLE_API_KEY"]

# Load system prompt
PROMPTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts")
VLM_PROMPT_PATH = os.path.join(PROMPTS_DIR, "vlm_system_prompt.txt")

try:
    with open(VLM_PROMPT_PATH, "r", encoding="utf-8") as f:
        VLM_SYSTEM_PROMPT = f.read()
except Exception as e:
    logger.error(f"Error loading system prompt in vlm_analyzer: {e}")
    VLM_SYSTEM_PROMPT = "You are an expert insurance visual claim inspector."


# Pydantic schema for single image description
class VisualIssue(BaseModel):
    issue_type: str = Field(
        description="The visible issue type: dent, scratch, crack, glass_shatter, broken_part, missing_part, torn_packaging, crushed_packaging, water_damage, stain, none"
    )
    object_part: str = Field(
        description="The relevant object part, e.g. windshield, bumper, keyboard, screen, box, seal, item, etc."
    )
    severity: str = Field(
        description="Estimated severity: low, medium, high, none, unknown"
    )
    justification: str = Field(description="1-sentence visual justification.")


class ImageVisualReportSchema(BaseModel):
    object_type: str = Field(
        description="Object type visible: car, laptop, package, other"
    )
    parts_visible: List[str] = Field(description="List of all visible object parts.")
    visible_issues: List[VisualIssue] = Field(description="List of detected damages.")
    quality_flags: List[str] = Field(
        description="Quality issues: blurry_image, cropped_or_obstructed, low_light_or_glare, wrong_angle"
    )
    trust_flags: List[str] = Field(
        description="Trust issues: non_original_image, possible_manipulation, text_instruction_present"
    )
    text_overlays: List[str] = Field(description="Any text visible inside the image.")
    detailed_description: str = Field(
        description="Detailed description of the image content."
    )


class GeminiVLMAnalyzer:
    def __init__(self, cache_file: str = "code/image_cache.json"):
        # Explicitly remove GOOGLE_API_KEY from environment to avoid SDK conflict
        if "GOOGLE_API_KEY" in os.environ:
            del os.environ["GOOGLE_API_KEY"]

        self.client = genai.Client()
        self.cache_file = os.path.abspath(cache_file)
        self.image_cache = self._load_cache()
        self.total_prompt_tokens = 0
        self.total_candidates_tokens = 0
        self.total_calls = 0

    def resolve_image_path(self, path: str) -> Optional[str]:
        path = path.strip()
        if not path:
            return None
        search_dirs = [
            ".",
            "dataset",
            "dataset/images",
            "dataset/images/sample",
            "dataset/images/test",
        ]
        if os.path.exists(path):
            return os.path.abspath(path)
        for d in search_dirs:
            p = os.path.join(d, path)
            if os.path.exists(p):
                return os.path.abspath(p)
            if path.startswith("images/"):
                p = os.path.join(d, path[7:])
                if os.path.exists(p):
                    return os.path.abspath(p)
        return None

    def _load_cache(self) -> Dict:
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    logger.info(
                        f"VLM Analyzer loaded image cache from {self.cache_file}"
                    )
                    return json.load(f)
            except Exception as e:
                logger.error(f"VLM Analyzer failed to load image cache: {e}")
        return {}

    def _save_cache(self):
        try:
            os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(self.image_cache, f, indent=2)
        except Exception as e:
            logger.error(f"VLM Analyzer failed to save image cache: {e}")

    def _get_image_md5(self, path: str) -> str:
        hasher = hashlib.md5()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def _call_vlm_model(self, model: str, contents: list) -> str:
        response = self.client.models.generate_content(
            model=model,
            contents=contents,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=ImageVisualReportSchema,
                temperature=0.1,
            ),
        )
        if response.usage_metadata:
            self.total_prompt_tokens += response.usage_metadata.prompt_token_count or 0
            self.total_candidates_tokens += (
                response.usage_metadata.candidates_token_count or 0
            )
        self.total_calls += 1
        return response.text

    def _call_vlm_with_fallback(self, contents: list) -> str:
        # Dynamic rotation models sequence to handle API quota caps
        models_to_try = [
            "gemini-2.5-flash-lite",
            "models/gemini-3.1-flash-lite",
            "gemini-2.5-flash",
            "gemini-flash-latest",
        ]

        for attempt in range(1, 4):
            last_err = None
            for model in models_to_try:
                try:
                    logger.info(f"Attempting visual analysis using model: {model}")
                    result_text = self._call_vlm_model(model, contents)
                    return result_text
                except Exception as e:
                    # Catch rate limits, resource exhausted, 503 unavailable, or transient server errors
                    err_str = str(e).upper()
                    if any(
                        w in err_str
                        for w in [
                            "429",
                            "RESOURCE_EXHAUSTED",
                            "503",
                            "UNAVAILABLE",
                            "500",
                            "INTERNAL",
                        ]
                    ):
                        logger.warning(
                            f"Model {model} returned transient error ({e}). Trying next model..."
                        )
                        last_err = e
                        continue
                    else:
                        logger.error(f"Fatal error on model {model}: {e}")
                        raise e

            # If all models returned errors, wait and retry the loop
            wait_time = attempt * 10
            logger.warning(
                f"All models returned transient errors. Attempt {attempt}/3. Waiting {wait_time}s before retry..."
            )
            time.sleep(wait_time)

        # If all fallback models fail, raise the last encountered error
        raise last_err

    def get_image_description(
        self, resolved_path: str, img_id: str, claim_object: str
    ) -> dict:
        if not os.path.exists(resolved_path):
            raise FileNotFoundError(f"Image not found at {resolved_path}")

        md5_hash = self._get_image_md5(resolved_path)
        if md5_hash in self.image_cache:
            logger.info(f"Cache HIT for image ID: {img_id} (MD5: {md5_hash})")
            return self.image_cache[md5_hash]

        logger.info(
            f"Cache MISS for image ID: {img_id} (MD5: {md5_hash}). Requesting Gemini inspection..."
        )
        try:
            img = Image.open(resolved_path)
            img.thumbnail((1024, 1024))

            prompt = (
                f"Please inspect this image. The claimed object type is '{claim_object}'. "
                f"Identify the object type, parts, quality issues, text, and damages."
            )

            contents = [VLM_SYSTEM_PROMPT, prompt, img]
            response_text = self._call_vlm_with_fallback(contents)

            visual_report = json.loads(response_text)

            # Save visual report to cache
            self.image_cache[md5_hash] = visual_report
            self._save_cache()
            return visual_report

        except Exception as e:
            logger.error(f"VLM Analyzer failed to process image {resolved_path}: {e}")
            # Return error-safe fallback report
            return {
                "object_type": "other",
                "parts_visible": ["unknown"],
                "visible_issues": [],
                "quality_flags": ["blurry_image"],
                "trust_flags": ["non_original_image"],
                "text_overlays": [],
                "detailed_description": f"Inspection failed: {e}",
            }


def analyze_images_with_vlm(
    image_paths: str,
    user_claim: str,
    claim_object: str,
    user_history: Dict,
    evidence_requirements: str,
) -> Dict:
    """Wrapper function matching legacy calls if needed."""
    analyzer = GeminiVLMAnalyzer()

    # Resolve and parse paths
    paths_list = [p.strip() for p in str(image_paths).split(";") if p.strip()]
    reports = {}
    loaded_ids = []

    search_dirs = [
        ".",
        "dataset",
        "dataset/images",
        "dataset/images/sample",
        "dataset/images/test",
    ]
    for path in paths_list:
        resolved = None
        for d in search_dirs:
            p = os.path.join(d, path)
            if os.path.exists(p):
                resolved = os.path.abspath(p)
                break
            if path.startswith("images/"):
                p = os.path.join(d, path[7:])
                if os.path.exists(p):
                    resolved = os.path.abspath(p)
                    break

        if resolved:
            img_id = os.path.splitext(os.path.basename(resolved))[0]
            report = analyzer.get_image_description(resolved, img_id, claim_object)
            reports[img_id] = report
            loaded_ids.append(img_id)

    return {
        "visual_reports": reports,
        "loaded_image_ids": loaded_ids,
        "total_calls": analyzer.total_calls,
        "prompt_tokens": analyzer.total_prompt_tokens,
        "candidates_tokens": analyzer.total_candidates_tokens,
    }
