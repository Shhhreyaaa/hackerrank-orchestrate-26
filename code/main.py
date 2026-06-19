import os
import sys
import json
import logging
import time
import pandas as pd
from typing import List, Dict, Optional
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from dotenv import load_dotenv
from tqdm import tqdm
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Clean environment variables to prevent Google SDK conflicts
if "GOOGLE_API_KEY" in os.environ:
    del os.environ["GOOGLE_API_KEY"]

# Add code directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rule_engine import apply_intelligent_rules
from vlm_analyzer import GeminiVLMAnalyzer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Load env
load_dotenv()

# Load system prompts from files
PROMPTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts")
DECISION_PROMPT_PATH = os.path.join(PROMPTS_DIR, "decision_system_prompt.txt")

try:
    with open(DECISION_PROMPT_PATH, "r", encoding="utf-8") as f:
        DECISION_SYSTEM_PROMPT = f.read()
except Exception as e:
    logger.error(f"Error loading decision system prompt: {e}")
    DECISION_SYSTEM_PROMPT = "You are a senior insurance claims decision engine."

# Pydantic schema for Step 2: Decision Maker
class DecisionAnalysisSchema(BaseModel):
    reasoning_steps: List[str] = Field(description="Reasoning steps analyzing visual reports and claim transcript.")
    evidence_standard_met: bool = Field(description="true if the submitted images meet the minimum evidence requirements to evaluate the claim; otherwise false")
    evidence_standard_met_reason: str = Field(description="A short explanation of why the evidence standard was or was not met.")
    visible_issue_type: str = Field(description="The visible issue type. Allowed: dent, scratch, crack, glass_shatter, broken_part, missing_part, torn_packaging, crushed_packaging, water_damage, stain, none, unknown")
    object_part: str = Field(description="The relevant object part. Choose the closest matching standard name.")
    claim_status: str = Field(description="Final decision. Allowed: supported, contradicted, not_enough_information")
    claim_status_justification: str = Field(description="Concise, image-grounded justification for the final decision.")
    supporting_image_ids: str = Field(description="Semicolon-separated list of image IDs (e.g. img_1;img_2) supporting the decision, or 'none'.")
    valid_image: bool = Field(description="true if the image set is valid and usable for automated review; false if wrong objects, non-original stock photos, or completely unreadable.")
    severity: str = Field(description="Estimated severity of the damage. Allowed: none, low, medium, high, unknown")
    detected_risk_flags: str = Field(description="Semicolon-separated list of risk flags. Use 'none' if no flags apply.")

class GeminiClaimVerifier:
    def __init__(self, cache_file: str = "code/image_cache.json"):
        # Ensure GOOGLE_API_KEY is unset to avoid SDK conflicts
        if "GOOGLE_API_KEY" in os.environ:
            del os.environ["GOOGLE_API_KEY"]
            
        self.client = genai.Client()
        self.vlm_analyzer = GeminiVLMAnalyzer(cache_file=cache_file)
        self.total_prompt_tokens = 0
        self.total_candidates_tokens = 0
        self.total_calls = 0

    def _call_reasoning_model(self, model: str, contents: list) -> str:
        response = self.client.models.generate_content(
            model=model,
            contents=contents,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=DecisionAnalysisSchema,
                temperature=0.1,
            )
        )
        if response.usage_metadata:
            self.total_prompt_tokens += response.usage_metadata.prompt_token_count or 0
            self.total_candidates_tokens += response.usage_metadata.candidates_token_count or 0
        self.total_calls += 1
        return response.text

    def _call_reasoning_with_fallback(self, contents: list) -> str:
        # Model rotation sequence on rate limits
        models_to_try = [
            "gemini-2.5-flash-lite",
            "models/gemini-3.1-flash-lite",
            "gemini-2.5-flash",
            "gemini-flash-latest"
        ]
        
        for attempt in range(1, 4):
            last_err = None
            for model in models_to_try:
                try:
                    logger.info(f"Attempting reasoning step using model: {model}")
                    result_text = self._call_reasoning_model(model, contents)
                    return result_text
                except Exception as e:
                    # Catch rate limits, resource exhausted, 503 unavailable, or transient server errors
                    err_str = str(e).upper()
                    if any(w in err_str for w in ["429", "RESOURCE_EXHAUSTED", "503", "UNAVAILABLE", "500", "INTERNAL"]):
                        logger.warning(f"Reasoning model {model} returned transient error ({e}). Trying next model...")
                        last_err = e
                        continue
                    else:
                        logger.error(f"Fatal error on model {model} in reasoning: {e}")
                        raise e
            
            # If all models returned errors, wait and retry the loop
            wait_time = attempt * 10
            logger.warning(f"All models returned transient errors in reasoning. Attempt {attempt}/3. Waiting {wait_time}s before retry...")
            time.sleep(wait_time)
            
        raise last_err


    def verify_claim(
        self,
        user_id: str,
        user_claim: str,
        claim_object: str,
        image_paths: List[str],
        user_history: dict,
        evidence_requirements: str,
        strategy: str = "enhanced"
    ) -> dict:
        loaded_image_ids = []
        visual_reports = {}
        
        # 1. Fetch visual descriptions using VLM Analyzer (cached)
        try:
            for raw_path in image_paths:
                resolved_p = self.vlm_analyzer.resolve_image_path(raw_path)
                if resolved_p:
                    img_id = os.path.splitext(os.path.basename(resolved_p))[0]
                    report = self.vlm_analyzer.get_image_description(resolved_p, img_id, claim_object)
                    visual_reports[img_id] = report
                    loaded_image_ids.append(img_id)
                else:
                    logger.warning(f"Could not resolve image: {raw_path}")
        except Exception as e:
            logger.error(f"Error during VLM visual inspection phase: {e}")
            return self._get_fallback_result(loaded_image_ids, user_history, f"VLM failure: {e}")

        # Fallback if no images are loaded/valid
        if not loaded_image_ids:
            return self._get_fallback_result(loaded_image_ids, user_history, "No valid or readable images were submitted with the claim.")

        # 2. Reasoning Decision Step (Text LLM call)
        history_summary = user_history.get("history_summary", "No history available.")
        history_flags = user_history.get("history_flags", "none")
        past_claims = user_history.get("past_claim_count", 0)
        rejected_claims = user_history.get("rejected_claim", 0)
        
        user_history_context = (
            f"User ID: {user_id}\n"
            f"Past claim count: {past_claims}\n"
            f"Rejected claims: {rejected_claims}\n"
            f"History summary: {history_summary}\n"
            f"History flags: {history_flags}"
        )
        
        prompt = (
            f"=== EVIDENCE REQUIREMENTS ===\n{evidence_requirements}\n\n"
            f"=== USER RISK HISTORY ===\n{user_history_context}\n\n"
            f"=== CLAIM CONVERSATION ===\n{user_claim}\n\n"
            f"=== IMAGE VISUAL REPORTS (TRUTH) ===\n"
            f"{json.dumps(visual_reports, indent=2)}\n\n"
            f"Please verify this claim for object '{claim_object}' and output your judgment in the required JSON schema."
        )

        try:
            response_text = self._call_reasoning_with_fallback([DECISION_SYSTEM_PROMPT, prompt])
            prediction = json.loads(response_text)
            
            # Synchronize accumulators from sub-analyzer for accurate telemetry
            self.total_calls += self.vlm_analyzer.total_calls
            self.total_prompt_tokens += self.vlm_analyzer.total_prompt_tokens
            self.total_candidates_tokens += self.vlm_analyzer.total_candidates_tokens
            
            # Reset sub-analyzer accumulators to avoid double-counting
            self.vlm_analyzer.total_calls = 0
            self.vlm_analyzer.total_prompt_tokens = 0
            self.vlm_analyzer.total_candidates_tokens = 0
            
            # Semicolon-split string mappings
            if isinstance(prediction.get("detected_risk_flags"), list):
                prediction["detected_risk_flags"] = ";".join(prediction["detected_risk_flags"])
            if isinstance(prediction.get("supporting_image_ids"), list):
                prediction["supporting_image_ids"] = ";".join(prediction["supporting_image_ids"])
                
            prediction["image_paths_list"] = loaded_image_ids
            
            # If strategy is baseline, skip the intelligent rule engine
            if strategy == "baseline":
                return self._sanitize_baseline_result(prediction, loaded_image_ids)
                
            # 3. Post-Processing Rules Step
            evidence_reqs_list = []
            for line in evidence_requirements.split("\n"):
                if line.startswith("- "):
                    parts = line[2:].split(": ", 1)
                    if len(parts) == 2:
                        evidence_reqs_list.append({
                            "requirement_id": "",
                            "applies_to": parts[0],
                            "minimum_image_evidence": parts[1]
                        })
            
            # Fallback loading requirements CSV directly
            if not evidence_reqs_list:
                try:
                    df_reqs = pd.read_csv("dataset/evidence_requirements.csv")
                    evidence_reqs_list = df_reqs.to_dict(orient="records")
                except Exception:
                    pass

            final_result = apply_intelligent_rules(
                vlm_output=prediction,
                user_history_row=user_history,
                evidence_reqs=evidence_reqs_list,
                claim_object=claim_object,
                user_claim=user_claim,
                visual_reports=visual_reports
            )
            return final_result

        except Exception as e:
            logger.error(f"Error in reasoning/post-processing decision: {e}")
            return self._get_fallback_result(loaded_image_ids, user_history, f"Reasoning failure: {e}")

    def _sanitize_baseline_result(self, result: dict, loaded_image_ids: List[str]) -> dict:
        evidence_standard_met_str = "true" if result.get("evidence_standard_met", True) else "false"
        valid_image_str = "true" if result.get("valid_image", True) else "false"
        
        risk_flags = result.get("detected_risk_flags", "none")
        if isinstance(risk_flags, list):
            risk_flags = ";".join(risk_flags)
            
        supporting_ids = result.get("supporting_image_ids", "none")
        if isinstance(supporting_ids, list):
            supporting_ids = ";".join(supporting_ids)
            
        return {
            "evidence_standard_met": evidence_standard_met_str,
            "evidence_standard_met_reason": result.get("evidence_standard_met_reason", ""),
            "risk_flags": risk_flags,
            "issue_type": result.get("visible_issue_type", "unknown"),
            "object_part": result.get("object_part", "unknown"),
            "claim_status": result.get("claim_status", "not_enough_information"),
            "claim_status_justification": result.get("claim_status_justification", ""),
            "supporting_image_ids": supporting_ids,
            "valid_image": valid_image_str,
            "severity": result.get("severity", "unknown")
        }

    # ============================================================
    # IMPROVEMENT 5: Robust Fallback output row
    # ============================================================
    def _get_fallback_result(self, loaded_image_ids: List[str], user_history: dict, reason: str) -> dict:
        history_flags = user_history.get("history_flags", "none")
        
        # User history risk merges if present in history_flags
        risk_flags_list = ["processing_error"]
        if "user_history_risk" in history_flags:
            risk_flags_list.append("user_history_risk")
        if "manual_review_required" in history_flags:
            risk_flags_list.append("manual_review_required")
            
        supporting_img = ";".join(loaded_image_ids) if loaded_image_ids else "none"

        return {
            "evidence_standard_met": "false",
            "evidence_standard_met_reason": f"Model execution failed. Reason: {reason}",
            "risk_flags": ";".join(risk_flags_list),
            "issue_type": "unknown",
            "object_part": "unknown",
            "claim_status": "not_enough_information",
            "claim_status_justification": "Robust fallback triggered due to VLM API error.",
            "supporting_image_ids": supporting_img,
            "valid_image": "false",
            "severity": "unknown"
        }

def run_predictions(input_csv: str, output_csv: str, strategy: str = "enhanced"):
    logger.info(f"Starting predictions run. Input: {input_csv}, Output: {output_csv}, Strategy: {strategy}")
    
    # Load all inputs once
    try:
        claims_df = pd.read_csv(input_csv)
        user_history_df = pd.read_csv("dataset/user_history.csv")
        evidence_req_df = pd.read_csv("dataset/evidence_requirements.csv")
    except Exception as e:
        logger.error(f"Error loading inputs: {e}")
        sys.exit(1)
        
    user_history_dict = user_history_df.set_index("user_id").to_dict(orient="index")
    verifier = GeminiClaimVerifier()
    
    results = []
    
    # Process row by row with tqdm progress bar
    for idx, row in tqdm(claims_df.iterrows(), total=len(claims_df), desc="Processing claims"):
        user_id = row["user_id"]
        image_paths_raw = row["image_paths"]
        user_claim = row["user_claim"]
        claim_object = row["claim_object"]
        
        # Retrieve user history
        user_history = user_history_dict.get(user_id, {
            "past_claim_count": 0,
            "accept_claim": 0,
            "manual_review_claim": 0,
            "rejected_claim": 0,
            "last_90_days_claim_count": 0,
            "history_flags": "none",
            "history_summary": "No history available."
        })
        
        # Retrieve evidence requirements
        obj_reqs = evidence_req_df[(evidence_req_df["claim_object"] == claim_object) | (evidence_req_df["claim_object"] == "all")]
        req_lines = []
        for _, req in obj_reqs.iterrows():
            req_lines.append(f"- {req['applies_to']}: {req['minimum_image_evidence']}")
        evidence_requirements_str = "\n".join(req_lines)
        
        image_paths_list = [p.strip() for p in str(image_paths_raw).split(";") if p.strip()]
        
        # Execute verification
        prediction = verifier.verify_claim(
            user_id=user_id,
            user_claim=user_claim,
            claim_object=claim_object,
            image_paths=image_paths_list,
            user_history=user_history,
            evidence_requirements=evidence_requirements_str,
            strategy=strategy
        )
        
        # Output record mapping
        result_row = {
            "user_id": user_id,
            "image_paths": image_paths_raw,
            "user_claim": user_claim,
            "claim_object": claim_object,
            "evidence_standard_met": prediction["evidence_standard_met"],
            "evidence_standard_met_reason": prediction["evidence_standard_met_reason"],
            "risk_flags": prediction["risk_flags"],
            "issue_type": prediction["issue_type"],
            "object_part": prediction["object_part"],
            "claim_status": prediction["claim_status"],
            "claim_status_justification": prediction["claim_status_justification"],
            "supporting_image_ids": prediction["supporting_image_ids"],
            "valid_image": prediction["valid_image"],
            "severity": prediction["severity"]
        }
        results.append(result_row)
        
        # Free tier limit safe sleeping (15 RPM)
        time.sleep(4.5)

    # Save to output file
    output_df = pd.DataFrame(results)
    
    # ============================================================
    # IMPROVEMENT 4: Hardcode the final column order exactly
    # ============================================================
    col_order = [
        "user_id", "image_paths", "user_claim", "claim_object",
        "evidence_standard_met", "evidence_standard_met_reason", "risk_flags",
        "issue_type", "object_part", "claim_status", "claim_status_justification",
        "supporting_image_ids", "valid_image", "severity"
    ]
    output_df = output_df[col_order]
    
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    output_df.to_csv(output_csv, index=False)
    logger.info(f"Saved {len(output_df)} predictions to {output_csv}")
    
    # Print summary statistics at the end
    print("\n" + "="*40)
    print("        PREDICTION SUMMARY STATISTICS        ")
    print("="*40)
    print(f"Total Claims Processed: {len(output_df)}")
    print("\nClaim Status Distribution:")
    print(output_df["claim_status"].value_counts().to_string())
    print("\nClaim Object Distribution:")
    print(output_df["claim_object"].value_counts().to_string())
    print("\nTop 5 Risk Flags:")
    all_flags = []
    for f_str in output_df["risk_flags"].dropna():
        all_flags.extend([f.strip() for f in f_str.split(";") if f.strip() and f.strip() != "none"])
    if all_flags:
        print(pd.Series(all_flags).value_counts().head(5).to_string())
    else:
        print("none detected")
    print("="*40 + "\n")

if __name__ == "__main__":
    run_predictions("dataset/claims.csv", "output.csv", "enhanced")
