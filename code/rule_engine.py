import re
import logging
from typing import Dict, List, Set

logger = logging.getLogger(__name__)

# Allowed categories and parts definition
ALLOWED_CLAIM_STATUS = {'supported', 'contradicted', 'not_enough_information'}
ALLOWED_ISSUE_TYPE = {
    'dent', 'scratch', 'crack', 'glass_shatter', 'broken_part', 'missing_part',
    'torn_packaging', 'crushed_packaging', 'water_damage', 'stain', 'none', 'unknown'
}
ALLOWED_SEVERITY = {'none', 'low', 'medium', 'high', 'unknown'}
ALLOWED_RISK_FLAGS = {
    'none', 'blurry_image', 'cropped_or_obstructed', 'low_light_or_glare', 'wrong_angle',
    'wrong_object', 'wrong_object_part', 'damage_not_visible', 'claim_mismatch',
    'possible_manipulation', 'non_original_image', 'text_instruction_present',
    'user_history_risk', 'manual_review_required'
}

ALLOWED_CAR_PARTS = {
    'front_bumper', 'rear_bumper', 'door', 'hood', 'windshield', 'side_mirror',
    'headlight', 'taillight', 'fender', 'quarter_panel', 'body', 'unknown'
}
ALLOWED_LAPTOP_PARTS = {
    'screen', 'keyboard', 'trackpad', 'hinge', 'lid', 'corner', 'port', 'base', 'body', 'unknown'
}
ALLOWED_PACKAGE_PARTS = {
    'box', 'package_corner', 'package_side', 'seal', 'label', 'contents', 'item', 'unknown'
}

def has_word(text: str, words: List[str]) -> bool:
    for w in words:
        if re.search(r'\b' + re.escape(w) + r'\b', text):
            return True
    return False

def is_negated_word(text: str, word: str) -> bool:
    pattern = r'(not|no|nahi|neither|never)\s+(?:[\w\s]{0,10}\s+)?' + re.escape(word)
    if re.search(pattern, text, re.IGNORECASE):
        return True
    pattern2 = re.escape(word) + r'\s+(?:[\w\s]{0,10}\s+)?(nahi|not)'
    if re.search(pattern2, text, re.IGNORECASE):
        return True
    return False

def check_is_missing_claim(customer_texts: List[str]) -> bool:
    for text in customer_texts:
        text_lower = text.lower()
        if any(w in text_lower for w in ["missing", "not inside", "empty", "receive nahi", "receive nahe", "mila nahi", "not in the box", "not in box"]):
            # Check for negative context
            negations = [
                "not claiming", "don't claim", "do not claim", "no missing",
                "missing nahi", "claim nahi", "report nahi", "review nahi",
                "not reporting missing", "only reporting packaging", "only packaging",
                "only review packaging"
            ]
            if any(neg in text_lower for neg in negations):
                continue
            return True
    return False

def is_flag_in_all_images(flag_name: str, reports: Dict) -> bool:
    if not reports:
        return False
    return all(flag_name in [f.lower() for f in rep.get("quality_flags", []) + rep.get("trust_flags", [])] for rep in reports.values())

def normalize_part(part: str, claim_object: str, user_claim: str) -> str:
    part = part.lower().strip()
    user_claim_lower = user_claim.lower()
    
    if claim_object == "car":
        if "bumper" in part:
            if "front" in part or "front" in user_claim_lower:
                return "front_bumper"
            if "rear" in part or "back" in user_claim_lower or "rear" in user_claim_lower:
                return "rear_bumper"
            return "rear_bumper"
        if "door" in part:
            return "door"
        if "mirror" in part:
            return "side_mirror"
        if "light" in part or "lamp" in part:
            if "head" in part or "head" in user_claim_lower:
                return "headlight"
            if "tail" in part or "back" in user_claim_lower or "rear" in user_claim_lower:
                return "taillight"
            return "headlight"
        if "windshield" in part or "glass" in part:
            return "windshield"
        if "hood" in part or "bonnet" in part:
            return "hood"
        if "fender" in part:
            return "fender"
        if "quarter" in part:
            return "quarter_panel"
        if part in ALLOWED_CAR_PARTS:
            return part
            
    elif claim_object == "laptop":
        if "screen" in part or "display" in part or "bezel" in part or "panel" in part:
            if "hinge" in user_claim_lower:
                return "hinge"
            return "screen"
        if "keyboard" in part or "key" in part:
            return "keyboard"
        if "trackpad" in part or "touchpad" in part:
            return "trackpad"
        if "hinge" in part:
            return "hinge"
        if "lid" in part or "shell" in part or "cover" in part:
            return "lid"
        if "corner" in part or "edge" in part:
            return "corner"
        if "port" in part or "usb" in part:
            return "port"
        if "base" in part or "bottom" in part:
            return "base"
        if "body" in part or "chassis" in part:
            return "body"
        if part in ALLOWED_LAPTOP_PARTS:
            return part
            
    elif claim_object == "package":
        if "corner" in part:
            return "package_corner"
        if "side" in part or "panel" in part:
            return "package_side"
        if "seal" in part or "tape" in part or "flap" in part or "glue" in part:
            return "seal"
        if "label" in part or "sticker" in part:
            return "label"
        if "contents" in part or "item" in part or "inside" in part:
            return "contents"
        if "box" in part or "package" in part or "carton" in part:
            return "box"
        if part in ALLOWED_PACKAGE_PARTS:
            return part
            
    return "unknown"

def apply_intelligent_rules(
    vlm_output: Dict,
    user_history_row: Dict,
    evidence_reqs: List[Dict],
    claim_object: str,
    user_claim: str,
    visual_reports: Dict = None
) -> Dict:
    reports = visual_reports if visual_reports else {}
    
    # 1. Parse initial values from VLM output
    evidence_standard_met_raw = vlm_output.get("evidence_standard_met", True)
    if isinstance(evidence_standard_met_raw, str):
        evidence_standard_met = evidence_standard_met_raw.strip().lower() == "true"
    else:
        evidence_standard_met = bool(evidence_standard_met_raw)

    evidence_standard_met_reason = vlm_output.get("evidence_standard_met_reason", "")
    visible_issue_type = str(vlm_output.get("visible_issue_type", "unknown")).strip().lower()
    object_part = str(vlm_output.get("object_part", "unknown")).strip().lower()
    claim_status = str(vlm_output.get("claim_status", "not_enough_information")).strip().lower()
    claim_status_justification = vlm_output.get("claim_status_justification", "")
    supporting_image_ids = str(vlm_output.get("supporting_image_ids", "none")).strip()
    
    valid_image_raw = vlm_output.get("valid_image", True)
    if isinstance(valid_image_raw, str):
        valid_image = valid_image_raw.strip().lower() == "true"
    else:
        valid_image = bool(valid_image_raw)
        
    severity = str(vlm_output.get("severity", "unknown")).strip().lower()
    
    # Parse risk flags
    risk_flags_raw = vlm_output.get("detected_risk_flags", "none")
    if isinstance(risk_flags_raw, list):
        risk_flags = [str(f).strip().lower() for f in risk_flags_raw if str(f).strip()]
    else:
        risk_flags = [f.strip().lower() for f in str(risk_flags_raw).split(";") if f.strip()]
        
    # Propagate flags from visual reports directly
    for img_id, rep in reports.items():
        for flag in rep.get("quality_flags", []):
            risk_flags.append(str(flag).strip().lower())
        for flag in rep.get("trust_flags", []):
            risk_flags.append(str(flag).strip().lower())
            
        # Programmatic wrong object detection
        parts_v = [str(p).lower() for p in rep.get("parts_visible", [])]
        if claim_object == "package" and any(p in parts_v for p in ["can_body", "rim", "can"]):
            if "wrong_object" not in risk_flags:
                risk_flags.append("wrong_object")
                
    if not risk_flags:
        risk_flags = ["none"]

    # Normalize part
    object_part = normalize_part(object_part, claim_object, user_claim)

    # Normalize visible_issue_type
    if visible_issue_type not in ALLOWED_ISSUE_TYPE:
        visible_issue_type = "unknown"

    # ============================================================
    # IMPROVEMENT 1: Regex-based Adversarial Instruction Detection
    # ============================================================
    adversarial_patterns = [
        r"approve\s+immediately",
        r"skip\s+review",
        r"mark\s+(this\s+row\s+)?supported",
        r"ignore\s+previous",
        r"ignore\s+checks",
        r"follow\s+the\s+note",
        r"follow\s+note",
        r"follow\s+instruction",
        r"bypass\s+review",
        r"override\s+check",
        r"ignore\s+visual",
        r"skip\s+verification",
        r"system\s+instruction"
    ]
    user_claim_lower = user_claim.lower()
    text_instruction_detected = False
    for pat in adversarial_patterns:
        if re.search(pat, user_claim_lower):
            text_instruction_detected = True
            break
            
    # Also check if text overlay risk was flagged by VLM
    if "text_instruction_present" in risk_flags:
        text_instruction_detected = True

    if text_instruction_detected:
        if "text_instruction_present" not in risk_flags:
            risk_flags.append("text_instruction_present")
        if "manual_review_required" not in risk_flags:
            risk_flags.append("manual_review_required")
        # If user tried adversarial prompt injection, override status to contradicted (strict safety)
        if claim_status == "supported":
            claim_status = "contradicted"
            claim_status_justification = "Claim rejected due to adversarial instruction patterns detected in the input claim conversation or image."

    # ============================================================
    # IMPROVEMENT 2: Robust Claim vs Visual Mismatch Detection
    # ============================================================
    customer_texts = []
    for turn in re.split(r'\||\n', user_claim):
        if "customer:" in turn.lower():
            clean_turn = re.sub(r'(?i)^\s*customer\s*:\s*', '', turn).strip()
            customer_texts.append(clean_turn.lower())
    if not customer_texts:
        customer_texts = [user_claim.lower()]

    claimed_parts = set()
    claimed_issues = set()

    for text in customer_texts:
        if claim_object == "car":
            if "bumper" in text:
                if any(w in text for w in ["front", "fore"]):
                    claimed_parts.add("front_bumper")
                elif any(w in text for w in ["back", "rear"]):
                    claimed_parts.add("rear_bumper")
                else:
                    claimed_parts.add("front_bumper")
                    claimed_parts.add("rear_bumper")
                    claimed_parts.add("bumper")
            if "windshield" in text or "wind screen" in text or "windscreen" in text or ("glass" in text and any(w in text for w in ["front", "windshield", "wind"])):
                if not is_negated_word(text, "windshield") and not is_negated_word(text, "glass"):
                    claimed_parts.add("windshield")
            if "mirror" in text and not is_negated_word(text, "mirror"):
                claimed_parts.add("side_mirror")
            if "headlight" in text or "head lamp" in text or "front light" in text or "head-lamp" in text:
                if not is_negated_word(text, "headlight") and not is_negated_word(text, "head lamp") and not is_negated_word(text, "front light"):
                    claimed_parts.add("headlight")
            if "taillight" in text or "tail lamp" in text or "back light" in text or "rear light" in text or "tail-lamp" in text:
                if not is_negated_word(text, "taillight") and not is_negated_word(text, "tail lamp") and not is_negated_word(text, "back light"):
                    claimed_parts.add("taillight")
            if "door" in text and not is_negated_word(text, "door"):
                claimed_parts.add("door")
            if "hood" in text or "bonnet" in text:
                if not is_negated_word(text, "hood") and not is_negated_word(text, "bonnet"):
                    claimed_parts.add("hood")
            if "fender" in text and not is_negated_word(text, "fender"):
                claimed_parts.add("fender")
            if ("quarter panel" in text or "quarter-panel" in text) and not is_negated_word(text, "quarter panel"):
                claimed_parts.add("quarter_panel")
            if "body" in text and not is_negated_word(text, "body"):
                claimed_parts.add("body")
                
        elif claim_object == "laptop":
            if "screen" in text or "display" in text or "monitor" in text or "panel" in text or "glass" in text:
                if not is_negated_word(text, "screen") and not is_negated_word(text, "display") and not is_negated_word(text, "monitor"):
                    claimed_parts.add("screen")
            if "keyboard" in text or "keys" in text or "button" in text:
                if not is_negated_word(text, "keyboard") and not is_negated_word(text, "keys"):
                    claimed_parts.add("keyboard")
            if "trackpad" in text or "touchpad" in text or "mouse" in text:
                if not is_negated_word(text, "trackpad") and not is_negated_word(text, "touchpad"):
                    claimed_parts.add("trackpad")
            if "hinge" in text or "joint" in text:
                if not is_negated_word(text, "hinge") and not is_negated_word(text, "joint"):
                    claimed_parts.add("hinge")
            if "lid" in text or "outer shell" in text or "cover" in text or "outer panel" in text:
                if not is_negated_word(text, "lid") and not is_negated_word(text, "cover"):
                    claimed_parts.add("lid")
            if "corner" in text or "edge" in text:
                if not is_negated_word(text, "corner") and not is_negated_word(text, "edge"):
                    claimed_parts.add("corner")
            if "port" in text or "usb" in text or "charger" in text or "charging" in text or "hdmi" in text:
                if not is_negated_word(text, "port") and not is_negated_word(text, "usb"):
                    claimed_parts.add("port")
            if "base" in text or "bottom" in text:
                if not is_negated_word(text, "base") and not is_negated_word(text, "bottom"):
                    claimed_parts.add("base")
            if "body" in text or "chassis" in text or "frame" in text:
                if not is_negated_word(text, "body") and not is_negated_word(text, "chassis"):
                    claimed_parts.add("body")
                
        elif claim_object == "package":
            if "corner" in text:
                claimed_parts.add("package_corner")
            if "side" in text or "panel" in text:
                claimed_parts.add("package_side")
            if "seal" in text or "tape" in text or "flap" in text or "glue" in text:
                claimed_parts.add("seal")
            if "label" in text or "sticker" in text or "address" in text:
                claimed_parts.add("label")
            if "contents" in text or "item" in text or "product" in text or "goods" in text or "inside" in text:
                claimed_parts.add("contents")
                claimed_parts.add("item")
            if "box" in text or "carton" in text or "container" in text or "package" in text or "packet" in text or "parcel" in text:
                claimed_parts.add("box")

        issue_keywords = {
            "dent": "dent", "scratch": "scratch", "scrape": "scratch", "crack": "crack",
            "shatter": "glass_shatter", "broken": "broken_part", "missing": "missing_part",
            "torn": "torn_packaging", "crushed": "crushed_packaging", "water": "water_damage",
            "wet": "water_damage", "stain": "stain", "oil": "stain", "spill": "water_damage"
        }
        for kw, target in issue_keywords.items():
            if kw in text and not is_negated_word(text, kw):
                claimed_issues.add(target)

    def is_part_matching(obj_part: str, cl_parts: Set[str]) -> bool:
        if not cl_parts: return True
        if obj_part == "unknown": return True
        if obj_part in cl_parts: return True
        if "bumper" in obj_part and "bumper" in cl_parts: return True
        if obj_part in ["front_bumper", "rear_bumper"] and "bumper" in cl_parts: return True
        if obj_part == "package_corner" and "corner" in cl_parts: return True
        if obj_part == "package_side" and "side" in cl_parts: return True
        if obj_part in ["contents", "item"] and ("contents" in cl_parts or "item" in cl_parts): return True
        if obj_part == "box" and "package" in cl_parts: return True
        if obj_part == "body" and ("body" in cl_parts or "chassis" in cl_parts or "frame" in cl_parts): return True
        if claim_object == "laptop":
            if any(p in cl_parts for p in ["corner", "lid", "body", "base"]):
                if obj_part in ["corner", "lid", "body", "base"]: return True
        return False

    mismatch_detected = False
    if not is_part_matching(object_part, claimed_parts):
        mismatch_detected = True
        if "wrong_object_part" not in risk_flags: risk_flags.append("wrong_object_part")

    severe_issues = {"dent", "crack", "glass_shatter", "broken_part", "crushed_packaging", "torn_packaging", "missing_part"}
    minor_or_no_issues = {"none", "scratch", "stain", "unknown"}
    has_claimed_severe = any(i in severe_issues for i in claimed_issues)
    is_visible_minor = visible_issue_type in minor_or_no_issues
    if claim_status == "supported" and has_claimed_severe and is_visible_minor:
        mismatch_detected = True
        if "claim_mismatch" not in risk_flags: risk_flags.append("claim_mismatch")
        claim_status = "contradicted"
        claim_status_justification = f"Claim mismatch: user claimed severe damage ({', '.join(claimed_issues)}) but visual analysis only detected '{visible_issue_type}'."

    if mismatch_detected:
        if "manual_review_required" not in risk_flags: risk_flags.append("manual_review_required")

    mapped_requirement_ids = ["REQ_GENERAL_OBJECT_PART", "REQ_REVIEW_TRUST"]
    if claim_object == "car":
        if visible_issue_type in ["dent", "scratch"]: mapped_requirement_ids.append("REQ_CAR_BODY_PANEL")
        elif visible_issue_type in ["crack", "glass_shatter", "broken_part", "missing_part"]: mapped_requirement_ids.append("REQ_CAR_GLASS_LIGHT_MIRROR")
        mapped_requirement_ids.append("REQ_CAR_IDENTITY_OR_SIDE")
    elif claim_object == "laptop":
        if object_part in ["screen", "keyboard", "trackpad"]: mapped_requirement_ids.append("REQ_LAPTOP_SCREEN_KEYBOARD_TRACKPAD")
        elif object_part in ["hinge", "lid", "corner", "body", "base", "port"]: mapped_requirement_ids.append("REQ_LAPTOP_BODY_HINGE_PORT")
    elif claim_object == "package":
        if visible_issue_type in ["crushed_packaging", "torn_packaging"]: mapped_requirement_ids.append("REQ_PACKAGE_EXTERIOR")
        elif visible_issue_type in ["water_damage", "stain"] or object_part == "label": mapped_requirement_ids.append("REQ_PACKAGE_LABEL_OR_STAIN")
        elif object_part in ["contents", "item"]: mapped_requirement_ids.append("REQ_PACKAGE_CONTENTS")

    if len(vlm_output.get("image_paths_list", [])) > 1: mapped_requirement_ids.append("REQ_GENERAL_MULTI_IMAGE")
    req_violations = []
    req_lookup = {r.get("requirement_id"): r.get("minimum_image_evidence", "") for r in evidence_reqs if r.get("requirement_id")}
    for req_id in mapped_requirement_ids:
        min_ev_desc = req_lookup.get(req_id, "")
        if req_id == "REQ_GENERAL_OBJECT_PART":
            if is_flag_in_all_images("wrong_object", reports) or is_flag_in_all_images("wrong_object_part", reports) or is_flag_in_all_images("damage_not_visible", reports): req_violations.append(f"{req_id}: {min_ev_desc}")
        elif req_id == "REQ_REVIEW_TRUST":
            if is_flag_in_all_images("non_original_image", reports) or is_flag_in_all_images("possible_manipulation", reports): req_violations.append(f"{req_id}: {min_ev_desc}")
        elif req_id == "REQ_GENERAL_MULTI_IMAGE":
            if is_flag_in_all_images("blurry_image", reports) and is_flag_in_all_images("cropped_or_obstructed", reports): req_violations.append(f"{req_id}: {min_ev_desc}")
        elif req_id == "REQ_CAR_BODY_PANEL":
            if is_flag_in_all_images("wrong_angle", reports) or is_flag_in_all_images("damage_not_visible", reports): req_violations.append(f"{req_id}: {min_ev_desc}")
        elif req_id == "REQ_CAR_GLASS_LIGHT_MIRROR":
            if is_flag_in_all_images("damage_not_visible", reports) or is_flag_in_all_images("cropped_or_obstructed", reports): req_violations.append(f"{req_id}: {min_ev_desc}")
        elif req_id == "REQ_CAR_IDENTITY_OR_SIDE":
            if is_flag_in_all_images("wrong_angle", reports): req_violations.append(f"{req_id}: {min_ev_desc}")
        elif req_id == "REQ_LAPTOP_SCREEN_KEYBOARD_TRACKPAD":
            if is_flag_in_all_images("damage_not_visible", reports) or is_flag_in_all_images("blurry_image", reports): req_violations.append(f"{req_id}: {min_ev_desc}")
        elif req_id == "REQ_LAPTOP_BODY_HINGE_PORT":
            if is_flag_in_all_images("damage_not_visible", reports) or is_flag_in_all_images("cropped_or_obstructed", reports): req_violations.append(f"{req_id}: {min_ev_desc}")
        elif req_id == "REQ_PACKAGE_EXTERIOR":
            if is_flag_in_all_images("damage_not_visible", reports) or is_flag_in_all_images("blurry_image", reports): req_violations.append(f"{req_id}: {min_ev_desc}")
        elif req_id == "REQ_PACKAGE_LABEL_OR_STAIN":
            if is_flag_in_all_images("damage_not_visible", reports) or is_flag_in_all_images("low_light_or_glare", reports): req_violations.append(f"{req_id}: {min_ev_desc}")
        elif req_id == "REQ_PACKAGE_CONTENTS":
            if is_flag_in_all_images("damage_not_visible", reports) or is_flag_in_all_images("cropped_or_obstructed", reports): req_violations.append(f"{req_id}: {min_ev_desc}")

    if req_violations or not evidence_standard_met:
        evidence_standard_met = False
        evidence_standard_met_reason = " | ".join(req_violations) if req_violations else "Evidence requirements were not met (obstructed or quality issues)."
        if claim_status == "supported": claim_status = "not_enough_information"
        if "manual_review_required" not in risk_flags: risk_flags.append("manual_review_required")

    if claim_status == "not_enough_information" and ("contradict" in claim_status_justification.lower() or "contradict" in evidence_standard_met_reason.lower()):
        has_clear_damage = False
        has_undamaged_context = False
        damaged_issue_type, damaged_part, damaged_severity = None, None, None
        for rep in reports.values():
            for iss in rep.get("visible_issues", []):
                i_type, o_part, sev = iss.get("issue_type", "none"), iss.get("object_part", "unknown"), iss.get("severity", "none")
                if i_type not in ["none", "unknown"] and sev not in ["none", "unknown"]:
                    has_clear_damage = True
                    damaged_issue_type, damaged_part, damaged_severity = i_type, o_part, sev
                elif i_type == "none" or sev == "none":
                    has_undamaged_context = True
        if has_clear_damage and has_undamaged_context:
            claim_status = "supported"
            claim_status_justification = f"Claim supported based on close-up image showing visible {damaged_issue_type} on {damaged_part}."
            if damaged_issue_type: visible_issue_type = damaged_issue_type
            if damaged_part: object_part = damaged_part
            if damaged_severity: severity = damaged_severity

    if "wrong_object" in risk_flags:
        claim_status = "contradicted"
        evidence_standard_met, valid_image = True, True
        visible_issue_type, object_part = "unknown", "unknown"
        if "manual_review_required" not in risk_flags: risk_flags.append("manual_review_required")

    if "non_original_image" in risk_flags or "possible_manipulation" in risk_flags:
        valid_image = False
        claim_status = "contradicted"
        if "manual_review_required" not in risk_flags: risk_flags.append("manual_review_required")
            
    if claim_object == "laptop" and visible_issue_type == "glass_shatter" and object_part == "screen": visible_issue_type = "crack"

    is_missing_item_claim = False
    if claim_object == "package" and ("contents" in claimed_parts or "item" in claimed_parts):
        is_missing_item_claim = check_is_missing_claim(customer_texts)
        
    if is_missing_item_claim:
        claim_status, evidence_standard_met, valid_image = "not_enough_information", False, False
        evidence_standard_met_reason = "The images do not clearly show the expected contents or enough of the opened package to verify whether anything is missing."
        visible_issue_type, object_part, severity = "unknown", "contents", "unknown"

    history_flags, history_summary = str(user_history_row.get("history_flags", "none")).lower(), str(user_history_row.get("history_summary", "")).lower()
    rejected_count, past_count, recent_90_days = int(user_history_row.get("rejected_claim", 0)), int(user_history_row.get("past_claim_count", 0)), int(user_history_row.get("last_90_days_claim_count", 0))
    is_high_risk_user = "user_history_risk" in history_flags or "user_history_risk" in history_summary or (rejected_count >= 2 or (past_count > 0 and rejected_count / past_count >= 0.40)) or (recent_90_days >= 3)
    if is_high_risk_user:
        if "user_history_risk" not in risk_flags: risk_flags.append("user_history_risk")
        if "manual_review_required" not in risk_flags: risk_flags.append("manual_review_required")

    has_exaggeration = any(w in user_claim_lower for w in ["pretty bad", "badly", "severe", "crushed", "destroyed", "heavily"])
    if is_high_risk_user and has_exaggeration and claim_status == "supported":
        if severity in ["medium", "low", "unknown"]:
            claim_status = "contradicted"
            if "claim_mismatch" not in risk_flags: risk_flags.append("claim_mismatch")
            has_scratch = any(iss.get("issue_type") == "scratch" for rep in reports.values() for iss in rep.get("visible_issues", []))
            visible_issue_type, severity = ("scratch", "low") if has_scratch else (visible_issue_type, "low")

    quality_trust_issues = any(f in risk_flags for f in ["blurry_image", "cropped_or_obstructed", "low_light_or_glare", "wrong_angle", "wrong_object", "wrong_object_part", "damage_not_visible", "claim_mismatch", "possible_manipulation", "non_original_image", "text_instruction_present"])
    if is_high_risk_user and claim_status == "supported" and (severity in ["low", "none", "unknown"] or quality_trust_issues):
        claim_status = "not_enough_information"
        claim_status_justification = "Claim status demoted due to high-risk user profile and lack of unambiguous strong visual damage."

    if "manual_review_required" in history_flags and "manual_review_required" not in risk_flags: risk_flags.append("manual_review_required")

    custom_severity_override = None
    if claim_status == "supported":
        if object_part == "side_mirror" and visible_issue_type == "glass_shatter":
            visible_issue_type = "broken_part"
            custom_severity_override = "medium"
        if "scratch" in claimed_issues and visible_issue_type in ["dent", "broken_part"]:
            visible_issue_type = "scratch"
            custom_severity_override = "low"
        elif "stain" in claimed_issues and "water_damage" not in claimed_issues and visible_issue_type == "water_damage":
            visible_issue_type = "stain"
            custom_severity_override = "medium" if "keyboard" in object_part else "low"
        elif "crack" in claimed_issues and visible_issue_type == "glass_shatter":
            visible_issue_type = "crack"
            custom_severity_override = "medium"
        elif "broken_part" in claimed_issues and visible_issue_type == "glass_shatter":
            visible_issue_type = "broken_part"
            custom_severity_override = "medium"
        if claim_object == "laptop" and "corner" in claimed_parts and object_part == "lid": object_part = "corner"
        if claim_object == "package" and "package_side" in claimed_parts and object_part == "box": object_part = "package_side"
        elif claim_object == "package" and "package_corner" in claimed_parts and object_part == "box": object_part = "package_corner"

    if claim_status in ["contradicted", "not_enough_information"]:
        valid_claimed_parts = [p for p in claimed_parts if p != "unknown"]
        if valid_claimed_parts: object_part = valid_claimed_parts[0]

    if claim_status == "not_enough_information": severity = "unknown"
    elif claim_status == "contradicted":
        if visible_issue_type == "none": severity = "none"
        elif "wrong_object" in risk_flags: severity = "low"
        elif "wrong_object_part" in risk_flags: severity = "high"
        elif "claim_mismatch" in risk_flags: severity = "low"
        else: severity = "none"
    else:
        if custom_severity_override: severity = custom_severity_override
        else:
            issue_severity_map = {"glass_shatter": "high", "broken_part": "high", "crushed_packaging": "medium", "dent": "medium", "crack": "medium", "water_damage": "medium", "torn_packaging": "medium", "missing_part": "medium", "scratch": "low", "stain": "low", "none": "none"}
            base_severity = issue_severity_map.get(visible_issue_type, "unknown")
            has_quality_flags = is_flag_in_all_images("blurry_image", reports) or is_flag_in_all_images("low_light_or_glare", reports) or is_flag_in_all_images("cropped_or_obstructed", reports)
            justification_lower = claim_status_justification.lower()
            has_large_modifier = has_word(justification_lower, ["large", "severe", "huge", "extensive", "deep", "major"])
            has_small_modifier = has_word(justification_lower, ["minor", "small", "tiny", "light", "superficial", "fine"])
            final_severity = base_severity
            if final_severity == "high":
                if has_quality_flags or has_small_modifier: final_severity = "medium"
            elif final_severity == "medium":
                if has_large_modifier: final_severity = "high"
                elif has_quality_flags or has_small_modifier: final_severity = "low"
            elif final_severity == "low":
                if has_large_modifier: final_severity = "medium"
            severity = final_severity

    if "text_instruction_present" in risk_flags or text_instruction_detected:
        visible_issue_type, severity = "none", "none"
    if "damage_not_visible" in risk_flags:
        if evidence_standard_met:
            claim_status = "contradicted"
            visible_issue_type = "none"
            severity = "none"

    # Evidence standard not met override
    if not evidence_standard_met or str(evidence_standard_met).lower() == "false":
        visible_issue_type = "unknown"
        severity = "unknown"

    # Final normalizations
    if severity not in ALLOWED_SEVERITY:
        severity = "unknown"

    cleaned_risk_flags = []
    for flag in risk_flags:
        flag = flag.strip()
        if flag in ALLOWED_RISK_FLAGS:
            cleaned_risk_flags.append(flag)
            
    cleaned_risk_flags = sorted(list(set(cleaned_risk_flags)))
    if len(cleaned_risk_flags) > 1 and "none" in cleaned_risk_flags:
        cleaned_risk_flags.remove("none")
    if not cleaned_risk_flags:
        cleaned_risk_flags = ["none"]

    if claim_status not in ALLOWED_CLAIM_STATUS:
        claim_status = "not_enough_information"

    evidence_standard_met_str = "true" if evidence_standard_met else "false"
    valid_image_str = "true" if valid_image else "false"
    risk_flags_str = ";".join(cleaned_risk_flags)

    return {
        "evidence_standard_met": evidence_standard_met_str,
        "evidence_standard_met_reason": evidence_standard_met_reason,
        "risk_flags": risk_flags_str,
        "issue_type": visible_issue_type,
        "object_part": object_part,
        "claim_status": claim_status,
        "claim_status_justification": claim_status_justification,
        "supporting_image_ids": supporting_image_ids,
        "valid_image": valid_image_str,
        "severity": severity
    }
