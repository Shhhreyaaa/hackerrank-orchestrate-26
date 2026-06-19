# Submission Checklist

Before submitting your solution, ensure that all the following conditions are met:

- [ ] **Exact 14 Columns in output.csv**:
  Verify that `output.csv` exists at the repository root and has exactly these headers in this order:
  `user_id, image_paths, user_claim, claim_object, evidence_standard_met, evidence_standard_met_reason, risk_flags, issue_type, object_part, claim_status, claim_status_justification, supporting_image_ids, valid_image, severity`
- [ ] **No Missing Predictions**:
  `output.csv` must have exactly 45 rows corresponding to the rows in `dataset/claims.csv`.
- [ ] **AGENTS.md Log Compliance**:
  Check that `%USERPROFILE%\hackerrank_orchestrate\log.txt` is updated after every command execution and user conversation. Do not delete prior entries.
- [ ] **No Secrets Committed**:
  Ensure `.env` contains only generic placeholders (like `your_key_here`) and no live API keys are committed to git.
- [ ] **Evaluation Output Present**:
  Verify that the file `code/evaluation/evaluation_report.md` exists and contains performance comparison data.
- [ ] **Environment Setup**:
  Requirements file `requirements.txt` lists all necessary dependencies.
- [ ] **Visual Cache File**:
  `code/image_cache.json` exists and is populated, confirming cache optimization was active.
- [ ] **Code Quality**:
  Ensure all files in `code/` are well-commented and free of temporary placeholders.
