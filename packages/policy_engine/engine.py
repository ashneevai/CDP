"""Config-driven selection of the cheapest useful next evidence source."""
from __future__ import annotations

from pathlib import Path

import yaml

from packages.policy_engine.contracts import DecisionContext, PolicyAction, PolicyDecision

DEFAULT_POLICY_PATH=Path(__file__).resolve().parents[2]/"config"/"adaptive_routing.yaml"
_CLOUD={PolicyAction.TEXTRACT,PolicyAction.GEMINI_CHEAP,PolicyAction.GEMINI_STANDARD,PolicyAction.GEMINI_ADVANCED}
_LOCAL_OCR={
    PolicyAction.RETRY_PREPROCESSING,
    PolicyAction.RAPIDOCR,
    PolicyAction.PADDLEOCR,
    PolicyAction.TESSERACT,
    PolicyAction.DOCLING,
}

class AdaptivePolicyEngine:
    def __init__(self, config: dict): self.config=config
    @classmethod
    def load(cls,path: str|Path=DEFAULT_POLICY_PATH): return cls(yaml.safe_load(Path(path).read_text("utf-8")))
    def decide(self,c: DecisionContext)->PolicyDecision:
        threshold=float(self.config["acceptance_thresholds"].get(c.criticality.lower(),self.config["acceptance_thresholds"]["critical"]))
        quality=self.config["quality_thresholds"]
        reject_below=float(quality["registration_reject_below"])
        normal_at=float(quality["registration_normal_at"])
        if c.registration_confidence < reject_below:
            return self._decision(PolicyAction.HITL,"registration",["registration_rejected"])
        if not c.crop_safety_passed:
            action=(PolicyAction.EXPAND_CROP if PolicyAction.EXPAND_CROP not in c.previous_attempts else PolicyAction.HITL)
            return self._decision(action,"crop_safety",["wrong_crop_suspected"])
        if reject_below<=c.registration_confidence<normal_at and PolicyAction.EXPAND_CROP not in c.previous_attempts:
            return self._decision(PolicyAction.EXPAND_CROP,"registration",["bounded_registration_uncertainty"])
        validation_complete=bool(c.validation_results) and all(c.validation_results.values())
        if c.evidence_policy_satisfied and validation_complete and not c.unresolved_contradiction and c.current_confidence>=threshold and c.registration_confidence>=normal_at:
            return self._decision(PolicyAction.ACCEPT,"accept",["evidence_policy_satisfied"])
        if c.image_quality<float(quality["image_quality_retry_below"]) and PolicyAction.RAPIDOCR in c.previous_attempts and PolicyAction.RETRY_PREPROCESSING not in c.previous_attempts:
            return self._decision(PolicyAction.RETRY_PREPROCESSING,"image_quality",["low_image_quality"])
        route=self._route(c); skipped=[]
        configured_route = [PolicyAction(name) for name in self.config["routes"][route]]
        for index, action in enumerate(configured_route):
            if action in c.previous_attempts: continue
            if action is PolicyAction.REFERENCE_LOOKUP and not c.reference_available: skipped.append("reference_unavailable"); continue
            if action in _CLOUD:
                required_local = {
                    candidate for candidate in configured_route[:index]
                    if candidate in _LOCAL_OCR
                }
                if not required_local.issubset(c.previous_attempts):
                    skipped.append("local_ocr_not_exhausted")
                    continue
            if action in _CLOUD and not c.cloud_processing_allowed: skipped.append("cloud_processing_disallowed"); continue
            p=self.config["actions"][action.value]
            if p["cost_usd"]>c.remaining_budget: skipped.append("budget_exceeded"); continue
            if p["latency_seconds"]>c.remaining_sla: skipped.append("sla_exceeded"); continue
            reasons = ["needs_more_evidence", *sorted(set(skipped))]
            if action in _CLOUD:
                reasons.append("local_ocr_exhausted")
            return self._decision(action,route,reasons)
        fallback=PolicyAction.HITL if PolicyAction.HITL not in c.previous_attempts else PolicyAction.ABSTAIN
        return self._decision(fallback,route,["automated_routes_exhausted",*sorted(set(skipped))])
    def _route(self,c):
        n=c.field_name.lower()
        if c.is_table_field:return "table"
        if "npi" in n:return "npi"
        if any(x in n for x in ("member_id","subscriber_id","patient_id")):return "member_id"
        if "name" in n:return "name"
        if any(x in n for x in ("address","addr","city","state","zip")):return "address"
        if any(x in n for x in ("date","dob")):return "date"
        if any(x in n for x in ("amount","charge","total","quantity","units")):return "amount"
        if any(x in n for x in ("code","cpt","hcpcs","icd")):return "code"
        return "default"
    def _decision(self,a,r,reasons):
        p=self.config["actions"].get(a.value,{"cost_usd":0,"latency_seconds":0})
        return PolicyDecision(action=a,route=r,reason_codes=reasons,estimated_cost_usd=p["cost_usd"],estimated_latency_seconds=p["latency_seconds"])
