import os
import json
import asyncio
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

# Install: pip install langchain-community sentence-transformers opencv-python pillow langchain-groq
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

# --- SHARED CONTEXT LEDGER ---
@dataclass
class ModerationContext:
    post_id: str
    text: str = ""
    image_path: Optional[str] = None
    user_region: str = "global"
    language: str = "en"
    agent_decisions: Dict[str, Any] = None
    conflicts: List[str] = None
    final_decision: str = "pending"
    evidence_log: List[Dict] = None
    
    def __post_init__(self):
        if self.agent_decisions is None: self.agent_decisions = {}
        if self.conflicts is None: self.conflicts = []
        if self.evidence_log is None: self.evidence_log = []

# --- AGENT BASE CLASS ---
class ModerationAgent:
    def __init__(self, name: str, llm, jurisdiction: str = "global"):
        self.name = name
        self.llm = llm
        self.jurisdiction = jurisdiction
        self.hard_limits = ["csam", "terrorism", "child_endangerment"]
    
    def log_evidence(self, context: ModerationContext, evidence: str, severity: float):
        context.evidence_log.append({
            "agent": self.name,
            "timestamp": datetime.utcnow().isoformat(),
            "evidence": evidence,
            "severity": severity,
            "jurisdiction": self.jurisdiction
        })

# --- TEXT ANALYSIS AGENT ---
class TextAnalysisAgent(ModerationAgent):
    async def moderate(self, context: ModerationContext) -> Dict[str, Any]:
        # Hard limit check
        for term in self.hard_limits:
            if term in context.text.lower():
                self.log_evidence(context, f"Hard limit violation: {term}", 1.0)
                return {"decision": "block", "reason": "hard_limit", "confidence": 1.0}
        
        # Hate speech detection prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a hate speech detector for {region}. Respond ONLY with valid JSON: {{'hate_speech': bool, 'severity': float, 'explanation': str}}"),
            ("human", "Text: {text}")
        ])
        chain = (
            prompt
            | self.llm
        )
        try:
            # Invoking the chain with specific inputs
            response = chain.invoke({"region": self.jurisdiction, "text": context.text})
            # Parse JSON from content (handling potential markdown wrappers)
            content = response.content.replace("```json", "").replace("```", "").strip()
            result = json.loads(content)
            
            self.log_evidence(context, result["explanation"], result["severity"])
            decision = "block" if result["hate_speech"] and result["severity"] > 0.7 else "allow"
            return {"decision": decision, "reason": "hate_speech", "confidence": result["severity"]}
        except Exception as e:
            return {"decision": "review", "reason": f"error_or_ambiguous: {str(e)}", "confidence": 0.5}

# --- IMAGE RECOGNITION AGENT ---
class ImageRecognitionAgent(ModerationAgent):
    async def moderate(self, context: ModerationContext) -> Dict[str, Any]:
        if not context.image_path:
            return {"decision": "allow", "reason": "no_image", "confidence": 1.0}
        
        # Simulate image analysis (replace with CV model or multimodal LLM)
        known_violations = ["nudity", "violence", "hate_symbols"]
        detected = []
        severity = 0.0
        
        # Mock detection logic based on keywords in the "path" or description
        image_desc = context.image_path.lower()
        if "nude" in image_desc or "nudity" in image_desc:
            detected.append("nudity")
            severity = 0.9
        if "gun" in image_desc or "violence" in image_desc or "fight" in image_desc:
            detected.append("violence")
            severity = 0.85
        
        # Hard limit check
        for term in self.hard_limits:
            if term in detected:
                self.log_evidence(context, f"Hard limit violation: {term}", 1.0)
                return {"decision": "block", "reason": "hard_limit", "confidence": 1.0}
        
        if detected:
            self.log_evidence(context, f"Detected: {', '.join(detected)}", severity)
            return {"decision": "block", "reason": "visual_harm", "confidence": severity}
        else:
            return {"decision": "allow", "reason": "safe_image", "confidence": 0.95}

# --- CULTURAL CONTEXT AGENT ---
class CulturalContextAgent(ModerationAgent):
    async def moderate(self, context: ModerationContext) -> Dict[str, Any]:
        # Check for culturally acceptable phrasing (simple mock)
        sensitive_phrases = {
            "fr": ["liberté", "résistance"],
            "jp": ["切腹", "侍"],
            "in": ["jai shree ram", "azaadi"]
        }
        
        region_phrases = sensitive_phrases.get(context.user_region, [])
        if any(phrase in context.text.lower() for phrase in region_phrases):
            self.log_evidence(context, f"Culturally sensitive phrase in {context.user_region}", 0.3)
            # Sensitive but maybe allowable in context, or flagged for review
            return {"decision": "review", "reason": "cultural_context_sensitive", "confidence": 0.6}
        
        return {"decision": "neutral", "reason": "no_cultural_flags", "confidence": 0.9}

# --- LEGAL COMPLIANCE AGENT ---
class LegalComplianceAgent(ModerationAgent):
    def __init__(self, name: str, llm, jurisdiction: str):
        super().__init__(name, llm, jurisdiction)
        self.laws = {
            "eu": ["DSA_hate_speech", "GDPR_pii"],
            "us": ["Section_230", "COPPA"],
            "sg": ["POFMA", "PDPA"]
        }
    
    async def moderate(self, context: ModerationContext) -> Dict[str, Any]:
        applicable_laws = self.laws.get(self.jurisdiction, [])
        if not applicable_laws:
            return {"decision": "neutral", "reason": "no_jurisdiction_laws", "confidence": 0.9}
        
        text_lower = context.text.lower()
        
        # Check for PII (mock)
        if "phone" in text_lower or "email@" in text_lower:
            self.log_evidence(context, f"PII detected under {self.jurisdiction} law", 0.8)
            return {"decision": "block", "reason": "legal_pii", "confidence": 0.85}
        
        # Hate speech under DSA
        if self.jurisdiction == "eu" and "hate" in text_lower:
            self.log_evidence(context, "DSA hate speech violation", 0.9)
            return {"decision": "block", "reason": "legal_hate_speech", "confidence": 0.9}
        
        return {"decision": "neutral", "reason": "compliant", "confidence": 0.95}

# --- ARBITRATION ENGINE ---
class ArbitrationEngine:
    def __init__(self):
        self.weights = {
            "LegalComplianceAgent": 0.4,
            "TextAnalysisAgent": 0.25,
            "ImageRecognitionAgent": 0.25,
            "CulturalContextAgent": 0.1
        }
        self.human_review_threshold = 0.3  # Max allowed confidence variance
    
    async def resolve(self, context: ModerationContext) -> str:
        decisions = context.agent_decisions
        if not decisions:
            return "review"
        
        # Hard limit override
        for agent, result in decisions.items():
            if result.get("reason") == "hard_limit":
                return "block"
        
        # Weighted voting
        block_score = 0.0
        allow_score = 0.0
        confidences = []
        
        for agent_name, result in decisions.items():
            weight = self.weights.get(agent_name, 0.1)
            confidence = result.get("confidence", 0.5)
            confidences.append(confidence)
            
            decision = result.get("decision", "neutral")
            
            if decision == "block":
                block_score += weight * confidence
            elif decision == "allow":
                allow_score += weight * confidence
            # neutral doesn't add to either
        
        # Conflict detection
        if confidences:
            max_conf = max(confidences)
            min_conf = min(confidences)
            if max_conf - min_conf > self.human_review_threshold:
                context.conflicts.append(f"High disagreement: {max_conf:.2f} vs {min_conf:.2f}")
                return "review"
        
        # Final decision
        if block_score > allow_score:
            return "block"
        elif allow_score > block_score:
            return "allow"
        else:
            return "review"

# --- MONITORING SYSTEM ---
class MonitoringSystem:
    def __init__(self):
        self.metrics = {
            "false_positive_rate": {},
            "disagreement_rate": 0.0,
            "hard_limit_hits": 0,
            "total_processed": 0
        }
    
    def log_decision(self, context: ModerationContext, ground_truth: Optional[str] = None):
        self.metrics["total_processed"] += 1
        
        # Track hard limit hits
        if any(e["severity"] == 1.0 for e in context.evidence_log):
            self.metrics["hard_limit_hits"] += 1
        
        # Fairness by region (mock)
        region = context.user_region
        if region not in self.metrics["false_positive_rate"]:
            self.metrics["false_positive_rate"][region] = {"fp": 0, "total": 0}
        
        self.metrics["false_positive_rate"][region]["total"] += 1
