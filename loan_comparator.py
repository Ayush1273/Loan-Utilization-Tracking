# loan_comparator.py

import os
import json
from typing import Dict, Any, Optional
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

# Load environment variables
load_dotenv()

class LoanComparisonAnalyzer:
    """
    Compares asset classification results with invoice OCR results to determine
    consistency and validity of loan applications using Google Gemini AI.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the comparison analyzer.
        
        Args:
            api_key: Google API key. If None, reads from GEMINI_API_KEY env var
        """
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found. Please set it as environment variable or pass it to constructor.")
        
        self.model = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp",
            google_api_key=self.api_key,
            temperature=0.1
        )
    
    def compare_results(self, asset_classification: Dict[str, Any], invoice_ocr: str) -> Dict[str, Any]:
        """
        Compare asset classification with invoice OCR results.
        
        Args:
            asset_classification: Result from loan_classifier
            invoice_ocr: JSON string result from ocr_extractor
            
        Returns:
            Dictionary with comparison analysis
        """
        try:
            # Parse invoice OCR if it's a JSON string
            if isinstance(invoice_ocr, str):
                try:
                    invoice_data = json.loads(invoice_ocr)
                except json.JSONDecodeError:
                    # If not JSON, treat as plain text
                    invoice_data = {"extracted_text": invoice_ocr, "category": "Unknown"}
            else:
                invoice_data = invoice_ocr
            
            return self._analyze_comparison(asset_classification, invoice_data)
            
        except Exception as e:
            return {"error": f"Comparison failed: {str(e)}"}
    
    def _analyze_comparison(self, asset_result: Dict[str, Any], invoice_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Internal method to perform the comparison analysis.
        
        Args:
            asset_result: Asset classification result
            invoice_result: Invoice OCR result
            
        Returns:
            Dictionary with comparison analysis
        """
        system_message = SystemMessage(content="""
You are an AI loan verification specialist that compares asset classification results with invoice/receipt analysis to determine loan application validity.

Your task:
1. Compare the asset classification with the invoice/receipt content
2. Determine if they are consistent and support each other
3. Identify any discrepancies or red flags
4. Provide an overall risk assessment

Loan Asset Categories for Reference:
1️⃣ Agriculture Loan Assets - Farm Equipment, Irrigation Systems
2️⃣ Education Loan Assets - Electronics for Study, Institutional Setup, Fee Proofs
3️⃣ Business / MSME Loan Assets - Shop/Store, Machinery/Tools
4️⃣ Home / Housing Loan Assets - Construction Site, Land & Property
5️⃣ Vehicle / Transport Loan Assets - Vehicles, Workshops/Maintenance

Analysis Criteria:
✅ CONSISTENT: Asset and invoice match the same category
✅ SUPPORTING: Invoice contains purchase details relevant to the asset
✅ AUTHENTIC: Invoice appears genuine with proper vendor, amount, date details
🚨 INCONSISTENT: Asset and invoice are from different categories
🚨 SUSPICIOUS: Asset flagged as suspicious OR invoice seems fake/irrelevant
🚨 MISSING INFO: Key information missing from either asset or invoice

Output format (JSON only, no markdown):
{
  "overall_status": "APPROVED/REVIEW_REQUIRED/REJECTED",
  "consistency_score": 0.0-1.0,
  "risk_level": "LOW/MEDIUM/HIGH",
  "category_match": true/false,
  "key_findings": [
    "finding 1",
    "finding 2"
  ],
  "discrepancies": [
    "discrepancy 1 if any",
    "discrepancy 2 if any"
  ],
  "recommendation": "detailed recommendation for loan officer",
  "confidence": 0.0-1.0,
  "verification_checklist": {
    "asset_category_valid": true/false,
    "invoice_authentic": true/false,
    "categories_match": true/false,
    "amounts_reasonable": true/false,
    "vendor_credible": true/false
  }
}

Important: Return only valid JSON format, no additional text or markdown formatting.
""")
        
        human_message = HumanMessage(content=f"""
Please analyze and compare these loan application components:

ASSET CLASSIFICATION RESULT:
{json.dumps(asset_result, indent=2)}

INVOICE/RECEIPT OCR RESULT:
{json.dumps(invoice_result, indent=2)}

Provide a comprehensive comparison analysis to determine if this loan application should be approved, requires review, or should be rejected.
""")
        
        response = self.model.invoke([system_message, human_message])
        result_text = response.content.strip()
        
        # Parse JSON from response
        try:
            # Remove markdown code blocks if present
            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0].strip()
            elif "```" in result_text:
                result_text = result_text.split("```")[1].split("```")[0].strip()
            
            result = json.loads(result_text)
            return result
        except json.JSONDecodeError:
            return {
                "overall_status": "REVIEW_REQUIRED",
                "consistency_score": 0.0,
                "risk_level": "HIGH",
                "category_match": False,
                "key_findings": ["Failed to parse comparison analysis"],
                "discrepancies": ["Analysis parsing error"],
                "recommendation": "Manual review required due to system error",
                "confidence": 0.0,
                "verification_checklist": {
                    "asset_category_valid": False,
                    "invoice_authentic": False,
                    "categories_match": False,
                    "amounts_reasonable": False,
                    "vendor_credible": False
                },
                "raw_response": result_text
            }


def compare_loan_application(asset_classification: Dict[str, Any], invoice_ocr: str, api_key: Optional[str] = None) -> Dict[str, Any]:
    """
    Standalone function to compare asset classification with invoice OCR.
    
    Args:
        asset_classification: Result from loan_classifier
        invoice_ocr: Result from ocr_extractor
        api_key: Optional Google API key
        
    Returns:
        Dictionary with comparison analysis
    """
    analyzer = LoanComparisonAnalyzer(api_key=api_key)
    return analyzer.compare_results(asset_classification, invoice_ocr)


if __name__ == "__main__":
    # Example usage
    sample_asset = {
        "category": "Vehicle / Transport Loan Assets",
        "sub_category": "Vehicles",
        "is_suspicious": False,
        "confidence": 0.95,
        "reason": "Clear vehicle image detected"
    }
    
    sample_invoice = {
        "document_type": "receipt",
        "category": "Vehicle / Transport Loan Assets", 
        "sub_category": "Vehicles",
        "is_relevant": True,
        "confidence": 0.9,
        "key_information": {
            "vendor_name": "ABC Motors",
            "amount": "$25,000",
            "date": "2024-10-01",
            "item_description": "2023 Honda Civic"
        }
    }
    
    result = compare_loan_application(sample_asset, json.dumps(sample_invoice))
    print(json.dumps(result, indent=2))