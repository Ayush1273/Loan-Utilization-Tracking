import os
import base64
import mimetypes
import json
import sys
from typing import Dict, Any, Optional
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

# Load environment variables
load_dotenv()

# System instruction for classification
SYSTEM_INSTRUCTION = """
You are an AI model designed to identify and classify images into one of the following loan asset categories:
Agriculture Loan Assets, Education Loan Assets, Business / MSME Loan Assets, Home / Housing Loan Assets, or Vehicle / Transport Loan Assets.

Analyze the visual content of each image and assign it to the most appropriate category and sub-category based on the following definitions and valid examples. Flag any suspicious or irrelevant images.

1️⃣ Agriculture Loan Assets
Sub-Categories:
- Farm Equipment: Tractor, Power tiller
- Irrigation Systems: Water pump, Drip pipe setup
✅ Valid: tractor, fertilizer, seeds, water pump
🚫 Suspicious: mobile, car, restaurant

2️⃣ Education Loan Assets
Sub-Categories:
- Electronics for Study: Laptop, Tablet, Desktop
- Institutional Setup: Classroom, College/school building
- Fee Proofs: Admission receipt, ID card
✅ Valid: laptop, classroom, books
🚫 Suspicious: food, party, bike

3️⃣ Business / MSME Loan Assets
Sub-Categories:
- Shop / Store: Shopfront, Grocery/retail shelves
- Machinery / Tools: Sewing machine, Packaging/printing machine
✅ Valid: machines, shop
🚫 Suspicious: sofa, jewellery, clothes for personal use

4️⃣ Home / Housing Loan Assets
Sub-Categories:
- Construction Site: Under-construction house, Brick wall, Cement bags
- Land & Property: Empty land plot, Building foundation
✅ Valid: land plot, under-construction home
🚫 Suspicious: cars, luxury furniture, gadgets

5️⃣ Vehicle / Transport Loan Assets
Sub-Categories:
- Vehicles: Car, Two-wheeler, Tractor
- Workshops / Maintenance: Garage, Vehicle service area
✅ Valid: car, bike, RC paper
🚫 Suspicious: unrelated electronics, home interiors

Output format (JSON only, no markdown):
{
  "category": "...",
  "sub_category": "...",
  "is_suspicious": true/false,
  "confidence": "0.0-1.0",
  "reason": "short reasoning"
}
"""


class LoanAssetClassifier:
    """Main classifier class for loan asset images."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the classifier.
        
        Args:
            api_key: Google API key. If None, reads from GEMINI_API_KEY env var
        """
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found. Please set it as environment variable or pass it to constructor.")
        
        self.model = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp",
            google_api_key=self.api_key,
            temperature=0
        )
    
    def encode_image(self, image_path: str) -> tuple[str, str]:
        """
        Encode image to base64 and determine MIME type.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Tuple of (mime_type, base64_encoded_data)
        """
        mime_type, _ = mimetypes.guess_type(image_path)
        if mime_type is None:
            mime_type = "image/jpeg"
        
        with open(image_path, "rb") as image_file:
            image_data = base64.b64encode(image_file.read()).decode('utf-8')
        
        return mime_type, image_data
    
    def encode_image_bytes(self, image_bytes: bytes, filename: str = "image.jpg") -> tuple[str, str]:
        """
        Encode image bytes to base64 and determine MIME type.
        
        Args:
            image_bytes: Raw image bytes
            filename: Original filename for MIME type detection
            
        Returns:
            Tuple of (mime_type, base64_encoded_data)
        """
        mime_type, _ = mimetypes.guess_type(filename)
        if mime_type is None:
            mime_type = "image/jpeg"
        
        image_data = base64.b64encode(image_bytes).decode('utf-8')
        return mime_type, image_data
    
    def classify(self, image_path: str) -> Dict[str, Any]:
        """
        Classify an image from file path.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary with classification result
        """
        try:
            if not os.path.exists(image_path):
                return {"error": f"File not found: {image_path}"}
            
            mime_type, image_data = self.encode_image(image_path)
            return self._classify_image(mime_type, image_data)
            
        except Exception as e:
            return {"error": f"Classification failed: {str(e)}"}
    
    def classify_bytes(self, image_bytes: bytes, filename: str = "image.jpg") -> Dict[str, Any]:
        """
        Classify an image from bytes.
        
        Args:
            image_bytes: Raw image bytes
            filename: Original filename for MIME type detection
            
        Returns:
            Dictionary with classification result
        """
        try:
            mime_type, image_data = self.encode_image_bytes(image_bytes, filename)
            return self._classify_image(mime_type, image_data)
            
        except Exception as e:
            return {"error": f"Classification failed: {str(e)}"}
    
    def _classify_image(self, mime_type: str, image_data: str) -> Dict[str, Any]:
        """
        Internal method to classify an encoded image.
        
        Args:
            mime_type: MIME type of the image
            image_data: Base64 encoded image data
            
        Returns:
            Dictionary with classification result
        """
        messages = [
            SystemMessage(content=SYSTEM_INSTRUCTION),
            HumanMessage(
                content=[
                    {
                        "type": "text",
                        "text": "Classify this loan asset image according to the instructions. Return only valid JSON."
                    },
                    {
                        "type": "image_url",
                        "image_url": f"data:{mime_type};base64,{image_data}"
                    }
                ]
            )
        ]
        
        response = self.model.invoke(messages)
        result_text = response.content
        
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
                "category": "Unknown",
                "sub_category": "Unknown",
                "is_suspicious": True,
                "confidence": "low",
                "reason": "Failed to parse model response",
                "raw_response": result_text
            }
    
    def classify_batch(self, image_paths: list[str]) -> Dict[str, Dict[str, Any]]:
        """
        Classify multiple images at once.
        
        Args:
            image_paths: List of paths to image files
            
        Returns:
            Dictionary mapping image paths to their classification results
        """
        results = {}
        for image_path in image_paths:
            results[image_path] = self.classify(image_path)
        return results


# For backwards compatibility and direct usage
def classify_loan_asset(image_path: str, api_key: Optional[str] = None) -> Dict[str, Any]:
    """
    Standalone function to classify a single image.
    
    Args:
        image_path: Path to the image file
        api_key: Optional Google API key
        
    Returns:
        Dictionary with classification result
    """
    classifier = LoanAssetClassifier(api_key=api_key)
    return classifier.classify(image_path)


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python loan_classifier.py <image_path>")
        sys.exit(1)
    
    result = classify_loan_asset(sys.argv[1])
    print(json.dumps(result, indent=2))