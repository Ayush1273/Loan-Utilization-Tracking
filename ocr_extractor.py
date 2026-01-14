# ocr_extractor.py

import os
import base64
import mimetypes
from typing import Optional
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

# Load environment variables
load_dotenv()

def extract_text_from_image(image_path: str, api_key: Optional[str] = None) -> str:
    """
    Extracts and analyzes text from a given image using Google Gemini OCR.
    Returns extracted text + analysis summary.
    """
    try:
        # Initialize the model
        api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        
        model = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp",
            google_api_key=api_key,
            temperature=0
        )
        
        # Encode image to base64
        mime_type, _ = mimetypes.guess_type(image_path)
        if mime_type is None:
            mime_type = "image/jpeg"  # Default fallback
        
        with open(image_path, "rb") as image_file:
            image_data = base64.b64encode(image_file.read()).decode('utf-8')
        
        # Create messages for OCR
        system_message = SystemMessage(content="""
You are an OCR AI model specialized in extracting text from document images and analyzing their relevance to loan asset verification.

Your task:
1. Extract ALL readable text from receipts, invoices, bills, ID cards, or any document
2. Analyze the document content and classify it according to loan asset categories
3. Provide structured output with extracted text and analysis

Loan Asset Categories:
1️⃣ Agriculture Loan Assets
Sub-Categories:
- Farm Equipment: Tractor, Power tiller
- Irrigation Systems: Water pump, Drip pipe setup
✅ Valid: tractor, fertilizer, seeds, water pump

2️⃣ Education Loan Assets
Sub-Categories:
- Electronics for Study: Laptop, Tablet, Desktop
- Institutional Setup: Classroom, College/school building
- Fee Proofs: Admission receipt, ID card
✅ Valid: laptop, classroom, books

3️⃣ Business / MSME Loan Assets
Sub-Categories:
- Shop / Store: Shopfront, Grocery/retail shelves
- Machinery / Tools: Sewing machine, Packaging/printing machine
✅ Valid: machines, shop

4️⃣ Home / Housing Loan Assets
Sub-Categories:
- Construction Site: Under-construction house, Brick wall, Cement bags
- Land & Property: Empty land plot, Building foundation
✅ Valid: land plot, under-construction home

5️⃣ Vehicle / Transport Loan Assets
Sub-Categories:
- Vehicles: Car, Two-wheeler, Tractor
- Workshops / Maintenance: Garage, Vehicle service area
✅ Valid: car, bike, RC paper

Output format (JSON only, no markdown):
{
  "document_type": "receipt/invoice/bill/id_card/other",
  "extracted_text": "complete extracted text maintaining structure",
  "category": "Agriculture Loan Assets/Education Loan Assets/Business / MSME Loan Assets/Home / Housing Loan Assets/Vehicle / Transport Loan Assets/Other",
  "sub_category": "...",
  "is_relevant": true/false,
  "confidence": 0.0-1.0,
  "key_information": {
    "vendor_name": "extracted vendor/company name",
    "amount": "extracted amount if any", 
    "date": "extracted date if any",
    "item_description": "main items/services mentioned"
  },
  "analysis": "brief summary of document relevance for loan verification"
}

Important: Return only valid JSON format, no additional text or markdown formatting.
""")
        
        human_message = HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": "Please perform OCR on this image and extract all readable text. Maintain the structure and provide a summary of the document type and key information."
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime_type};base64,{image_data}"
                    }
                }
            ]
        )
        
        # Get response
        response = model.invoke([system_message, human_message])
        result_text = response.content.strip()
        
        # Try to parse as JSON first
        try:
            import json
            # Remove markdown code blocks if present
            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0].strip()
            elif "```" in result_text:
                result_text = result_text.split("```")[1].split("```")[0].strip()
            
            result = json.loads(result_text)
            return json.dumps(result, indent=2)
        except json.JSONDecodeError:
            # If JSON parsing fails, return the raw text
            return result_text
        
    except Exception as e:
        return f"Error extracting text from image: {str(e)}"
