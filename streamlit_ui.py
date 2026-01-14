import streamlit as st
import os
from PIL import Image
import io
import json
from dotenv import load_dotenv
from loan_classifier import LoanAssetClassifier
from ocr_extractor import extract_text_from_image
from loan_comparator import LoanComparisonAnalyzer

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Loan Asset Classifier",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Simple styling (removed custom CSS for cleaner look)

# Initialize session state
if 'classifier' not in st.session_state:
    st.session_state.classifier = None
if 'comparator' not in st.session_state:
    st.session_state.comparator = None
if 'form_results' not in st.session_state:
    st.session_state.form_results = []

# Initialize classifier with API key from .env
api_key = os.environ.get("GEMINI_API_KEY")
if api_key:
    try:
        if not st.session_state.classifier:
            st.session_state.classifier = LoanAssetClassifier(api_key=api_key)
        if not st.session_state.comparator:
            st.session_state.comparator = LoanComparisonAnalyzer(api_key=api_key)
    except Exception as e:
        st.error(f"❌ Error initializing services: {str(e)}")
        st.stop()
else:
    st.error("❌ GEMINI_API_KEY not found in environment variables. Please check your .env file.")
    st.stop()

# Sidebar for Information
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/bank-building.png", width=80)
    st.title("📊 Loan Categories")
    
    categories = {
        "🌾 Agriculture": "Farm equipment, irrigation systems",
        "📚 Education": "Laptops, books, tuition fees",
        "🏪 Business/MSME": "Shops, machinery, tools",
        "🏠 Housing": "Construction, land, property",
        "🚗 Vehicle": "Cars, bikes, transport"
    }
    
    for cat, desc in categories.items():
        st.markdown(f"**{cat}**")
        st.caption(desc)
    
    st.divider()
    
    # Clear history button
    if st.button("🗑️ Clear History", use_container_width=True):
        st.session_state.form_results = []
        st.rerun()

# Main content
st.title("🏦 Loan Asset Classifier & Invoice Processor")
st.markdown("### Upload asset image and invoice to process loan application")

# Create tabs
tab1, tab2 = st.tabs(["� Loan Application Form", "📜 Processing History"])

with tab1:
    st.subheader("📋 Loan Application Processing Form")
    
    # Form for loan application
    with st.form("loan_application_form", clear_on_submit=True):
        st.markdown("### Required Documents")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📷 Asset Image **(Required)**")
            asset_file = st.file_uploader(
                "Upload asset image for classification",
                type=["jpg", "jpeg", "png", "webp"],
                key="asset_upload",
                help="Upload an image of the asset related to the loan (vehicle, property, equipment, etc.)"
            )
            
            if asset_file:
                asset_image = Image.open(asset_file)
                st.image(asset_image, caption="Asset Image", width='stretch')
        
        with col2:
            st.markdown("#### � Invoice/Receipt **(Required)**")
            invoice_file = st.file_uploader(
                "Upload invoice or receipt image",
                type=["jpg", "jpeg", "png", "webp"],
                key="invoice_upload",
                help="Upload an invoice, receipt, or document related to the asset purchase"
            )
            
            if invoice_file:
                invoice_image = Image.open(invoice_file)
                st.image(invoice_image, caption="Invoice/Receipt Image", width='stretch')
        
        # Submit button
        submitted = st.form_submit_button("🚀 Process Loan Application", type="primary", use_container_width=True)
        
        if submitted:
            if not asset_file:
                st.error("❌ Please upload an asset image")
            elif not invoice_file:
                st.error("❌ Please upload an invoice/receipt image")
            else:
                with st.spinner("Processing your loan application..."):
                    try:
                        # Process asset classification
                        asset_bytes = asset_file.getvalue()
                        asset_result = st.session_state.classifier.classify_bytes(
                            asset_bytes,
                            asset_file.name
                        )
                        
                        # Process invoice OCR
                        # Save invoice temporarily for OCR processing
                        temp_invoice_path = f"temp_{invoice_file.name}"
                        with open(temp_invoice_path, "wb") as f:
                            f.write(invoice_file.getvalue())
                        
                        try:
                            invoice_text = extract_text_from_image(temp_invoice_path)
                        finally:
                            # Clean up temporary file
                            if os.path.exists(temp_invoice_path):
                                os.remove(temp_invoice_path)
                        
                        # Perform comparison analysis
                        comparison_result = st.session_state.comparator.compare_results(
                            asset_result, 
                            invoice_text
                        )
                        
                        # Store results
                        form_result = {
                            "asset_filename": asset_file.name,
                            "invoice_filename": invoice_file.name,
                            "asset_classification": asset_result,
                            "invoice_ocr": invoice_text,
                            "comparison_analysis": comparison_result,
                            "asset_image": asset_image,
                            "invoice_image": invoice_image,
                            "processed_at": st.session_state.get("current_time", "Now")
                        }
                        
                        st.session_state.form_results.append(form_result)
                        
                        # Display immediate results
                        st.success("✅ Loan application processed successfully!")
                        
                        # Show results in three columns
                        col_a, col_b, col_c = st.columns(3)
                        
                        with col_a:
                            st.subheader("🏷️ Asset Classification")
                            if "error" in asset_result:
                                st.error(f"❌ Error: {asset_result['error']}")
                            else:
                                is_suspicious = asset_result.get("is_suspicious", False)
                                
                                st.write(f"**Category:** {asset_result.get('category', 'Unknown')}")
                                st.write(f"**Sub-Category:** {asset_result.get('sub_category', 'Unknown')}")
                                st.write(f"**Confidence:** {asset_result.get('confidence', 'unknown')}")
                                st.write(f"**Status:** {'🚨 Suspicious' if is_suspicious else '✅ Valid'}")
                                st.write(f"**Reason:** {asset_result.get('reason', 'N/A')}")
                        
                        with col_b:
                            st.subheader("📄 Invoice OCR")
                            try:
                                invoice_data = json.loads(invoice_text) if isinstance(invoice_text, str) and invoice_text.startswith('{') else {"extracted_text": invoice_text}
                                if "category" in invoice_data:
                                    st.write(f"**Category:** {invoice_data.get('category', 'Unknown')}")
                                if "document_type" in invoice_data:
                                    st.write(f"**Document Type:** {invoice_data.get('document_type', 'Unknown')}")
                                if "is_relevant" in invoice_data:
                                    st.write(f"**Relevant:** {'✅ Yes' if invoice_data.get('is_relevant') else '❌ No'}")
                                if "confidence" in invoice_data:
                                    st.write(f"**Confidence:** {invoice_data.get('confidence', 0)}")
                                
                                with st.expander("View Extracted Text"):
                                    extracted = invoice_data.get('extracted_text', invoice_text)
                                    st.text_area("OCR Output", extracted, height=200, disabled=True, label_visibility="collapsed")
                            except:
                                st.text_area("OCR Output", invoice_text, height=250, disabled=True, label_visibility="collapsed")
                        
                        with col_c:
                            st.subheader("🔍 Comparison Analysis")
                            if "error" in comparison_result:
                                st.error(f"❌ Error: {comparison_result['error']}")
                            else:
                                status = comparison_result.get("overall_status", "UNKNOWN")
                                risk = comparison_result.get("risk_level", "UNKNOWN")
                                score = comparison_result.get("consistency_score", 0)
                                
                                # Status indicator
                                if status == "APPROVED":
                                    st.success(f"✅ **Status:** {status}")
                                elif status == "REJECTED":
                                    st.error(f"❌ **Status:** {status}")
                                else:
                                    st.warning(f"⚠️ **Status:** {status}")
                                
                                st.write(f"**Risk Level:** {risk}")
                                st.write(f"**Consistency Score:** {score:.2f}")
                                st.write(f"**Category Match:** {'✅ Yes' if comparison_result.get('category_match') else '❌ No'}")
                                
                                # Key findings
                                findings = comparison_result.get("key_findings", [])
                                if findings:
                                    st.write("**Key Findings:**")
                                    for finding in findings[:3]:  # Show top 3
                                        st.write(f"• {finding}")
                                
                                # Recommendation
                                recommendation = comparison_result.get("recommendation", "")
                                if recommendation:
                                    with st.expander("View Recommendation"):
                                        st.write(recommendation)
                        
                    except Exception as e:
                        st.error(f"❌ Error processing application: {str(e)}")

with tab2:
    st.subheader("📜 Processing History")
    
    if not st.session_state.form_results:
        st.info("No loan applications processed yet. Use the form to get started!")
    else:
        # Summary statistics
        total = len(st.session_state.form_results)
        suspicious_count = sum(1 for item in st.session_state.form_results 
                              if item["asset_classification"].get("is_suspicious", False))
        approved_count = sum(1 for item in st.session_state.form_results 
                           if item.get("comparison_analysis", {}).get("overall_status") == "APPROVED")
        rejected_count = sum(1 for item in st.session_state.form_results 
                           if item.get("comparison_analysis", {}).get("overall_status") == "REJECTED")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Applications", total)
        with col2:
            st.metric("Approved", approved_count, delta=None)
        with col3:
            st.metric("Rejected", rejected_count, delta=None)
        with col4:
            st.metric("Suspicious Assets", suspicious_count, delta=None)
        
        st.divider()
        
        # Display history
        for idx, item in enumerate(reversed(st.session_state.form_results)):
            # Get comparison status for the header
            comparison = item.get("comparison_analysis", {})
            status = comparison.get("overall_status", "UNKNOWN")
            status_icon = "✅" if status == "APPROVED" else "❌" if status == "REJECTED" else "⚠️"
            
            with st.expander(f"{status_icon} Application #{len(st.session_state.form_results) - idx}: {item['asset_filename']} & {item['invoice_filename']} - {status}", expanded=(idx == 0)):
                # Show comparison results first
                if "comparison_analysis" in item:
                    st.subheader("🔍 Loan Application Analysis")
                    comp = item["comparison_analysis"]
                    
                    col_status1, col_status2, col_status3 = st.columns(3)
                    with col_status1:
                        status = comp.get("overall_status", "UNKNOWN")
                        if status == "APPROVED":
                            st.success(f"**Final Status:** {status}")
                        elif status == "REJECTED":
                            st.error(f"**Final Status:** {status}")
                        else:
                            st.warning(f"**Final Status:** {status}")
                    
                    with col_status2:
                        risk = comp.get("risk_level", "UNKNOWN")
                        st.write(f"**Risk Level:** {risk}")
                        st.write(f"**Consistency:** {comp.get('consistency_score', 0):.2f}")
                    
                    with col_status3:
                        st.write(f"**Category Match:** {'✅ Yes' if comp.get('category_match') else '❌ No'}")
                        st.write(f"**Confidence:** {comp.get('confidence', 0):.2f}")
                    
                    # Key findings and recommendation
                    if comp.get("key_findings"):
                        st.write("**Key Findings:**")
                        for finding in comp.get("key_findings", []):
                            st.write(f"• {finding}")
                    
                    if comp.get("recommendation"):
                        st.info(f"**Recommendation:** {comp.get('recommendation')}")
                    
                    st.divider()
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### 📷 Asset Image")
                    st.image(item["asset_image"], width='stretch')
                    st.caption(item["asset_filename"])
                    
                    # Asset classification results
                    result = item["asset_classification"]
                    if "error" in result:
                        st.error(f"❌ Error: {result['error']}")
                    else:
                        is_suspicious = result.get("is_suspicious", False)
                        
                        st.write(f"**Category:** {result.get('category', 'Unknown')}")
                        st.write(f"**Sub-Category:** {result.get('sub_category', 'Unknown')}")
                        st.write(f"**Confidence:** {result.get('confidence', 'unknown').upper()}")
                        st.write(f"**Status:** {'🚨 Suspicious' if is_suspicious else '✅ Valid'}")
                        st.write(f"**Reason:** {result.get('reason', 'N/A')}")
                
                with col2:
                    st.markdown("#### 📄 Invoice/Receipt")
                    st.image(item["invoice_image"], width='stretch')
                    st.caption(item["invoice_filename"])
                    
                    # OCR results
                    st.write("**Extracted Text:**")
                    if item["invoice_ocr"].strip():
                        st.text_area("OCR Output", item["invoice_ocr"], height=200, disabled=True, key=f"ocr_{idx}")
                    else:
                        st.info("No text extracted from this invoice")

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem;'>
    <p>🏦 Loan Asset Classifier & Invoice Processor powered by Google Gemini</p>
    <p style='font-size: 0.8rem;'>Process loan applications with automatic asset classification and invoice OCR</p>
</div>
""", unsafe_allow_html=True)