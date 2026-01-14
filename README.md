#  Loan Utilization Tracking System

##  Project Overview
The **Loan Utilization Tracking System** is an AI-based application that helps verify whether loan or subsidy money is being used for its intended purpose.

In this project, a user uploads:
- An **asset image** (what was purchased using the loan)
- An **invoice or receipt image** (proof of purchase)

The system then:
- Classifies the asset type
- Extracts text from the invoice using OCR
- Compares both results
- Generates a **risk score and approval status**

This helps automate a process that is usually manual, slow, and prone to misuse.

---

##  Why This Project
In real-world scenarios, governments and banks face difficulty in tracking how loan money is actually used after disbursement.

Common problems include:
- Funds being misused
- Fake or irrelevant bills
- Manual verification taking too much time

This project explores how **AI + OCR + automation** can make loan monitoring:
- Faster
- More transparent
- Data-driven

---

##  What This System Does
- Accepts asset images and invoice images from users
- Classifies the asset into loan categories (Agriculture, Education, Business, Housing, Vehicle)
- Extracts text from invoices using OCR
- Checks whether the invoice matches the asset purpose
- Flags suspicious or mismatched cases
- Displays results in an easy-to-use dashboard

---


## 📂 Project Structure
Loan-Utilization-Tracking/
│
├── streamlit_ui.py # Main Streamlit dashboard
├── loan_classifier.py # Asset image classification logic
├── loan_comparator.py # Compares asset & invoice results
├── ocr_extractor.py # OCR text extraction from invoices
├── .env # Environment variables (API keys)
├── .gitignore # Files to ignore in GitHub
├── pyproject.toml # Project dependencies
├── README.md # Project documentation


##  How the System Works
1. User uploads an **asset image** and an **invoice image**
2. Asset image is classified using AI
3. Invoice image text is extracted using OCR
4. Both results are compared
5. System generates:
   - Approval / Rejection status
   - Risk level
   - Consistency score
6. Results are displayed on the dashboard

---

##  How to Run the Project
1. Clone or download the repository
2. Create a `.env` file and add your API key:
GEMINI_API_KEY=your_api_key_here
3. Install required dependencies
4. Run the Streamlit app:
streamlit run streamlit_ui.py


---

##  Output
- Asset category and confidence score
- Extracted invoice text
- Approval or rejection decision
- Risk and consistency score
- Highlighted suspicious cases

---

##  Expected Outcome
A working prototype that demonstrates how AI can be used to:
- Monitor loan usage
- Reduce misuse of funds
- Support transparent decision-making

---

##  Future Improvements
- User authentication
- Government officer dashboard
- Database integration
- Power BI or advanced analytics
- Mobile app version

---

## Conclusion
This project demonstrates a practical use of AI in financial monitoring and governance.  
It shows how automation can help reduce fraud, improve transparency, and save manual effort.
