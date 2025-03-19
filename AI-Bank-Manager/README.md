# AI Bank Manager

A video-based loan assistance application that provides a seamless, branch-like experience for users applying for loans digitally.

## Features

1. **Virtual AI Branch Manager**
   - Users interact with a virtual bank manager through the application
   - Structured guidance for loan applications

2. **Identity Verification**
   - Basic facial verification to ensure the same applicant continues throughout the process
   - Secure identity management

3. **Document Submission & Processing**
   - Upload or capture images of Aadhaar, PAN, and income proof documents
   - Automated extraction of key details from documents (name, DOB, income, etc.)

4. **Loan Interview with Video Responses**
   - Interactive video-based interview process
   - Users record video responses to loan-related questions

5. **Loan Eligibility & Decisioning**
   - Rule-based system evaluates loan eligibility
   - Provides instant feedback with approval, rejection, or requests for more information

## Setup Instructions

### Prerequisites

- Python 3.7+
- Tesseract OCR (for document processing)
- Webcam access for video recording

### Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd AI-Bank-Manager
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Install Tesseract OCR:
   - For macOS: `brew install tesseract`
   - For Ubuntu: `sudo apt install tesseract-ocr`
   - For Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki

4. Run the application:
   ```
   streamlit run app.py
   ```

## Usage

1. Fill in personal information on the welcome screen
2. Complete identity verification using your webcam
3. Upload or take photos of required documents
4. Record video responses to loan-related questions
5. Review loan eligibility and decision
6. Download the application summary

## System Requirements

- Modern web browser with webcam access
- Internet connection
- Minimum 4GB RAM
- 50MB free disk space

## Security Features

- Facial verification for identity confirmation
- Document validation for authenticity
- Secure data handling throughout the process

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- Streamlit for the interactive web application framework
- OpenCV and face_recognition for facial verification
- Tesseract OCR for document processing