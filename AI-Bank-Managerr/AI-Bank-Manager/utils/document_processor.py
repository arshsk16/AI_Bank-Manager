import cv2
import numpy as np
import re
import os
from PIL import Image
from datetime import datetime

# Try to import pytesseract, but provide fallback if not available
try:
    import pytesseract
    PYTESSERACT_AVAILABLE = True
except ImportError:
    PYTESSERACT_AVAILABLE = False
    print("Warning: pytesseract module not found, using fallback for document processing.")
    print("For better document processing, install pytesseract: pip install pytesseract")

def validate_document_quality(img):
    """
    Validate the quality of the uploaded document image.

    Args:
        img: Input image

    Returns:
        tuple: (is_valid, list of issues)
    """
    issues = []

    # Check image dimensions
    height, width = img.shape[:2]
    if width < 800 or height < 600:
        issues.append("Image resolution is too low. Please upload a higher quality image (minimum 800x600 pixels)")

    # Check if image is too blurry
    laplacian_var = cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
    if laplacian_var < 100:
        issues.append("Image is too blurry. Please provide a clearer photo")

    # Check brightness
    brightness = np.mean(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    if brightness < 50:
        issues.append("Image is too dark. Please retake the photo in better lighting")
    elif brightness > 200:
        issues.append("Image is too bright. Please retake the photo with less glare")

    # Check for excessive rotation (more relevant for mobile uploads)
    angle = get_image_rotation_angle(img)
    if abs(angle) > 10: # if rotated more than 10 degrees
        issues.append("Document seems rotated. Please ensure document is upright.")

    return len(issues) == 0, issues

def get_document_requirements(doc_type):
    """
    Get document-specific requirements and guidelines.

    Args:
        doc_type (str): Type of document

    Returns:
        dict: Document requirements and guidelines
    """
    requirements = {
        "aadhaar": {
            "mandatory_fields": ["aadhaar_number", "name", "dob"],
            "format": "12-digit number with spaces (e.g., 1234 5678 9012)",
            "guidelines": [
                "Ensure all four corners of the card are visible",
                "Both front and back sides should be clear and readable",
                "QR code should be clearly visible (if present)",
                "No part of the Aadhaar number should be masked"
            ]
        },
        "pan": {
            "mandatory_fields": ["pan_number", "name", "father_name", "dob"],
            "format": "10 characters (AAAAA0000A)",
            "guidelines": [
                "PAN card should be original (not a photocopy)",
                "All text should be clearly readable",
                "Card should not be damaged or laminated with reflective material",
                "Photo on PAN card should be clear"
            ]
        },
        "income_proof": {
            "mandatory_fields": ["name", "monthly_income", "period", "company"],
            "format": "Salary slip or IT return document",
            "guidelines": [
                "Document should be on company letterhead (for salary slip)",
                "All amounts should be clearly visible",
                "Period/date of income should be recent (within last 3 months for salary slip)",
                "Company name and employee details should be clearly visible"
            ]
        }
    }
    return requirements.get(doc_type, {})

def process_document(doc_path, doc_type):
    """
    Process a document image and extract relevant information.

    Args:
        doc_path (str): Path to the document image
        doc_type (str): Type of document ('aadhaar', 'pan', 'income_proof')

    Returns:
        tuple: (is_valid, extracted_text, extracted_info, validation_result)
    """
    try:
        # Read the document image
        img = cv2.imread(doc_path)

        if img is None:
            return False, "", {}, {
                "status": "error",
                "message": "Unable to read the uploaded file. Please ensure it's a valid image.",
                "issues": ["Invalid or corrupted image file"]
            }

        # Validate document quality
        is_quality_ok, quality_issues = validate_document_quality(img)
        if not is_quality_ok:
            return False, "", {}, {
                "status": "quality_issues",
                "message": "Document image quality issues detected",
                "issues": quality_issues
            }

        # Deskew the image before further processing
        deskewed_img = deskew_image(img)

        # Preprocess the image for better OCR
        processed_img = preprocess_image(deskewed_img)

        # Extract text using OCR if available, or use mock data for demo
        if PYTESSERACT_AVAILABLE:
            extracted_text = pytesseract.image_to_string(processed_img)
        else:
            extracted_text = generate_mock_document_text(doc_type)

        # Extract information based on document type
        is_valid, extracted_info = extract_document_info(extracted_text, doc_type)

        # Validate extracted information against requirements
        validation_result = validate_extracted_info(extracted_info, doc_type)

        return is_valid, extracted_text, extracted_info, validation_result

    except Exception as e:
        return False, "", {}, {
            "status": "error",
            "message": f"Error processing document: {str(e)}",
            "issues": ["Unexpected error during document processing"]
        }

def validate_extracted_info(extracted_info, doc_type):
    """
    Validate extracted information against document requirements.

    Args:
        extracted_info (dict): Extracted document information
        doc_type (str): Type of document

    Returns:
        dict: Validation result with status and issues
    """
    requirements = get_document_requirements(doc_type)
    if not requirements:
        return {"status": "error", "message": "Unknown document type"}

    issues = []
    missing_fields = []

    # Check mandatory fields
    for field in requirements["mandatory_fields"]:
        if field not in extracted_info or extracted_info[field] == "Not found":
            missing_fields.append(field.replace("_", " ").title())

    if missing_fields:
        issues.append(f"Missing required fields: {', '.join(missing_fields)}")

    # Document-specific validations
    if doc_type == "aadhaar":
        if "aadhaar_number" in extracted_info:
            aadhaar_num = extracted_info["aadhaar_number"]
            if aadhaar_num != "Not found" and not re.match(r'^\d{4}\s\d{4}\s\d{4}$', aadhaar_num):
                issues.append("Invalid Aadhaar number format. Expected format: 1234 5678 9012")

    elif doc_type == "pan":
        if "pan_number" in extracted_info:
            pan_num = extracted_info["pan_number"]
            if pan_num != "Not found" and not re.match(r'^[A-Z]{5}[0-9]{4}[A-Z]$', pan_num):
                issues.append("Invalid PAN number format. Expected format: AAAAA0000A")

    elif doc_type == "income_proof":
        if "monthly_income" in extracted_info and extracted_info["monthly_income"] == 0:
            issues.append("Could not detect income amount in the document")
        if "period" in extracted_info and extracted_info["period"] == "Not found":
            issues.append("Could not determine the period/date of the income proof")

    # Determine validation status
    if not issues:
        return {
            "status": "success",
            "message": "Document validation successful",
            "requirements": requirements["guidelines"]
        }
    else:
        return {
            "status": "validation_failed",
            "message": "Document validation failed",
            "issues": issues,
            "requirements": requirements["guidelines"]
        }

def preprocess_image(img):
    """
    Preprocess an image for better OCR.

    Args:
        img: Input image

    Returns:
        Preprocessed image
    """
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )

    # Denoise
    denoised = cv2.fastNlMeansDenoising(thresh, None, 10, 7, 21)

    return denoised


def extract_document_info(text, doc_type):
    """
    Extract relevant information from document text based on document type.

    Args:
        text (str): Extracted text from document
        doc_type (str): Type of document ('aadhaar', 'pan', 'income_proof')

    Returns:
        tuple: (is_valid, extracted_info)
    """
    if doc_type == "aadhaar":
        return extract_aadhaar_info(text)
    elif doc_type == "pan":
        return extract_pan_info(text)
    elif doc_type == "income_proof":
        return extract_income_proof_info(text)
    else:
        return False, {}


def extract_aadhaar_info(text):
    """Extract information from Aadhaar card"""
    # Normalize text
    text = text.lower()

    # Check if it's an Aadhaar card by looking for key patterns
    is_aadhaar = "aadhaar" in text or "unique identification" in text or "government of india" in text

    if not is_aadhaar:
        return False, {}

    # Extract Aadhaar number
    aadhaar_pattern = r'\d{4}\s\d{4}\s\d{4}'
    aadhaar_matches = re.findall(aadhaar_pattern, text)
    aadhaar_number = aadhaar_matches[0] if aadhaar_matches else "Not found"

    # Extract name (typically after "to " or on a line with capital letters)
    name = "Not found"
    lines = text.split('\n')
    for i, line in enumerate(lines):
        if "to " in line.lower():
            name_parts = line.split("to ", 1)
            if len(name_parts) > 1:
                name = name_parts[1].strip()
                break

    # Extract DOB
    dob_pattern = r'\b\d{2}/\d{2}/\d{4}\b'
    dob_matches = re.findall(dob_pattern, text)
    dob = dob_matches[0] if dob_matches else "Not found"

    # Extract gender
    gender = "Not found"
    if "male" in text:
        gender = "Male"
    elif "female" in text:
        gender = "Female"

    extracted_info = {
        "document_type": "Aadhaar Card",
        "aadhaar_number": aadhaar_number,
        "name": name,
        "dob": dob,
        "gender": gender
    }

    return True, extracted_info


def extract_pan_info(text):
    """Extract information from PAN card"""
    # Normalize text
    text = text.lower()

    # Check if it's a PAN card
    is_pan = "income tax" in text or "permanent account number" in text

    if not is_pan:
        return False, {}

    # Extract PAN number (format: AAAAA0000A)
    pan_pattern = r'[a-zA-Z]{5}[0-9]{4}[a-zA-Z]{1}'
    pan_matches = re.findall(pan_pattern, text)
    pan_number = pan_matches[0].upper() if pan_matches else "Not found"

    # Extract name
    name = "Not found"
    lines = text.split('\n')
    for i, line in enumerate(lines):
        if "name" in line.lower():
            if i+1 < len(lines):
                name = lines[i+1].strip()
                break

    # Extract father's name
    father_name = "Not found"
    for i, line in enumerate(lines):
        if "father" in line.lower():
            if i+1 < len(lines):
                father_name = lines[i+1].strip()
                break

    # Extract DOB
    dob_pattern = r'\b\d{2}/\d{2}/\d{4}\b'
    dob_matches = re.findall(dob_pattern, text)
    dob = dob_matches[0] if dob_matches else "Not found"

    extracted_info = {
        "document_type": "PAN Card",
        "pan_number": pan_number,
        "name": name,
        "father_name": father_name,
        "dob": dob
    }

    return True, extracted_info


def extract_income_proof_info(text):
    """Extract information from income proof (salary slip or IT returns)"""
    # Normalize text
    text = text.lower()

    # Determine document type
    is_salary_slip = "salary" in text or "payslip" in text or "pay slip" in text
    is_it_return = "income tax return" in text or "itr" in text

    if not (is_salary_slip or is_it_return):
        return False, {}

    doc_subtype = "Salary Slip" if is_salary_slip else "Income Tax Return"

    # Extract name
    name = "Not found"
    lines = text.split('\n')
    for i, line in enumerate(lines):
        if "name" in line.lower():
            # Try to get name from the same line or next line
            name_parts = line.split("name", 1)
            if len(name_parts) > 1 and len(name_parts[1].strip()) > 0:
                name = name_parts[1].strip()
                break
            elif i+1 < len(lines):
                name = lines[i+1].strip()
                break

    # Extract income info
    income = 0
    income_patterns = [
        r'gross\s+salary\s*:?\s*rs\.?\s*(\d+[\d,.]*)',
        r'net\s+salary\s*:?\s*rs\.?\s*(\d+[\d,.]*)',
        r'total\s+income\s*:?\s*rs\.?\s*(\d+[\d,.]*)',
        r'gross\s+income\s*:?\s*rs\.?\s*(\d+[\d,.]*)',
        r'salary\s*:?\s*rs\.?\s*(\d+[\d,.]*)',
        r'income\s*:?\s*rs\.?\s*(\d+[\d,.]*)'
    ]

    for pattern in income_patterns:
        income_matches = re.findall(pattern, text)
        if income_matches:
            # Clean the matched income string and convert to int
            income_str = income_matches[0].replace(',', '').replace('.', '')
            try:
                income = int(income_str)
                break
            except:
                continue

    # Extract period/year
    period = "Not found"
    period_patterns = [
        r'for\s+the\s+month\s+of\s+([a-zA-Z]+\s+\d{4})',
        r'period\s*:?\s*([a-zA-Z]+\s+\d{4})',
        r'assessment\s+year\s+(\d{4}-\d{2,4})',
        r'financial\s+year\s+(\d{4}-\d{2,4})',
        r'\b(FY\s+\d{4}-\d{2,4})\b',
        r'\b(AY\s+\d{4}-\d{2,4})\b',
    ]

    for pattern in period_patterns:
        period_matches = re.findall(pattern, text)
        if period_matches:
            period = period_matches[0]
            break

    # Extract company/employer name
    company = "Not found"
    company_patterns = [
        r'company\s+name\s*:?\s*([^\n]+)',
        r'employer\s+name\s*:?\s*([^\n]+)',
        r'organization\s*:?\s*([^\n]+)',
    ]

    for pattern in company_patterns:
        company_matches = re.findall(pattern, text, re.IGNORECASE)
        if company_matches:
            company = company_matches[0].strip()
            break

    extracted_info = {
        "document_type": "Income Proof",
        "proof_type": doc_subtype,
        "name": name,
        "monthly_income" if is_salary_slip else "annual_income": income,
        "period": period,
        "company": company
    }

    return True, extracted_info

def detect_document_boundaries(img):
    """
    Detect document boundaries in an image.

    Args:
        img: Input image (BGR format)

    Returns:
        tuple: (success, corners, score) where corners are the detected document corners
               and score is a confidence measure between 0-1
    """
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Apply edge detection
        edges = cv2.Canny(blurred, 75, 200)

        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return False, None, 0.0

        # Sort contours by area in descending order
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        # Get the largest contour
        largest_contour = contours[0]

        # Calculate area ratio
        img_area = img.shape[0] * img.shape[1]
        contour_area = cv2.contourArea(largest_contour)
        area_ratio = contour_area / img_area

        # If contour is too small or too large, reject it
        if area_ratio < 0.2 or area_ratio > 0.95:
            return False, None, 0.0

        # Approximate the contour
        peri = cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, 0.02 * peri, True)

        # If we don't have 4 corners, try to find the best 4 corners
        if len(approx) != 4:
            # Use minimum area rectangle if approximation didn't yield 4 corners
            rect = cv2.minAreaRect(largest_contour)
            box = cv2.boxPoints(rect)
            approx = np.int0(box)

        # Order points in clockwise order: top-left, top-right, bottom-right, bottom-left
        pts = approx.reshape(4, 2)
        rect = np.zeros((4, 2), dtype="float32")

        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]  # Top-left: smallest sum
        rect[2] = pts[np.argmax(s)]  # Bottom-right: largest sum

        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]  # Top-right: smallest difference
        rect[3] = pts[np.argmax(diff)]  # Bottom-left: largest difference

        # Calculate confidence score based on multiple factors
        # 1. How close to a rectangle (aspect ratio)
        width = np.linalg.norm(rect[1] - rect[0])
        height = np.linalg.norm(rect[3] - rect[0])
        aspect_ratio = min(width, height) / max(width, height)
        aspect_score = min(aspect_ratio / 0.6, 1.0)  # Most documents have ratio around 0.6-0.7

        # 2. Area coverage
        area_score = min(area_ratio / 0.5, 1.0)

        # 3. Angle alignment (should be mostly aligned with image axes)
        angles = []
        for i in range(4):
            p1 = rect[i]
            p2 = rect[(i + 1) % 4]
            angle = abs(np.degrees(np.arctan2(p2[1] - p1[1], p2[0] - p1[0])) % 90)
            angles.append(min(angle, 90 - angle))
        angle_score = 1 - (sum(angles) / (4 * 45))  # 45 degrees is maximum deviation

        # Combine scores
        confidence = (aspect_score * 0.4 + area_score * 0.3 + angle_score * 0.3)

        return True, rect, confidence

    except Exception as e:
        print(f"Error in document detection: {str(e)}")
        return False, None, 0.0

def extract_document(img, corners):
    """
    Extract and rectify document from image using detected corners.

    Args:
        img: Input image
        corners: Four corners of the document

    Returns:
        Extracted and rectified document image
    """
    # Convert corners to float32
    corners = corners.astype("float32")

    # Compute the width and height of the rectified document
    width_a = np.linalg.norm(corners[1] - corners[0])
    width_b = np.linalg.norm(corners[2] - corners[3])
    max_width = max(int(width_a), int(width_b))

    height_a = np.linalg.norm(corners[3] - corners[0])
    height_b = np.linalg.norm(corners[2] - corners[1])
    max_height = max(int(height_a), int(height_b))

    # Define the destination points for perspective transform
    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]
    ], dtype="float32")

    # Calculate perspective transform matrix and apply it
    M = cv2.getPerspectiveTransform(corners, dst)
    warped = cv2.warpPerspective(img, M, (max_width, max_height))

    return warped

def draw_document_boundaries(img, corners):
    """
    Draw detected document boundaries on the image.

    Args:
        img: Input image
        corners: Four corners of the document

    Returns:
        Image with drawn boundaries
    """
    # Create a copy of the image
    output = img.copy()

    # Draw the contour
    cv2.drawContours(output, [corners.astype(int)], -1, (0, 255, 0), 2)

    # Draw corner points
    for corner in corners.astype(int):
        cv2.circle(output, tuple(corner), 5, (0, 0, 255), -1)

    return output

def is_document_clear(img):
    """
    Check if the document image is clear enough for processing.

    Args:
        img: Input image

    Returns:
        tuple: (is_clear, issues)
    """
    issues = []

    # Check image dimensions
    height, width = img.shape[:2]
    if width < 800 or height < 600:
        issues.append("Image resolution is too low")

    # Check blurriness
    laplacian_var = cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
    if laplacian_var < 100:
        issues.append("Image is too blurry")

    # Check brightness
    brightness = np.mean(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    if brightness < 50:
        issues.append("Image is too dark")
    elif brightness > 200:
        issues.append("Image is too bright")

    return len(issues) == 0, issues

def get_image_rotation_angle(image):
    """
    Detect the rotation angle of an image.
    This is a simplified approach and might need refinement for complex cases.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)

    if lines is None:
        return 0 # No lines detected, assume no rotation

    angles = []
    for line in lines:
        for x1, y1, x2, y2 in line:
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            angles.append(angle)

    if not angles:
        return 0

    # Take median angle to be robust to outliers
    median_angle = np.median(angles)
    return median_angle

def deskew_image(img):
    """
    Deskew an image (rotate to correct minor rotations).
    """
    angle = get_image_rotation_angle(img)
    height, width = img.shape[:2]
    center = (width // 2, height // 2)
    M = cv2.getRotationMatrix2D(center, -angle, 1.0) # Negative angle to counter rotation
    rotated_img = cv2.warpAffine(img, M, (width, height), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated_img
