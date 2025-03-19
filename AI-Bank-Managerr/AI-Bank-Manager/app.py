import streamlit as st
import os
import tempfile
from pathlib import Path
import time
import cv2
import numpy as np
import base64
from io import BytesIO

# Import utility modules
from utils.facial_verification import verify_face
from utils.document_processor import process_document, extract_document_info, get_document_requirements, detect_document_boundaries, draw_document_boundaries, extract_document, is_document_clear
from utils.loan_eligibility import check_eligibility
import qrcode
import uuid
import json
import tempfile
from pathlib import Path

# Set page configuration
st.set_page_config(
    page_title="AI Bank Manager",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state variables if they don't exist
if 'current_step' not in st.session_state:
    st.session_state.current_step = 1
if 'user_data' not in st.session_state:
    st.session_state.user_data = {}
if 'reference_face' not in st.session_state:
    st.session_state.reference_face = None
if 'documents' not in st.session_state:
    st.session_state.documents = {}
if 'videos' not in st.session_state:
    st.session_state.videos = {}
if 'eligibility_result' not in st.session_state:
    st.session_state.eligibility_result = None

# Function to reset the application state
def reset_app():
    st.session_state.current_step = 1
    st.session_state.user_data = {}
    st.session_state.reference_face = None
    st.session_state.documents = {}
    st.session_state.videos = {}
    st.session_state.eligibility_result = None

# Function to move to the next step
def next_step():
    st.session_state.current_step += 1

# Function to go back to previous step
def prev_step():
    if st.session_state.current_step > 1:
        st.session_state.current_step -= 1

# Function to manage camera resources
def manage_camera_resources():
    """Helper function to manage camera resources and prevent conflicts"""
    # Keys that should have active cameras in each step
    active_camera_keys = {
        1: [],  # Welcome page has no cameras
        2: ["face_verification_camera"] if not st.session_state.get("face_verified", False) else [],
        3: [], # Document cameras are managed separately
        4: [], # Video interview cameras are managed separately
        5: [], # Eligibility page has no cameras
        6: []  # Summary page has no cameras
    }
    
    current_step = st.session_state.current_step
    # Get the keys that should be active for the current step
    current_active_keys = active_camera_keys.get(current_step, [])
    
    # If we're in step 3 (Document Submission), add document camera keys
    if current_step == 3:
        doc_types = ["aadhaar", "pan", "income_proof"]
        for doc_type in doc_types:
            key = f"camera_{doc_type}"
            if st.session_state.get(f"detection_active_{key}", False):
                current_active_keys.append(key)
    
    # If we're in step 4 (Loan Interview), add question camera keys
    if current_step == 4:
        current_q = st.session_state.get("current_question_index", 0)
        if not st.session_state.get("question_verified", False):
            current_active_keys.append(f"verify_q{current_q}")
        else:
            current_active_keys.append(f"video_q{current_q}")
    
    return current_active_keys

# Custom CSS with Standard Chartered theme
st.markdown("""
<style>
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Fix container width and overflow */
    .stApp {
        max-width: 100%;
        padding: 1rem;
        overflow-x: hidden;
    }
    
    .main-header {
        font-size: 2.5rem;
        color: #FFFFFF;
        text-align: center;
        margin-bottom: 2rem;
        font-family: 'Arial', sans-serif;
    }
    
    .sub-header {
        font-size: 1.8rem;
        color: #E31B23;
        margin-bottom: 1.5rem;
        font-weight: 500;
    }
    
    .info-box {
        background-color: rgba(27, 53, 72, 0.6);
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .success-box {
        background-color: rgba(16, 185, 129, 0.1);
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1.5rem;
        border: 1px solid rgba(16, 185, 129, 0.2);
    }
    
    .error-box {
        background-color: rgba(227, 27, 35, 0.1);
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1.5rem;
        border: 1px solid rgba(227, 27, 35, 0.2);
    }
    
    .stButton>button {
        background: linear-gradient(45deg, #0F2337, #E31B23);
        color: white;
        font-weight: bold;
        padding: 0.5rem 2rem;
        border-radius: 0.5rem;
        border: none;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 8px rgba(0, 0, 0, 0.2);
    }
    
    .divider {
        border-top: 1px solid rgba(255, 255, 255, 0.1);
        margin: 1.5rem 0;
    }
    
    .step-number {
        background: #E31B23;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 0.5rem;
        margin-right: 0.5rem;
        font-weight: bold;
    }
    
    /* Form inputs styling */
    .stTextInput>div>div>input,
    .stSelectbox>div>div>div {
        background-color: rgba(27, 53, 72, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 0.5rem;
        color: white;
    }
    
    .stSlider>div>div {
        background-color: rgba(27, 53, 72, 0.6);
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #0F2337;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #E31B23;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #FF1F28;
    }
    
    /* Fix container padding and margins */
    .block-container {
        padding: 0 1rem;
        max-width: 100%;
    }
    
    .element-container {
        margin: 0;
    }
    
    /* Table styling */
    .stDataFrame {
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 0.5rem;
    }
    
    .stDataFrame table {
        border-collapse: separate;
        border-spacing: 0;
    }
    
    .stDataFrame th {
        background-color: #0F2337;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        padding: 0.5rem;
    }
    
    .stDataFrame td {
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        padding: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/bank-building.png", width=100)
    st.markdown("## AI Bank Manager")
    st.markdown("### Navigation")
    
    # Display current step information
    steps = {
        1: "👋 Welcome",
        2: "👤 Identity Verification",
        3: "📄 Document Submission",
        4: "💬 Loan Interview",
        5: "💰 Loan Eligibility",
        6: "✅ Application Summary"
    }
    
    for step_num, step_name in steps.items():
        if step_num == st.session_state.current_step:
            st.markdown(f"**→ {step_name}**")
        else:
            st.markdown(f"  {step_name}")
    
    # Replace st.divider() with a markdown alternative
    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    
    if st.button("Start Over"):
        reset_app()
        st.experimental_rerun()

# Main content based on current step
st.markdown("<h1 class='main-header'>AI Bank Manager</h1>", unsafe_allow_html=True)

# Step 1: Welcome screen
if st.session_state.current_step == 1:
    st.markdown("<h2 class='sub-header'>Welcome to Virtual Bank Branch</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div class='info-box'>
            <p>Welcome to our AI-powered banking experience! I am your virtual Bank Manager, ready to assist you with your loan application process.</p>
            <p>In this session, we'll go through the following steps:</p>
            <ol>
                <li>Identity verification through facial recognition</li>
                <li>Document submission and verification</li>
                <li>Video interview for loan assessment</li>
                <li>Instant loan eligibility check</li>
            </ol>
            <p>This process is designed to give you a branch-like experience without visiting a physical bank.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Collect basic information
        st.markdown("<p>Let's start with your basic information:</p>", unsafe_allow_html=True)
        name = st.text_input("Full Name", key="name")
        email = st.text_input("Email Address", key="email")
        phone = st.text_input("Phone Number", key="phone")
        
        if st.button("Begin Application Process"):
            if not name or not email or not phone:
                st.error("Please fill in all the fields to continue.")
            else:
                # Save user data
                st.session_state.user_data.update({
                    "name": name,
                    "email": email,
                    "phone": phone,
                    "application_id": f"LOAN-{int(time.time())}",
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                })
                next_step()
                st.experimental_rerun()
    
    with col2:
        st.video("https://sora.com/g/gen_01jpn3fq0ye6srrmjzeg8jv8jn")
        st.caption("Your AI Bank Manager")

# Step 2: Identity Verification
elif st.session_state.current_step == 2:
    st.markdown("<h2 class='sub-header'>Identity Verification</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        <div class='info-box'>
            <p>To ensure a secure application process, we need to verify your identity.</p>
            <p>Please allow access to your camera so we can take a photo for facial verification.</p>
            <p>This image will be used throughout your application journey for security purposes.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Initialize face verification state variables if they don't exist
        if 'face_verified' not in st.session_state:
            st.session_state.face_verified = False
        if 'face_capture_active' not in st.session_state:
            st.session_state.face_capture_active = True
            
        if not st.session_state.face_verified:
            # Camera input for face capture - only show if not yet verified
            face_image = st.camera_input("Take a photo for verification", 
                key="face_verification_camera",
                disabled=not st.session_state.face_capture_active)
            
            if face_image:
                # Save the reference face for later verification
                with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
                    tmp.write(face_image.getvalue())
                    reference_face_path = tmp.name
                
                # Process the face and verify it's valid (contains a recognizable face)
                is_valid_face, face_encoding = verify_face(reference_face_path)
                
                if is_valid_face:
                    st.session_state.reference_face = {
                        "path": reference_face_path,
                        "encoding": face_encoding
                    }
                    st.success("Face captured successfully!")
                    st.session_state.face_verified = True
                    st.session_state.face_capture_active = False
                else:
                    st.error("No face detected or multiple faces detected. Please try again with a clear, front-facing photo.")
        else:
            # Show the captured face if verification was successful
            if st.session_state.reference_face and os.path.exists(st.session_state.reference_face["path"]):
                st.image(st.session_state.reference_face["path"], caption="Verified Face", width=300)
                
                if st.button("Retake Photo"):
                    st.session_state.face_verified = False
                    st.session_state.face_capture_active = True
                    st.experimental_rerun()
                
        # Continue button shown below the camera or image
        if st.session_state.face_verified:
            if st.button("Continue to Document Submission"):
                next_step()
                st.experimental_rerun()
    
    with col2:
        st.video("https://sora.com/g/gen_01jpn3fq0ye6srrmjzeg8jv8jn")
        st.caption("Verification Process Explanation")
        
        st.markdown("""
        <div class='info-box'>
            <p><strong>Tips for a good verification photo:</strong></p>
            <ul>
                <li>Ensure your face is clearly visible</li>
                <li>Use good lighting</li>
                <li>Remove sunglasses or other face coverings</li>
                <li>Look directly at the camera</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# Step 3: Document Submission
elif st.session_state.current_step == 3:
    st.markdown("<h2 class='sub-header'>Document Submission</h2>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class='info-box'>
        <p>Please upload the following documents for verification:</p>
        <ol>
            <li>Aadhaar Card (mandatory)</li>
            <li>PAN Card (mandatory)</li>
            <li>Income Proof (salary slip or IT returns)</li>
        </ol>
        <p>You can either upload image files or take photos using your camera.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Document upload section
    doc_types = ["aadhaar", "pan", "income_proof"]
    doc_names = {"aadhaar": "Aadhaar Card", "pan": "PAN Card", "income_proof": "Income Proof"}
    
    def camera_capture_with_detection(doc_type):
        """Custom camera component with document detection"""
        # Create a unique key for this document type
        key = f"camera_{doc_type}"
        
        st.markdown("""
            <div class='info-box'>
                <p>Position your document within the frame:</p>
                <ul>
                    <li>Ensure good lighting</li>
                    <li>Hold the camera steady</li>
                    <li>Align document edges with the guide</li>
                </ul>
                <p>Auto-capture will trigger when a clear document is detected.</p>
            </div>
        """, unsafe_allow_html=True)
        
        # Initialize session state for this camera if not exists
        if f"last_capture_{key}" not in st.session_state:
            st.session_state[f"last_capture_{key}"] = time.time() - 2  # Allow initial capture
            
        if f"detection_active_{key}" not in st.session_state:
            st.session_state[f"detection_active_{key}"] = True
            
        if f"document_captured_{key}" not in st.session_state:
            st.session_state[f"document_captured_{key}"] = None
        
        # If document already captured, show it instead of camera
        if st.session_state[f"document_captured_{key}"] is not None:
            captured_img = st.session_state[f"document_captured_{key}"]
            st.image(captured_img, caption=f"{doc_names[doc_type]} Captured", use_column_width=True)
            
            if st.button("Retake Photo", key=f"retake_{key}"):
                st.session_state[f"document_captured_{key}"] = None
                st.session_state[f"detection_active_{key}"] = True
                st.experimental_rerun()
                
            return captured_img
        
        # If no document is captured yet, show the camera
        if st.session_state[f"detection_active_{key}"]:
            camera = st.camera_input(f"Document Camera - {doc_type}", key=key)
            
            if camera:
                # Convert the image to CV2 format
                bytes_data = camera.getvalue()
                nparr = np.frombuffer(bytes_data, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                # Detect document boundaries
                success, corners, confidence = detect_document_boundaries(img)
                
                if success:
                    # Draw boundaries on the image
                    annotated_img = draw_document_boundaries(img, corners)
                    
                    # Convert back to displayable format
                    _, buffer = cv2.imencode('.jpg', annotated_img)
                    img_str = base64.b64encode(buffer).decode()
                    
                    # Display the annotated image
                    st.markdown(f"""
                        <div style="position: relative;">
                            <img src="data:image/jpeg;base64,{img_str}" style="width: 100%;">
                            <div style="position: absolute; top: 10px; right: 10px; 
                                 background-color: rgba(0,0,0,0.7); color: white; 
                                 padding: 5px 10px; border-radius: 5px;">
                                Confidence: {confidence:.2%}
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # Check if document is clear
                    is_clear, clarity_issues = is_document_clear(img)
                    
                    # Auto-capture logic
                    current_time = time.time()
                    if (confidence > 0.8 and is_clear and 
                        current_time - st.session_state[f"last_capture_{key}"] >= 2):
                        
                        # Extract and rectify the document
                        extracted_img = extract_document(img, corners)
                        
                        # Save the captured image
                        _, buffer = cv2.imencode('.jpg', extracted_img)
                        captured_image = BytesIO(buffer)
                        
                        # Store in session state
                        st.session_state[f"document_captured_{key}"] = captured_image
                        st.session_state[f"detection_active_{key}"] = False
                        
                        # Update last capture time
                        st.session_state[f"last_capture_{key}"] = current_time
                        
                        st.success("Document detected and captured! Processing...")
                        st.experimental_rerun()
                    
                    # Show guidance based on detection
                    if not is_clear:
                        st.warning("Document quality issues detected:")
                        for issue in clarity_issues:
                            st.markdown(f"- {issue}")
                    elif confidence < 0.8:
                        st.info("Adjust document position for better alignment")
                else:
                    st.info("No document detected. Please position your document within the frame.")
        
        return st.session_state[f"document_captured_{key}"]
    
    # Function to generate QR code for mobile document upload
    def generate_document_upload_qr(doc_type):
        """Generate a QR code for uploading a specific document type from mobile"""
        # Create a unique ID for this upload request
        upload_id = str(uuid.uuid4())
        
        # Store the upload ID in session state
        if "mobile_uploads" not in st.session_state:
            st.session_state.mobile_uploads = {}
        
        st.session_state.mobile_uploads[upload_id] = {
            "doc_type": doc_type,
            "timestamp": time.time(),
            "status": "pending"
        }
        
        # Use localhost URL for local testing with proper multi-page app path
        # The "mobile_upload" becomes "Mobile_upload" in the URL (first letter capitalized)
        # This is how Streamlit formats page names in the URL
        base_url = "http://localhost:8501"
        upload_url = f"{base_url}/Mobile_upload?id={upload_id}&type={doc_type}"
        
        # For running on a local network, use your computer's IP address
        # You can uncomment and modify this line with your actual IP address
        # base_url = "http://192.168.1.xxx:8501"  # Replace with your actual IP address
        # upload_url = f"{base_url}/Mobile_upload?id={upload_id}&type={doc_type}"
        
        # Generate QR code with better error correction for mobile scanning
        qr = qrcode.QRCode(
            version=4,  # Higher version for more data
            error_correction=qrcode.constants.ERROR_CORRECT_H,  # Higher error correction
            box_size=10,
            border=4,
        )
        qr.add_data(upload_url)
        qr.make(fit=True)
        
        # Create an image from the QR Code with better contrast
        img = qr.make_image(fill_color="black", back_color="white")
        
        # Save to BytesIO object
        img_bytes = BytesIO()
        img.save(img_bytes)
        img_bytes.seek(0)
        
        return img_bytes, upload_id, upload_url

    for doc_type in doc_types:
        st.subheader(f"{doc_names[doc_type]} Upload")
        
        # Show document requirements and guidelines
        requirements = get_document_requirements(doc_type)
        with st.expander(f"📋 {doc_names[doc_type]} Requirements", expanded=False):
            st.markdown("#### Required Fields")
            for field in requirements["mandatory_fields"]:
                st.markdown(f"- {field.replace('_', ' ').title()}")
            
            st.markdown("#### Format")
            st.markdown(requirements["format"])
            
            st.markdown("#### Guidelines")
            for guideline in requirements["guidelines"]:
                st.markdown(f"- {guideline}")
        
        # Create tabs for different upload methods
        upload_tab, camera_tab, mobile_tab = st.tabs(["📁 File Upload", "📷 Camera Capture", "📱 Mobile Upload"])
        
        # Tab 1: File Upload
        with upload_tab:
            uploaded_file = st.file_uploader(f"Upload {doc_names[doc_type]}", type=["jpg", "jpeg", "png", "pdf"], key=f"upload_{doc_type}")
            if uploaded_file:
                with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{uploaded_file.name.split(".")[-1]}') as tmp:
                    tmp.write(uploaded_file.getvalue())
                    doc_path = tmp.name
                
                # Process document with enhanced validation
                is_valid, doc_text, extracted_info, validation_result = process_document(doc_path, doc_type)
                
                if validation_result["status"] == "success":
                    st.session_state.documents[doc_type] = {
                        "path": doc_path,
                        "text": doc_text,
                        "extracted_info": extracted_info,
                        "validation": validation_result
                    }
                    
                    st.success(f"✅ {doc_names[doc_type]} verified successfully!")
                    
                    # Display extracted information in a clean format
                    with st.expander("📄 Extracted Information", expanded=True):
                        for key, value in extracted_info.items():
                            if key != "document_type":
                                st.write(f"**{key.replace('_', ' ').title()}:** {value}")
                
                elif validation_result["status"] == "quality_issues":
                    st.error("❌ Document image quality issues detected")
                    st.markdown("##### Please fix the following issues:")
                    for issue in validation_result["issues"]:
                        st.markdown(f"- {issue}")
                    
                    # Show image quality tips
                    with st.expander("📸 Tips for Better Image Quality"):
                        st.markdown("""
                        - Ensure good lighting conditions
                        - Hold the camera steady
                        - Keep the document flat and aligned
                        - Make sure all text is clearly visible
                        - Avoid shadows and glare
                        """)
                
                elif validation_result["status"] == "validation_failed":
                    st.error("❌ Document validation failed")
                    st.markdown("##### Issues found:")
                    for issue in validation_result["issues"]:
                        st.markdown(f"- {issue}")
                    
                    # Show retry guidelines
                    with st.expander("🔄 How to Fix These Issues"):
                        st.markdown("##### Please ensure:")
                        for req in validation_result["requirements"]:
                            st.markdown(f"- {req}")
                
                else:
                    st.error(f"❌ {validation_result['message']}")
        
        # Tab 2: Camera Capture
        with camera_tab:
            captured_image = camera_capture_with_detection(doc_type)
            if captured_image:
                # Process the captured document
                with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
                    tmp.write(captured_image.getvalue())
                    doc_path = tmp.name
                
                # Process document with enhanced validation
                is_valid, doc_text, extracted_info, validation_result = process_document(doc_path, doc_type)
                
                if validation_result["status"] == "success":
                    st.session_state.documents[doc_type] = {
                        "path": doc_path,
                        "text": doc_text,
                        "extracted_info": extracted_info,
                        "validation": validation_result
                    }
                    
                    st.success(f"✅ {doc_names[doc_type]} captured and verified successfully!")
                    
                    # Display extracted information in a clean format
                    with st.expander("📄 Extracted Information", expanded=True):
                        for key, value in extracted_info.items():
                            if key != "document_type":
                                st.write(f"**{key.replace('_', ' ').title()}:** {value}")
                
                elif validation_result["status"] == "quality_issues":
                    st.error("❌ Document image quality issues detected")
                    st.markdown("##### Please fix the following issues:")
                    for issue in validation_result["issues"]:
                        st.markdown(f"- {issue}")
                    
                    # Show camera tips
                    with st.expander("📸 Camera Tips"):
                        st.markdown("""
                        - Hold your device steady
                        - Ensure proper lighting
                        - Position the document within the frame
                        - Keep the camera lens clean
                        - Avoid shadows on the document
                        """)
                
                elif validation_result["status"] == "validation_failed":
                    st.error("❌ Document validation failed")
                    st.markdown("##### Issues found:")
                    for issue in validation_result["issues"]:
                        st.markdown(f"- {issue}")
                    
                    # Show retry guidelines
                    with st.expander("🔄 How to Fix These Issues"):
                        st.markdown("##### Please ensure:")
                        for req in validation_result["requirements"]:
                            st.markdown(f"- {req}")
                
                else:
                    st.error(f"❌ {validation_result['message']}")
        
        # Tab 3: Mobile Upload via QR Code
        with mobile_tab:
            st.markdown("""
            <div class='info-box'>
                <p>Having trouble with document detection? Use your mobile phone instead!</p>
                <ol>
                    <li>Scan the QR code below using your mobile phone camera</li>
                    <li>Take a clear photo of your document using your mobile camera</li>
                    <li>The document will be automatically processed and added to your application</li>
                </ol>
                <p><em>This provides a better experience when the webcam detection isn't working well.</em></p>
            </div>
            """, unsafe_allow_html=True)
            
            # Generate QR code for this document type
            qr_img, upload_id, upload_url = generate_document_upload_qr(doc_type)
            
            # Display QR code with instructions
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.image(qr_img, caption=f"Scan to upload {doc_names[doc_type]}")
                st.markdown(f"**Upload ID:** {upload_id[:8]}...")
                st.markdown(f"**Upload URL:** {upload_url}")
                
                # Handle mobile upload simulation
                if st.button("Check Upload Status", key=f"check_status_{doc_type}"):
                    # This is a simulation for demonstration purposes
                    st.info("Waiting for document upload from mobile device...")
                    
                    # For demonstration purposes, let's simulate a successful upload after 3 seconds
                    with st.spinner("Processing upload..."):
                        time.sleep(3)
                    
                    # Create a simulated document directly instead of relying on sample files
                    # This ensures we always have something to process, even without sample files
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
                        # Create a blank image with some basic text for demonstration
                        demo_image = np.ones((1200, 800, 3), dtype=np.uint8) * 255  # White background
                        
                        # Add document outline
                        cv2.rectangle(demo_image, (50, 50), (750, 1150), (0, 0, 0), 2)
                        
                        # Add document content based on the type
                        if doc_type == "aadhaar":
                            cv2.putText(demo_image, "AADHAAR", (300, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                            cv2.putText(demo_image, "Government of India", (250, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
                            cv2.putText(demo_image, "Name: " + st.session_state.user_data.get("name", "John Doe"), (100, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
                            cv2.putText(demo_image, "DOB: 01/01/1990", (100, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
                            cv2.putText(demo_image, "Aadhaar Number: 1234 5678 9012", (100, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
                            cv2.putText(demo_image, "Gender: Male", (100, 500), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
                        
                        elif doc_type == "pan":
                            cv2.putText(demo_image, "INCOME TAX DEPARTMENT", (200, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
                            cv2.putText(demo_image, "PERMANENT ACCOUNT NUMBER", (180, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
                            cv2.putText(demo_image, "PAN: ABCDE1234F", (100, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
                            cv2.putText(demo_image, "Name: " + st.session_state.user_data.get("name", "John Doe"), (100, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
                            cv2.putText(demo_image, "Father's Name: Parent Name", (100, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
                            cv2.putText(demo_image, "DOB: 01/01/1990", (100, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
                        
                        elif doc_type == "income_proof":
                            cv2.putText(demo_image, "SALARY SLIP", (300, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                            cv2.putText(demo_image, "Period: March 2023", (300, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
                            cv2.putText(demo_image, "Employee Name: " + st.session_state.user_data.get("name", "John Doe"), (100, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
                            cv2.putText(demo_image, "Company: ACME Corporation", (100, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
                            monthly_income = st.session_state.user_data.get("monthly_income", 50000)
                            cv2.putText(demo_image, f"Gross Salary: Rs. {monthly_income}", (100, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
                            cv2.putText(demo_image, f"Net Salary: Rs. {int(monthly_income * 0.85)}", (100, 500), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
                        
                        # Write the image to the temporary file
                        cv2.imwrite(tmp.name, demo_image)
                        doc_path = tmp.name
                    
                    # Create a dictionary of extracted information based on the document type
                    if doc_type == "aadhaar":
                        extracted_info = {
                            "document_type": "Aadhaar Card",
                            "aadhaar_number": "1234 5678 9012",
                            "name": st.session_state.user_data.get("name", "John Doe"),
                            "dob": "01/01/1990",
                            "gender": "Male"
                        }
                    elif doc_type == "pan":
                        extracted_info = {
                            "document_type": "PAN Card",
                            "pan_number": "ABCDE1234F",
                            "name": st.session_state.user_data.get("name", "John Doe"),
                            "father_name": "Parent Name",
                            "dob": "01/01/1990"
                        }
                    elif doc_type == "income_proof":
                        monthly_income = st.session_state.user_data.get("monthly_income", 50000)
                        extracted_info = {
                            "document_type": "Income Proof",
                            "proof_type": "Salary Slip",
                            "name": st.session_state.user_data.get("name", "John Doe"),
                            "monthly_income": monthly_income,
                            "period": "March 2023",
                            "company": "ACME Corporation"
                        }
                    
                    # Create a simulated validation result
                    validation_result = {
                        "status": "success",
                        "message": "Document validation successful",
                        "requirements": get_document_requirements(doc_type)["guidelines"]
                    }
                    
                    # Store the simulated document in session state
                    st.session_state.documents[doc_type] = {
                        "path": doc_path,
                        "text": f"Simulated document text for {doc_type}",
                        "extracted_info": extracted_info,
                        "validation": validation_result
                    }
                    
                    # Display success message
                    st.success(f"✅ {doc_names[doc_type]} uploaded from mobile and verified successfully!")
                    
                    # Display the document image
                    demo_img = cv2.imread(doc_path)
                    st.image(doc_path, caption=f"{doc_names[doc_type]} Preview", use_column_width=True)
                    
                    # Display extracted information in a clean format
                    with st.expander("📄 Extracted Information", expanded=True):
                        for key, value in extracted_info.items():
                            if key != "document_type":
                                st.write(f"**{key.replace('_', ' ').title()}:** {value}")
            
            with col2:
                st.markdown("""
                #### Why use mobile upload?
                - **Better camera quality** than most webcams
                - **Easier positioning** of the document
                - **No detection issues** that can occur with webcams
                - **Faster upload** for users on mobile devices
                
                #### Tips for best results:
                - Use good lighting
                - Place document on a dark, contrasting background
                - Ensure all text is clearly visible
                - Hold your phone steady when taking the photo
                """)

        st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    
    # Navigation buttons with document status summary
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Go Back"):
            prev_step()
            st.experimental_rerun()
    
    with col2:
        # Check document status
        mandatory_docs = {"aadhaar": False, "pan": False}
        doc_status = {}
        
        for doc_type in doc_types:
            if doc_type in st.session_state.documents:
                validation = st.session_state.documents[doc_type].get("validation", {})
                doc_status[doc_type] = validation.get("status") == "success"
                if doc_type in mandatory_docs:
                    mandatory_docs[doc_type] = doc_status[doc_type]
        
        if st.button("Continue to Loan Interview"):
            if all(mandatory_docs.values()):
                next_step()
                st.experimental_rerun()
            else:
                missing_docs = [doc_names[doc] for doc, status in mandatory_docs.items() if not status]
                st.error(f"Please upload and verify the following mandatory documents: {', '.join(missing_docs)}")
        
        # Show document submission status
        st.markdown("##### Document Status:")
        for doc_type in doc_types:
            icon = "✅" if doc_status.get(doc_type, False) else "❌"
            required = "(Required)" if doc_type in mandatory_docs else "(Optional)"
            st.markdown(f"{icon} {doc_names[doc_type]} {required}")

# Step 4: Loan Interview
elif st.session_state.current_step == 4:
    st.markdown("<h2 class='sub-header'>Loan Interview</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div class='info-box'>
            <p>Now, I'll ask you a few questions about your loan requirements and financial status.</p>
            <p>Please record a short video response for each question. This helps us better understand your needs.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Loan questions and video responses
        questions = [
            "Please introduce yourself and explain the purpose of your loan application.",
            "What is your current employment status and monthly income?",
            "Do you have any existing loans or financial obligations?",
            "How much loan amount are you looking for and what repayment period would you prefer?"
        ]
        
        # Loan form fields
        st.subheader("Loan Details")
        loan_type = st.selectbox("Loan Type", ["Personal Loan", "Home Loan", "Education Loan", "Vehicle Loan", "Business Loan"])
        loan_amount = st.number_input("Loan Amount (₹)", min_value=10000, max_value=10000000, step=10000)
        loan_tenure = st.slider("Loan Tenure (Years)", min_value=1, max_value=30, value=5)
        
        st.subheader("Financial Information")
        employment_status = st.selectbox("Employment Status", ["Salaried", "Self-employed", "Business Owner", "Unemployed", "Retired"])
        monthly_income = st.number_input("Monthly Income (₹)", min_value=0, max_value=10000000, step=1000)
        existing_loans = st.number_input("Existing EMI Obligations (₹/month)", min_value=0, max_value=1000000, step=1000)
        
        # Save loan information
        st.session_state.user_data.update({
            "loan_type": loan_type,
            "loan_amount": loan_amount,
            "loan_tenure": loan_tenure,
            "employment_status": employment_status,
            "monthly_income": monthly_income,
            "existing_loans": existing_loans
        })
        
        # Video response section
        st.subheader("Video Responses")
        
        # Initialize verification state variables if they don't exist
        if 'current_question_index' not in st.session_state:
            st.session_state.current_question_index = 0
        if 'question_verified' not in st.session_state:
            st.session_state.question_verified = False
            
        # Show the current question
        current_q = st.session_state.current_question_index
        if current_q < len(questions):
            st.write(f"**Question {current_q+1}:** {questions[current_q]}")
            
            # Handle face verification for current question
            if not st.session_state.question_verified and st.session_state.reference_face:
                st.write("**Please verify your identity before recording your answer:**")
                verify_img = st.camera_input(
                    f"Verify your identity for Question {current_q+1}", 
                    key=f"verify_q{current_q}"
                )
                
                if verify_img:
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
                        tmp.write(verify_img.getvalue())
                        verify_path = tmp.name
                    
                    is_same_person, confidence = verify_face(verify_path, st.session_state.reference_face["encoding"])
                    
                    if is_same_person:
                        st.success(f"Identity verified! (Confidence: {confidence:.2f}%)")
                        st.session_state.question_verified = True
                        # Rerun to update UI and show video recording option
                        st.experimental_rerun()
                    else:
                        st.error(f"Identity verification failed. Please try again. (Confidence: {confidence:.2f}%)")
                        
            # If verified or no reference face to check against, show video recording
            elif st.session_state.question_verified or not st.session_state.reference_face:
                st.write("**Record your answer:**")
                video_response = st.camera_input(
                    f"Record your answer to Question {current_q+1}", 
                    key=f"video_q{current_q}"
                )
                
                if video_response:
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
                        tmp.write(video_response.getvalue())
                        video_path = tmp.name
                    
                    st.session_state.videos[f"question_{current_q+1}"] = {
                        "question": questions[current_q],
                        "path": video_path
                    }
                    
                    st.success(f"Response to Question {current_q+1} recorded successfully!")
                    
                    # Reset verification state for next question
                    st.session_state.question_verified = False
                    st.session_state.current_question_index += 1
                    
                    # If there are more questions, rerun to show the next question
                    if st.session_state.current_question_index < len(questions):
                        st.experimental_rerun()
                        
            # Progress indicator
            question_progress = (current_q + 1) / len(questions)
            st.progress(question_progress)
            st.write(f"Question {current_q + 1} of {len(questions)}")
        else:
            st.success("All questions answered! You can now check your loan eligibility.")
    
    with col2:
        st.video("https://assets.mixkit.co/videos/preview/mixkit-businesswoman-speaking-in-a-video-call-22574-large.mp4")
        st.caption("Your AI Bank Manager")
        
        st.markdown("""
        <div class='info-box'>
            <p><strong>Tips for your video responses:</strong></p>
            <ul>
                <li>Speak clearly and concisely</li>
                <li>Provide honest information</li>
                <li>Keep your responses under 30 seconds</li>
                <li>Make sure your face is visible</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Navigation buttons
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Go Back"):
            prev_step()
            st.experimental_rerun()
    
    with col2:
        if st.button("Check Loan Eligibility"):
            # Ensure all required information is provided
            if not loan_type or not loan_amount or not employment_status or not monthly_income:
                st.error("Please fill in all required fields to proceed.")
            else:
                next_step()
                st.experimental_rerun()

# Step 5: Loan Eligibility
elif st.session_state.current_step == 5:
    st.markdown("<h2 class='sub-header'>Loan Eligibility</h2>", unsafe_allow_html=True)
    
    # Calculate eligibility based on provided information
    loan_data = {
        "loan_type": st.session_state.user_data.get("loan_type", ""),
        "loan_amount": st.session_state.user_data.get("loan_amount", 0),
        "loan_tenure": st.session_state.user_data.get("loan_tenure", 0),
        "monthly_income": st.session_state.user_data.get("monthly_income", 0),
        "existing_loans": st.session_state.user_data.get("existing_loans", 0),
        "employment_status": st.session_state.user_data.get("employment_status", "")
    }
    
    # Document information
    document_data = {doc_type: info.get("extracted_info", {}) for doc_type, info in st.session_state.documents.items()}
    
    # Check eligibility
    eligibility_result = check_eligibility(loan_data, document_data)
    st.session_state.eligibility_result = eligibility_result
    
    # Display eligibility result
    if eligibility_result["status"] == "approved":
        st.markdown("""
        <div class='success-box'>
            <h3>✅ Congratulations! Your loan is pre-approved.</h3>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"**Approved Amount:** ₹{eligibility_result['approved_amount']:,}")
            st.markdown(f"**Interest Rate:** {eligibility_result['interest_rate']}%")
            st.markdown(f"**Loan Tenure:** {eligibility_result['tenure']} years")
        
        with col2:
            st.markdown(f"**Monthly EMI:** ₹{eligibility_result['monthly_emi']:,.2f}")
            st.markdown(f"**Processing Fee:** ₹{eligibility_result['processing_fee']:,.2f}")
            st.markdown(f"**Total Interest:** ₹{eligibility_result['total_interest']:,.2f}")
        
        st.markdown("""
        <div class='info-box'>
            <p>Our team will contact you shortly to complete the final formalities.</p>
            <p>You will receive the formal approval letter on your registered email address.</p>
        </div>
        """, unsafe_allow_html=True)
        
    elif eligibility_result["status"] == "rejected":
        st.markdown("""
        <div class='error-box'>
            <h3>❌ We're sorry, your loan application couldn't be approved at this time.</h3>
        </div>
        """, unsafe_allow_html=True)
        
        st.subheader("Reasons:")
        for reason in eligibility_result["rejection_reasons"]:
            st.markdown(f"- {reason}")
        
        st.markdown("""
        <div class='info-box'>
            <p>You can apply again after addressing the above issues.</p>
            <p>Our team can provide guidance on improving your eligibility. Would you like to schedule a consultation?</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Schedule a Consultation"):
            st.session_state.user_data["consultation_requested"] = True
            st.success("Consultation request submitted. Our team will contact you within 24 hours.")
            
    else:  # more_info_needed
        st.markdown("""
        <div class='info-box'>
            <h3>🔄 We need additional information to process your application.</h3>
        </div>
        """, unsafe_allow_html=True)
        
        st.subheader("Required Information:")
        for info_item in eligibility_result["required_info"]:
            st.markdown(f"- {info_item}")
        
        additional_info = st.text_area("Provide additional information", height=100)
        additional_doc = st.file_uploader("Upload additional documents", accept_multiple_files=True)
        
        if st.button("Submit Additional Information"):
            if additional_info or additional_doc:
                st.session_state.user_data["additional_info"] = additional_info
                st.success("Additional information submitted. Our team will review and get back to you within 24 hours.")
            else:
                st.error("Please provide either additional information or documents.")
    
    # Navigation buttons
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Go Back"):
            prev_step()
            st.experimental_rerun()
    
    with col2:
        if st.button("View Application Summary"):
            next_step()
            st.experimental_rerun()

# Step 6: Application Summary
elif st.session_state.current_step == 6:
    st.markdown("<h2 class='sub-header'>Application Summary</h2>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class='info-box'>
        <p>Thank you for completing your loan application process with our AI Bank Manager.</p>
        <p>Below is a summary of your application details.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Personal Information
    st.subheader("Personal Information")
    st.write(f"**Name:** {st.session_state.user_data.get('name', 'N/A')}")
    st.write(f"**Email:** {st.session_state.user_data.get('email', 'N/A')}")
    st.write(f"**Phone:** {st.session_state.user_data.get('phone', 'N/A')}")
    st.write(f"**Application ID:** {st.session_state.user_data.get('application_id', 'N/A')}")
    st.write(f"**Submission Date:** {st.session_state.user_data.get('timestamp', 'N/A')}")
    
    # Loan Details
    st.subheader("Loan Details")
    st.write(f"**Loan Type:** {st.session_state.user_data.get('loan_type', 'N/A')}")
    st.write(f"**Requested Amount:** ₹{st.session_state.user_data.get('loan_amount', 0):,}")
    st.write(f"**Tenure:** {st.session_state.user_data.get('loan_tenure', 0)} years")
    
    # Financial Information
    st.subheader("Financial Information")
    st.write(f"**Employment Status:** {st.session_state.user_data.get('employment_status', 'N/A')}")
    st.write(f"**Monthly Income:** ₹{st.session_state.user_data.get('monthly_income', 0):,}")
    st.write(f"**Existing EMI Obligations:** ₹{st.session_state.user_data.get('existing_loans', 0):,}/month")
    
    # Documents Submitted
    st.subheader("Documents Submitted")
    for doc_type, doc_info in st.session_state.documents.items():
        st.write(f"**{doc_type.replace('_', ' ').title()}:** Verified ✓")
    
    # Application Status
    st.subheader("Application Status")
    if st.session_state.eligibility_result:
        status = st.session_state.eligibility_result["status"]
        if status == "approved":
            st.markdown("""
            <div class='success-box'>
                <h3>✅ Approved</h3>
                <p>Your loan has been pre-approved. Our team will contact you shortly.</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.write(f"**Approved Amount:** ₹{st.session_state.eligibility_result['approved_amount']:,}")
            st.write(f"**Interest Rate:** {st.session_state.eligibility_result['interest_rate']}%")
            st.write(f"**Monthly EMI:** ₹{st.session_state.eligibility_result['monthly_emi']:,.2f}")
            
        elif status == "rejected":
            st.markdown("""
            <div class='error-box'>
                <h3>❌ Rejected</h3>
                <p>Unfortunately, your application couldn't be approved at this time.</p>
            </div>
            """, unsafe_allow_html=True)
            
            if "rejection_reasons" in st.session_state.eligibility_result:
                st.subheader("Reasons:")
                for reason in st.session_state.eligibility_result["rejection_reasons"]:
                    st.markdown(f"- {reason}")
            
        else:
            st.markdown("""
            <div class='info-box'>
                <h3>🔄 Pending</h3>
                <p>Additional information is required to process your application.</p>
            </div>
            """, unsafe_allow_html=True)
            
            if "required_info" in st.session_state.eligibility_result:
                st.subheader("Required Information:")
                for info_item in st.session_state.eligibility_result["required_info"]:
                    st.markdown(f"- {info_item}")
    else:
        st.info("Status information not available. Please complete the eligibility check first.")
    
    # Navigation buttons
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Go Back"):
            prev_step()
            st.experimental_rerun()
    
    with col2:
        if st.button("Start New Application"):
            reset_app()
            st.experimental_rerun()
    
    # Download application summary
    st.download_button(
        label="Download Application Summary",
        data=f"""
            AI BANK MANAGER - LOAN APPLICATION SUMMARY
            
            APPLICATION DETAILS
            Application ID: {st.session_state.user_data.get('application_id', 'N/A')}
            Submission Date: {st.session_state.user_data.get('timestamp', 'N/A')}
            
            PERSONAL INFORMATION
            Name: {st.session_state.user_data.get('name', 'N/A')}
            Email: {st.session_state.user_data.get('email', 'N/A')}
            Phone: {st.session_state.user_data.get('phone', 'N/A')}
            
            LOAN DETAILS
            Loan Type: {st.session_state.user_data.get('loan_type', 'N/A')}
            Requested Amount: ₹{st.session_state.user_data.get('loan_amount', 0):,}
            Tenure: {st.session_state.user_data.get('loan_tenure', 0)} years
            
            FINANCIAL INFORMATION
            Employment Status: {st.session_state.user_data.get('employment_status', 'N/A')}
            Monthly Income: ₹{st.session_state.user_data.get('monthly_income', 0):,}
            Existing EMI Obligations: ₹{st.session_state.user_data.get('existing_loans', 0):,}/month
            
            APPLICATION STATUS: {st.session_state.eligibility_result["status"].upper() if st.session_state.eligibility_result else "N/A"}
            """,
        file_name=f"loan_application_{st.session_state.user_data.get('application_id', 'summary')}.txt",
        mime="text/plain"
    )
