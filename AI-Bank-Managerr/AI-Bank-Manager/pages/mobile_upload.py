import streamlit as st
import os
import time
import uuid
import tempfile
import cv2
from io import BytesIO
from PIL import Image

# Import utility modules
from utils.document_processor import process_document, get_document_requirements

# Set page configuration for mobile view
st.set_page_config(
    page_title="Upload Document",
    page_icon="📱",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Get URL parameters
query_params = st.experimental_get_query_params()
upload_id = query_params.get("id", [""])[0]
doc_type = query_params.get("type", [""])[0]

# Define document types
doc_names = {"aadhaar": "Aadhaar Card", "pan": "PAN Card", "income_proof": "Income Proof"}

# Custom CSS for mobile view
st.markdown("""
<style>
    /* Mobile-friendly styles */
    .main-header {
        font-size: 1.8rem;
        color: #E31B23;
        text-align: center;
        margin-bottom: 1rem;
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

    /* Button styling */
    .stButton>button {
        width: 100%;
        background: linear-gradient(45deg, #0F2337, #E31B23);
        color: white;
        font-weight: bold;
        padding: 0.8rem 1rem;
        border-radius: 0.5rem;
        border: none;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-top: 1rem;
    }

    /* Make camera input full width */
    .stCamera > div {
        display: flex;
        justify-content: center;
    }
</style>
""", unsafe_allow_html=True)

# Main function
def main():
    st.markdown("<h1 class='main-header'>Mobile Document Upload</h1>", unsafe_allow_html=True)

    # Check if we have valid parameters
    if not upload_id or not doc_type or doc_type not in doc_names:
        st.error("Invalid upload link. Please scan the QR code again.")
        return

    # Show document information
    st.markdown(f"### Upload {doc_names.get(doc_type, 'Document')}")
    st.markdown(f"**Upload ID:** {upload_id[:8]}...")

    # Show document requirements
    requirements = get_document_requirements(doc_type)
    with st.expander("📋 Document Requirements", expanded=True):
        st.markdown("#### Required Fields")
        for field in requirements["mandatory_fields"]:
            st.markdown(f"- {field.replace('_', ' ').title()}")

        st.markdown("#### Format")
        st.markdown(requirements["format"])

        st.markdown("#### Guidelines")
        for guideline in requirements["guidelines"]:
            st.markdown(f"- {guideline}")

    # Document upload options
    st.markdown("### Take Photo or Upload File")

    uploaded_file = st.file_uploader("Choose a file", type=["jpg", "jpeg", "png", "pdf"])
    camera_input = st.camera_input("Take a picture")

    # Process uploaded file
    if uploaded_file:
        process_upload(uploaded_file, upload_id, doc_type)

    # Process camera input
    if camera_input:
        process_upload(camera_input, upload_id, doc_type)

    # Tips for good photo
    st.markdown("""
    ### Tips for a clear document photo:
    - Ensure good lighting with no shadows
    - Place the document on a dark, flat surface
    - Make sure all text is clearly visible
    - Include all corners of the document
    - Hold your phone steady when taking the picture
    """)

def process_upload(file_obj, upload_id, doc_type):
    """Process an uploaded document file"""
    try:
        # Show processing status
        with st.spinner("Processing document..."):
            # Save the file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
                tmp.write(file_obj.getvalue())
                file_path = tmp.name

            # In a production environment, you would upload this to your server
            # For this demo, we'll just show what happens next
            time.sleep(2)  # Simulate processing delay

            # Process document with enhanced validation
            is_valid, doc_text, extracted_info, validation_result = process_document(file_path, doc_type)

            if validation_result["status"] == "success":
                # Show success message
                st.success("Document uploaded and verified successfully!")
                st.markdown("""
                <div class='success-box'>
                    <p>✅ Your document has been successfully uploaded and verified.</p>
                    <p>You can now return to the main application.</p>
                    <p>The document details will automatically appear in your application.</p>
                </div>
                """, unsafe_allow_html=True)

                # Display a preview of the uploaded document
                image = Image.open(file_obj)
                st.image(image, caption=f"{doc_names.get(doc_type, 'Document')} Preview", use_column_width=True)


            elif validation_result["status"] == "quality_issues":
                st.error("❌ Document image quality issues detected")
                st.markdown("##### Please fix the following issues:")
                for issue in validation_result["issues"]:
                    st.markdown(f"- {issue}")

                # Show image quality tips - consider adding tips here as well, or link to them

            elif validation_result["status"] == "validation_failed":
                st.error("❌ Document validation failed")
                st.markdown("##### Issues found:")
                for issue in validation_result["issues"]:
                    st.markdown(f"- {issue}")

                # Show retry guidelines - consider adding guidelines here or link to them

            else: # general error
                st.error(f"❌ Document processing failed: {validation_result.get('message', 'Unknown error')}")

            # Cleanup the temp file when done
            os.unlink(file_path)

    except Exception as e:
        st.error(f"Error processing document: {str(e)}")
        st.markdown("Please try again with a clearer image.")

if __name__ == "__main__":
    main()
