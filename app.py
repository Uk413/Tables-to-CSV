import streamlit as st
import os
from mistralai import Mistral
import pandas as pd
from PIL import Image
from dotenv import load_dotenv  # Import the library to load .env variables

# Load environment variables from .env file
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Image to CSV Converter",
    page_icon="📊",
    layout="wide"
)

# Initialize Mistral client
@st.cache_resource
def get_mistral_client():
    api_key = os.getenv("MISTRAL_API_KEY")  # Retrieve API key from environment variables
    if not api_key:
        raise ValueError("MISTRAL_API_KEY is not set in the .env file.")
    return Mistral(api_key=api_key)

def process_image(image_bytes):
    import base64
    client = get_mistral_client()
    
    # Convert image bytes to base64
    image_base64 = base64.b64encode(image_bytes).decode('utf-8')
    
    # Process with OCR using the correct format
    result = client.ocr.process(
        model="mistral-ocr-latest",
        document={
            "type": "image_url",
            "image_url": f"data:image/jpeg;base64,{image_base64}"
        }
    )
    
    return result

def parse_ocr_response(result):
    # Check if the response has pages
    if hasattr(result, 'pages') and len(result.pages) > 0:
        # Get the markdown content from the first page
        markdown_content = result.pages[0].markdown
        
        # Display the raw markdown
        st.subheader("Raw Markdown")
        st.code(markdown_content, language='markdown')
        
        # Convert markdown table to DataFrame
        import pandas as pd
        # Split the markdown into lines and filter out the separator line
        lines = [line.strip() for line in markdown_content.split('\n') if line.strip()]
        # Remove the separator line (contains :--:)
        lines = [line for line in lines if ':--:' not in line]
        
        # Parse header and data
        header = [col.strip() for col in lines[0].strip('|').split('|')]
        data = []
        for line in lines[1:]:
            row = [cell.strip() for cell in line.strip('|').split('|')]
            data.append(row)
            
        # Create DataFrame
        return pd.DataFrame(data, columns=header)
    return None

def main():
    st.title("📊 Image to CSV Converter")
    st.write("Upload an image containing tabular data to convert it to CSV format.")
    
    uploaded_file = st.file_uploader(
        "Choose an image file", 
        type=["png", "jpg", "jpeg"],
        help="Maximum file size: 10MB"
    )
    
    if uploaded_file:
        try:
            # Display original image
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Original Image")
                image = Image.open(uploaded_file)
                st.image(image, use_container_width=True)
            
            # Process image
            with st.spinner("Processing image..."):
                result = process_image(uploaded_file.getvalue())
                
                # Parse the OCR result
                df = parse_ocr_response(result)
                
                if df is not None:
                    with col2:
                        st.subheader("Extracted Table")
                        st.dataframe(df)
                        
                        # Download button
                        st.download_button(
                            label="Download CSV",
                            data=df.to_csv(index=False),
                            file_name="converted_table.csv",
                            mime="text/csv"
                        )
                else:
                    st.error("Could not extract table from the image")
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.write("Full error:", str(e))
            
if __name__ == "__main__":
    main()