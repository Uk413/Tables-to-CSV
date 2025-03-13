import streamlit as st
import os
import boto3
import pandas as pd
from PIL import Image
import io
import zipfile
from dotenv import load_dotenv
load_dotenv()
# Page configuration
st.set_page_config(
    page_title="Image to CSV Converter",
    page_icon="📊",
    layout="wide"
)

# Initialize Textract client
@st.cache_resource
def get_textract_client():
    return boto3.client("textract", aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"), aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"), region_name=os.getenv("AWS_REGION")
)  # Update region if necessary

def preprocess_image(image_bytes):
    """
    Preprocess the image before sending to Textract
    """
    image = Image.open(io.BytesIO(image_bytes))
    
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize if image is too large (Textract has limits)
    max_size = 10000
    if max(image.size) > max_size:
        ratio = max_size / max(image.size)
        new_size = tuple(int(dim * ratio) for dim in image.size)
        image = image.resize(new_size, Image.Resampling.LANCZOS)
    
    # Convert back to bytes
    buffer = io.BytesIO()
    image.save(buffer, format='JPEG', quality=95)
    return buffer.getvalue()

def process_image_with_textract(image_bytes):
    """
    Process the uploaded image using AWS Textract with enhanced table detection.
    """
    client = get_textract_client()
    
    # Enhanced analyze_document call with all feature types
    response = client.analyze_document(
        Document={'Bytes': image_bytes},
        FeatureTypes=['TABLES', 'FORMS', 'LAYOUT']  # Added more feature types
    )
    
    return response

def parse_textract_response(response):
    """
    Parse AWS Textract response with improved table detection and validation.
    """
    tables = []
    blocks = response['Blocks']
    
    # Create a mapping of block IDs to blocks for faster lookup
    block_map = {block['Id']: block for block in blocks}
    
    # Step 1: Extract all text content
    text_map = {}
    for block in blocks:
        if block['BlockType'] in ['LINE', 'WORD']:
            text_map[block['Id']] = block.get('Text', '')
    
    # Step 2: Process tables
    for block in blocks:
        if block['BlockType'] == 'TABLE':
            table_matrix = {}
            table_has_content = False
            
            # Get all cells belonging to this table
            if 'Relationships' in block:
                for relationship in block['Relationships']:
                    if relationship['Type'] == 'CHILD':
                        for cell_id in relationship['Ids']:
                            cell = block_map.get(cell_id)
                            
                            if cell and cell['BlockType'] == 'CELL':
                                row_index = cell['RowIndex'] - 1
                                col_index = cell['ColumnIndex'] - 1
                                
                                # Get cell content
                                cell_content = ''
                                if 'Relationships' in cell:
                                    for cell_relationship in cell['Relationships']:
                                        if cell_relationship['Type'] == 'CHILD':
                                            for word_id in cell_relationship['Ids']:
                                                word_block = block_map.get(word_id)
                                                if word_block and 'Text' in word_block:
                                                    cell_content += word_block['Text'] + ' '
                                    cell_content = cell_content.strip()
                                
                                if not cell_content and cell['Id'] in text_map:
                                    cell_content = text_map[cell['Id']]
                                
                                # Store cell content in matrix
                                if row_index not in table_matrix:
                                    table_matrix[row_index] = {}
                                table_matrix[row_index][col_index] = cell_content
                                
                                if cell_content.strip():
                                    table_has_content = True
            
            # Convert matrix to DataFrame if table has content
            if table_has_content:
                # Determine table dimensions
                max_row = max(table_matrix.keys()) + 1
                max_col = max(max(cell.keys()) for cell in table_matrix.values()) + 1
                
                # Create empty table
                table_data = [['' for _ in range(max_col)] for _ in range(max_row)]
                
                # Fill in the data
                for row_idx in table_matrix:
                    for col_idx in table_matrix[row_idx]:
                        table_data[row_idx][col_idx] = table_matrix[row_idx][col_idx]
                
                df = pd.DataFrame(table_data)
                
                # Clean up the DataFrame
                df = df.replace(r'^\s*$', '', regex=True)  # Remove empty cells
                
                # Only append if DataFrame is not empty
                if not df.empty and df.values.any():
                    tables.append(df)
    
    return tables


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
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Original Image")
                image = Image.open(uploaded_file)
                st.image(image, use_container_width=True)
            
            with st.spinner("Processing image..."):
                # Preprocess the image
                processed_image = preprocess_image(uploaded_file.getvalue())
                
                # Process with Textract
                result = process_image_with_textract(processed_image)
                
                tables = parse_textract_response(result)
                
                if tables:
                    with col2:
                        st.subheader(f"Found {len(tables)} Tables")
                        
                        zip_buffer = io.BytesIO()
                        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                            for i, df in enumerate(tables):
                                st.subheader(f"Table {i+1}")
                                st.dataframe(df)
                                
                                csv_data = df.to_csv(index=False)
                                st.download_button(
                                    label=f"Download Table {i+1}",
                                    data=csv_data,
                                    file_name=f"table_{i+1}.csv",
                                    mime="text/csv"
                                )
                                
                                zip_file.writestr(f"table_{i+1}.csv", csv_data)
                        
                        st.download_button(
                            label="Download All Tables (ZIP)",
                            data=zip_buffer.getvalue(),
                            file_name="all_tables.zip",
                            mime="application/zip"
                        )
                else:
                    st.warning("No tables found in the image. Please ensure the image contains clear tabular data.")
                    # Display the raw response for debugging
                    with st.expander("Debug Information"):
                        st.json(result)
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.write("Full error:", str(e))

if __name__ == "__main__":
    main()