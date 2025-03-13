import streamlit as st
import os
from mistralai import Mistral
import pandas as pd
from PIL import Image
from dotenv import load_dotenv  

load_dotenv()

st.set_page_config(
    page_title="Image to CSV Converter",
    page_icon="📊",
    layout="wide"
)

@st.cache_resource
def get_mistral_client():
    api_key = os.getenv("MISTRAL_API_KEY")  
    if not api_key:
        raise ValueError("MISTRAL_API_KEY is not set in the .env file.")
    return Mistral(api_key=api_key)

def process_image(image_bytes):
    import base64
    client = get_mistral_client()
    
    image_base64 = base64.b64encode(image_bytes).decode('utf-8')
    
    result = client.ocr.process(
        model="mistral-ocr-latest",
        document={
            "type": "image_url",
            "image_url": f"data:image/jpeg;base64,{image_base64}"
        }
    )
    
    return result

def parse_ocr_response(result):
    tables = []
    if hasattr(result, 'pages') and len(result.pages) > 0:
        markdown_content = result.pages[0].markdown
        table_blocks = markdown_content.split('\n\n')
        
        for table_block in table_blocks:
            if '|' in table_block:  # Verify it's a table
                try:
                    lines = [line.strip() for line in table_block.split('\n') if line.strip()]
                    lines = [line for line in lines if ':--:' not in line]
                    
                    if len(lines) >= 2:  
                        header = [col.strip() for col in lines[0].strip('|').split('|')]
                        header = [col for col in header if col]  
                        
                        unique_headers = []
                        seen = {}
                        for h in header:
                            if h in seen:
                                seen[h] += 1
                                unique_headers.append(f"{h}_{seen[h]}")
                            else:
                                seen[h] = 0
                                unique_headers.append(h)
                        
                        data = []
                        for line in lines[1:]:
                            row = [cell.strip() for cell in line.strip('|').split('|')]
                            row = [cell for cell in row if cell]  
                            if len(row) > len(unique_headers):
                                row = row[:len(unique_headers)]
                            elif len(row) < len(unique_headers):
                                row.extend([''] * (len(unique_headers) - len(row)))
                            data.append(row)
                        
                        df = pd.DataFrame(data, columns=unique_headers)
                        tables.append(df)
                except Exception as e:
                    st.warning(f"Skipped invalid table: {str(e)}")
    
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
                result = process_image(uploaded_file.getvalue())
                
                tables = parse_ocr_response(result)
                
                if tables:
                    with col2:
                        st.subheader(f"Found {len(tables)} Tables")
                        
                        import io
                        import zipfile
                        
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
                    st.error("No tables found in the image")
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.write("Full error:", str(e))

if __name__ == "__main__":
    main()