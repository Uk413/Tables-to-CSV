import os
from dotenv import load_dotenv
from mistralai import Mistral
load_dotenv()

api_key = os.getenv("MISTRAL_API_KEY")
client = Mistral(api_key=api_key)

ocr_response = client.ocr.process(
    model="mistral-ocr-latest",
    document={
        "type": "image_url",
        "image_url": "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fwww.visualwatermark.com%2Fimages%2Fadd-text-to-photos%2Fadd-text-to-image-3.webp&f=1&nofb=1&ipt=e4709f8c418d3d93d1aa75ed3dadbdb7824a0dbdd58b3234fa349d4d7937bb3c&ipo=images"
    }
)

print(ocr_response)