from PIL import Image
import cv2
import numpy as np
import imutils
from imutils import perspective
from fastapi import FastAPI, UploadFile, File, HTTPException
from paddleocr import PaddleOCR
from fastapi.responses import JSONResponse
import re  # Import regular expressions for text processing

# Initialize FastAPI
app = FastAPI()

# Initialize PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang='en')

def process_upload_file(upload_file: UploadFile) -> np.ndarray:
    try:
        image = Image.open(upload_file.file)
        image = image.convert('RGB')  # Ensure it's in RGB format
        image_np = np.array(image)
        print(f"Processed image type: {type(image_np)}, shape: {image_np.shape}")
        return image_np
    except Exception as e:
        print(f"Error in processing upload file: {e}")
        raise HTTPException(status_code=400, detail="Invalid image file")

def preprocess_and_blur(image: np.ndarray) -> np.ndarray:
    # Check if the image is already grayscale
    if len(image.shape) == 2:
        grayscale_image = image  # Already grayscale
    elif len(image.shape) == 3:
        grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError("Unsupported image shape")

    # Apply Gaussian blur
    blurred_image = cv2.GaussianBlur(grayscale_image, (5, 5), 0)

    return blurred_image


def extract_text_from_image(image: np.ndarray) -> list:
    if not isinstance(image, np.ndarray):
        raise ValueError("The image must be a valid NumPy array.")
    print(f"Image type before OCR: {type(image)}, shape: {image.shape}")

    blurred_image = preprocess_and_blur(image)
    
    # Perform OCR on the processed image
    result = ocr.ocr(blurred_image, cls=True)
    
    # Parse OCR results into a list of text lines
    result_list = [res[1][0] for line in result for res in line]
    
    print(f"Extracted text: {result_list}")
    return result_list

def find_closest_number(text_list, keyword, max_distance=3):
    keyword_idx = None
    for i, text in enumerate(text_list):
        if keyword in text.upper():
            keyword_idx = i
            break

    if keyword_idx is not None:
        for offset in range(1, max_distance + 1):
            # Look before and after the keyword within max_distance
            if keyword_idx - offset >= 0:
                before_text = text_list[keyword_idx - offset]
                before_number = re.findall(r'\d+', before_text)
                if before_number:
                    return before_number[0]

            if keyword_idx + offset < len(text_list):
                after_text = text_list[keyword_idx + offset]
                after_number = re.findall(r'\d+', after_text)
                if after_number:
                    return after_number[0]

    return None


# Optimized function to extract vehicle logbook data
def extract_vehicle_logbook_data(text_list: list) -> dict:
    info = {}
    registration_counter = 0
    
    for i, text in enumerate(text_list):
        upper_text = text.upper()
        
        if "ORIGINAL NO" in upper_text:
            info['ORIGINAL NO'] = find_closest_number(text_list, 'ORIGINAL NO')
        elif "MAN YEAR" in upper_text:
            info['MAN YEAR'] = find_closest_number(text_list, 'MAN YEAR')
        elif "RATING" in upper_text:
            info['RATING'] = find_closest_number(text_list, 'RATING')
        elif "PASSENGERS" in upper_text:
            info['PASSENGERS'] = find_closest_number(text_list, 'PASSENGERS')
        elif "AXLES" in upper_text:
            info['AXLES'] = find_closest_number(text_list, 'AXLES')
        elif "LOAD CAPACITY" in upper_text:
            info['LOAD CAPACITY'] = find_closest_number(text_list, 'LOAD CAPACITY')
        elif "CODE" in upper_text:
            info['CODE'] = find_closest_number(text_list, 'CODE')
        elif "ENTRY NO" in upper_text or "ENTRYNO" in upper_text:
            info['ENTRY NO'] = text_list[i + 1] if i + 1 < len(text_list) else None
        elif "REGISTRATION" in upper_text or "ROGISTRATION" in upper_text or "RCGISTRATION" in upper_text:
            if "REGISTRATION CERTIFICATE" not in upper_text:
                registration_counter += 1
                if registration_counter == 1:
                    info['REGISTRATION'] = text_list[i + 1] if i + 1 < len(text_list) else None
        elif "CHASSIS/FRAME" in upper_text or "FRAME" in upper_text or "CHASSIS:" in upper_text:
            info['CHASSIS/FRAME'] = text_list[i + 1] if i + 1 < len(text_list) else None
        elif "MAKE" in upper_text or "MAKC" in upper_text or "MAKO" in upper_text:
            info['MAKE'] = text_list[i + 1] if i + 1 < len(text_list) else None
        elif "MODEL" in upper_text or "MODCL" in upper_text or "MODOL" in upper_text:
            info['MODEL'] = text_list[i + 1] if i + 1 < len(text_list) else None
        elif "TYPE" in upper_text or "TYPC" in upper_text or "TYPO" in upper_text:
            info['TYPE'] = text_list[i + 1] if i + 1 < len(text_list) else None
        elif "BODY" in upper_text:
            info['BODY'] = text_list[i + 1] if i + 1 < len(text_list) else None
        elif "FUEL" in upper_text or "FUCL" in upper_text or "FUOL" in upper_text:
            if i + 1 < len(text_list):
                next_word = text_list[i + 1].strip().lower()
                if 'petrol' in next_word:
                    info['FUEL'] = 'Petrol'
                elif 'diesel' in next_word:
                    info['FUEL'] = 'Diesel'
                elif 'electricity' in next_word:
                    info['FUEL'] = 'Electricity'
                else:
                    info['FUEL'] = next_word
        elif "ENGINE NO" in upper_text or "ENGINC NO" in upper_text or "ENGINO NO" in upper_text:
            info['ENGINE NO'] = text_list[i + 1] if i + 1 < len(text_list) else None
        elif "COLOR" in upper_text:
            info['COLOR'] = text_list[i + 1] if i + 1 < len(text_list) else None
        elif "REGDATE" in upper_text or "REGDATC" in upper_text or "REGDATO" in upper_text or "REGDATER" in upper_text:
            info['REGDATE'] = text_list[i + 1] if i + 1 < len(text_list) else None
        elif "GROSSWEIGHT" in upper_text  or "GROSSWOIGHT" in upper_text or "GROSS WOIGHT:" in upper_text:
            info['GROSS WEIGHT'] = text_list[i + 1] if i + 1 < len(text_list) else None
        elif "DUTY" in upper_text:
            info['DUTY'] = text_list[i + 1] if i + 1 < len(text_list) else None
        elif "TARE WEIGHT" in upper_text:
            info['TARE WEIGHT'] = text_list[i + 1] if i + 1 < len(text_list) else None
        elif "TAX CLASS" in upper_text:
            info['TAX CLASS'] = text_list[i + 1] if i + 1 < len(text_list) else None
        elif "PREVIOUS REG. COUNTRY" in upper_text:
            info['PREVIOUS REG. COUNTRY'] = text_list[i + 1] if i + 1 < len(text_list) else None
        elif "PIN" in upper_text:
            if i + 1 < len(text_list):
                pin_value = text_list[i + 1]
                info['PIN'] = pin_value[:11]
        elif "NAME" in upper_text:
            info['NAME'] = text_list[i + 1] if i + 1 < len(text_list) else None
        elif "BOXNO" in upper_text:
            info['BOXNO'] = text_list[i + 1] if i + 1 < len(text_list) else None
        elif "TOWN" in upper_text:
            info['TOWN'] = text_list[i + 1] if i + 1 < len(text_list) else None
    
    return info

# Optimized function to extract ID card data
def extract_id_card_data(text_list: list) -> dict:
    info = {}

    info['ID NUMBER'] = find_closest_number(text_list, 'ON')

    # Extract other fields based on keywords
    for i, text in enumerate(text_list):
        upper_text = text.upper()
        if "FULL" in upper_text and "NAMES" in upper_text:
            info['FULL NAMES'] = text_list[i + 1] if i + 1 < len(text_list) else None
        elif "BIRTH" in upper_text and "DATE" in upper_text:
            info['DATE OF BIRTH'] = text_list[i + 1] if i + 1 < len(text_list) else None
        elif "SEX" in upper_text:
            info['SEX'] = text_list[i + 1] if i + 1 < len(text_list) else None
        elif "DISTRICT" in upper_text and "OF BIRTH" in upper_text:
            info['DISTRICT OF BIRTH'] = text_list[i + 1] if i + 1 < len(text_list) else None
        elif "PLACE" in upper_text and "OF ISSUE" in upper_text:
            info['PLACE OF ISSUE'] = text_list[i + 1] if i + 1 < len(text_list) else None
        elif "ISSUE" in upper_text and "DATE" in upper_text:
            info['DATE OF ISSUE'] = text_list[i + 1] if i + 1 < len(text_list) else None

    return info

@app.post("/extract_id_card/")
async def extract_id_card(file: UploadFile = File(...)):
    try:
        image = process_upload_file(file)
        # processed_image = preprocess_and_blur(image)
        text_results = extract_text_from_image(image)
        extracted_data = extract_id_card_data(text_results)
        return JSONResponse(content=extracted_data)
    except Exception as e:
        print(f"Error occurred: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/extract_vehicle_logbook/")
async def extract_vehicle_logbook(file: UploadFile = File(...)):
    try:
        image = process_upload_file(file)
        processed_image = preprocess_and_blur(image)
        text_results = extract_text_from_image(processed_image)
        extracted_data = extract_vehicle_logbook_data(text_results)
        return JSONResponse(content=extracted_data)
    except Exception as e:
        print(f"Error occurred: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Run the application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
