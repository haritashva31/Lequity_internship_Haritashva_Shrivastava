# Task 3: OCR-based PDF Text Extraction and Question-Answering Chatbot

This project extracts text from a PDF using OCR (Tesseract and EasyOCR), processes it into JSON format, 
and builds a simple chatbot for question answering based on the extracted data.

## Steps:
1. Convert the PDF pages into images.
2. Perform OCR to extract text from each page.
3. Structure the extracted text into JSON format.
4. Implement a chatbot that searches the JSON for relevant answers.

### Instructions
1. Install the dependencies using `pip install -r requirements.txt`.
2. Run the script: `python app.py`.
3. Ask questions based on the content of the PDF.

### Example
Input PDF: `Jamin Learning Doc.pdf`.
Output: `output_text.txt` (extracted text) and `output_text.json` (JSON-formatted text).

