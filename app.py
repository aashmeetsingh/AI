import torch
import pickle
import json
import os
import fitz  # PyMuPDF for PDFs
import docx
import io
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from model.model import NeuralNet
from utils.nltk_utils import preprocess_text
from utils.gemini_feedback import generate_resume_feedback
from sklearn.feature_extraction.text import TfidfVectorizer

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load vectorizer
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Load categories from dataset
with open("data/resume_data.json", "r") as f:
    resume_data = json.load(f)
    categories = sorted(set(resume["category"] for resume in resume_data["resumes"]))

# Model parameters
input_size = len(vectorizer.get_feature_names_out())
hidden_size = 32
output_size = len(categories)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load trained model
model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(torch.load("resume_model.pth", map_location=device))
model.eval()

# Function to truncate text to 500 words
def truncate_text(text, max_words=500):
    words = text.split()
    return " ".join(words[:max_words]) + ("..." if len(words) > max_words else "")

# Function to extract text from different file formats
def extract_text_from_file(file: UploadFile):
    """Extract text from PDF, DOCX, and TXT files."""
    file_extension = file.filename.split(".")[-1].lower()

    if file_extension == "txt":
        return file.file.read().decode("utf-8")

    elif file_extension == "pdf":
        with fitz.open(stream=file.file.read(), filetype="pdf") as pdf_reader:
            text = "\n".join([page.get_text("text") for page in pdf_reader])
        print(f"Extracted PDF text: {text}")  # Log the extracted text
        return text if text.strip() else "Error: Could not extract text."

    elif file_extension == "docx":
        doc = docx.Document(io.BytesIO(file.file.read()))
        text = "\n".join([para.text for para in doc.paragraphs])
        return text if text.strip() else "Error: Could not extract text."

    else:
        return "Error: Unsupported file format."

# Function to classify resumes
def classify_resume(resume_text):
    processed_text = preprocess_text(resume_text)
    processed_text = " ".join(processed_text)
    vectorized_text = vectorizer.transform([processed_text]).toarray()
    input_tensor = torch.tensor(vectorized_text, dtype=torch.float32).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = torch.max(output, dim=1)

    return categories[predicted.item()]

@app.post("/analyze_resume/")
async def analyze_resume_api(
    resume: UploadFile = File(None),
    user_type: str = Form("job_seeker"),
    job_input: str = Form(None),
    input_type: str = Form("full_description")
):
    try:
        # Extract resume text if a file is uploaded
        resume_text = ""
        if user_type == "recruiter":
            if resume:
                resume_text = extract_text_from_file(resume)
                print(f"Recruiter: Extracted PDF text: {resume_text}")  # Log for recruiter
                print(f"Recruiter: Resume text passed to feedback: {resume_text}") # Log for recruiter
                if resume_text.startswith("Error:"):
                    return {"error": resume_text}
                print(f"Recruiter: job_input: {job_input}")  # Log job_input
                print(f"Recruiter: input_type: {input_type}") # Log input_type
                feedback = generate_resume_feedback(resume_text, user_type, job_input, input_type)
                predicted_category = classify_resume(resume_text) if resume_text else "Unknown"
            else:
                return {"error": "Resume file required for recruiter view"}
        else:
            if resume:
                resume_text = extract_text_from_file(resume)
                if resume_text.startswith("Error:"):
                    return {"error": resume_text}
                predicted_category = classify_resume(resume_text)
                feedback = generate_resume_feedback(resume_text, user_type, job_input, input_type)
            else:
                return {"error": "Resume file required for job seeker view"}

        return {
            "predicted_category": predicted_category,
            "resume_feedback": feedback["resume_feedback"] 
        }

    except Exception as e:
        return {"error": f"Failed to process resume: {str(e)}"}