import torch
import pickle
import json
from model.model import NeuralNet
from utils.nltk_utils import preprocess_text
from sklearn.feature_extraction.text import TfidfVectorizer
from utils.gemini_feedback import generate_resume_feedback

# Load vectorizer
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Load categories from resume dataset
with open("data/resume_data.json", "r") as f:
    resume_data = json.load(f)
    categories = sorted(set(resume["category"] for resume in resume_data["resumes"]))

# Model setup
input_size = len(vectorizer.get_feature_names_out())
hidden_size = 32
output_size = len(categories)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(torch.load("resume_model.pth"))
model.eval()

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

def get_multiline_input(prompt_text):
    """Get multiline input from the user"""
    print(f"{prompt_text} (Type 'DONE' on a new line when finished):")
    lines = []
    while True:
        line = input()
        if line == "DONE":
            break
        lines.append(line)
    return "\n".join(lines)

def main():
    """Interactive CLI for the resume feedback tool"""
    print("Resume Feedback Tool")
    print("====================")
    
    user_type = input("Are you a job seeker or recruiter? ").lower()
    
    if user_type not in ["job_seeker", "recruiter"]:
        print("Invalid user type. Please enter 'job_seeker' or 'recruiter'.")
        return
    
    input_method = input("Would you like to [1] upload a file or [2] type your resume? (Enter 1 or 2): ")
    
    resume_text = ""
    
    if input_method == "1":
        resume_path = input("Enter the path to your resume text file: ")
        try:
            with open(resume_path, 'r') as file:
                resume_text = file.read()
        except FileNotFoundError:
            print(f"File not found: {resume_path}")
            return
    elif input_method == "2":
        resume_text = get_multiline_input("Please type your resume")
    else:
        print("Invalid choice. Please restart the program.")
        return
    
    # Check if resume was provided
    if not resume_text.strip():
        print("Error: Resume text is empty.")
        return
    
    job_input = None
    input_type = "full_description"
    
    if user_type == "recruiter":
        job_input_type = input("Would you like to use [1] a full job description or [2] job keywords? (Enter 1 or 2): ")
        
        if job_input_type == "1":
            input_type = "full_description"
            jd_input_method = input("Would you like to [1] upload a job description file or [2] type the job description? (Enter 1 or 2): ")
            
            if jd_input_method == "1":
                jd_path = input("Enter the path to the job description text file: ")
                try:
                    with open(jd_path, 'r') as file:
                        job_input = file.read()
                except FileNotFoundError:
                    print(f"File not found: {jd_path}")
                    return
            elif jd_input_method == "2":
                job_input = get_multiline_input("Please type the job description")
            else:
                print("Invalid choice. Please restart the program.")
                return
        elif job_input_type == "2":
            input_type = "keywords"
            job_input = input("Enter job keywords (comma-separated): ")
        else:
            print("Invalid choice. Please restart the program.")
            return
        
        # Check if job input was provided
        if not job_input.strip():
            print("Error: Job information is empty.")
            return
    
    print("\nGenerating feedback... This may take a moment.\n")
    feedback = generate_resume_feedback(resume_text, user_type, job_input, input_type)
    print("Feedback:")
    print("=========")
    print(feedback)

if __name__ == "__main__":
    main()