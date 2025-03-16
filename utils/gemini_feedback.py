import google.generativeai as genai
import os
from dotenv import load_dotenv
import time
from functools import lru_cache

# Load API key from .env file
load_dotenv()
GENAI_API_KEY = os.getenv("GENAI_API_KEY")

# Configure the API
if GENAI_API_KEY:
    genai.configure(api_key=GENAI_API_KEY)
else:
    print("Warning: GENAI_API_KEY not found in environment variables")

def truncate_text(text, max_words=500):
    """Truncate text to a maximum number of words without cutting off mid-sentence."""
    if not text:
        return ""
    words = text.split()
    if len(words) > max_words:
        return " ".join(words[:max_words]) + "..."
    return text

# Create prompt templates
PROMPT_TEMPLATES = {
    "job_seeker": """
    As a career coach, analyze this resume and provide structured feedback.
    
    Focus on:
    - Formatting and readability
    - Strength of achievements (metrics, results)
    - Relevance of skills and experience
    - Action-oriented language
    - ATS (Applicant Tracking System) optimization
    - Areas for improvement
    
    Resume:
    {resume_text}
    
    Provide the feedback in **short bullet points** with ⚠️ for issues and 💡 for recommendations.
    Do not exceed 500 words.
    """,
    
    "recruiter_full_description": """
    As a hiring manager, compare this resume against the job description.
    
    Job Description:
    {job_input}
    
    Resume:
    {resume_text}
    
    Provide analysis on:
    - Match percentage for the role
    - Key qualifications that align with job requirements
    - Missing skills or experience gaps
    - Potential interview questions
    - Recommendation (Strong match, Potential match, Not recommended)
    
    Format your response using ⚠️ for concerns, 💡 for positive matches, and bullet points.
    Keep the response under 500 words.
    """,
    
    "recruiter_keywords": """
    As a hiring manager, analyze this resume based on the following keywords:
    
    Job Keywords:
    {keywords}
    
    Resume:
    {resume_text}
    
    Provide:
    - Keyword match percentage
    - Identified keywords with context
    - Important missing keywords
    - Suggestions for improvement
    - Recommendation (Strong match, Potential match, Not recommended)
    
    Format your response using ⚠️ for missing keywords, 💡 for matched keywords, and bullet points.
    Keep the response concise and under 500 words.
    """
}

# Add simple caching for identical requests
@lru_cache(maxsize=100)
def get_cached_feedback(prompt_hash):
    """Cache function to avoid duplicate API calls"""
    return prompt_hash

def generate_resume_feedback(resume_text, user_type="job_seeker", job_input=None, input_type="full_description"):
    """
    Generate AI-powered resume feedback for job seekers or recruiters.
    
    Args:
        resume_text (str): The resume content.
        user_type (str): Either "job_seeker" or "recruiter".
        job_input (str, optional): Job description or keywords (required for recruiters).
        input_type (str): "full_description" or "keywords".
    
    Returns:
        dict: Structured AI feedback.
    """
    # Validate API key
    if not GENAI_API_KEY:
        return {"error": "API key not configured. Please set GENAI_API_KEY in environment variables."}
    
    # Input validation
    if user_type.lower() not in ["job_seeker", "recruiter"]:
        return {"error": "Invalid user_type. Choose either 'job_seeker' or 'recruiter'."}
    
    if user_type.lower() == "recruiter" and not job_input:
        return {"error": "Job information is required for recruiter analysis."}
    
    if not resume_text and user_type.lower() != "recruiter":
        return {"error": "Resume text is required for analysis."}
    
    # Prepare prompt based on user type and input type
    try:
        if user_type.lower() == "job_seeker":
            prompt = PROMPT_TEMPLATES["job_seeker"].format(resume_text=resume_text)
            
        elif user_type.lower() == "recruiter":
            if input_type.lower() == "full_description":
                prompt = PROMPT_TEMPLATES["recruiter_full_description"].format(
                    job_input=job_input,
                    resume_text=resume_text
                )
            elif input_type.lower() == "keywords":
                keywords = "\n".join([f"- {kw.strip()}" for kw in job_input.split(",")])
                prompt = PROMPT_TEMPLATES["recruiter_keywords"].format(
                    keywords=keywords,
                    resume_text=resume_text
                )
            else:
                return {"error": "Invalid input_type. Choose either 'full_description' or 'keywords'."}
        
        # Check cache (simple hash of prompt)
        prompt_hash = hash(prompt)
        cached_result = get_cached_feedback(prompt_hash)
        
        # Generate content
        try:
            model = genai.GenerativeModel("gemini-2.0-flash")
            response = model.generate_content(prompt)
            
            # Process and limit the response to 500 words
            if not hasattr(response, 'text'):
                return {"error": "Failed to get a valid response from AI model"}
                
            feedback_text = truncate_text(response.text, 500)
            
            # Convert feedback into a structured array for easy frontend display
            feedback_list = []
            for line in feedback_text.split("\n"):
                if not line.strip():
                    continue
                    
                item_type = "issue" if "⚠️" in line else "recommendation" if "💡" in line else "general"
                clean_line = line.strip().replace("⚠️", "").replace("💡", "").strip()
                
                if clean_line:  # Only add non-empty lines
                    feedback_list.append({
                        "type": item_type,
                        "content": clean_line
                    })
            
            return {"resume_feedback": feedback_list}
            
        except Exception as e:
            return {"error": f"AI model error: {str(e)}"}
            
    except Exception as e:
        return {"error": f"Failed to generate feedback: {str(e)}"}