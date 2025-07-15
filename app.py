
# Resume Match and Ranking Script (Web GUI with AI + Deployable)
# This version includes real-time OpenAI summaries and is ready for Streamlit Cloud deployment

import pandas as pd
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
import openai
import os

# ---------------------------
# CONFIGURATION
# ---------------------------
openai.api_key = st.secrets["OPENAI_API_KEY"] if "OPENAI_API_KEY" in st.secrets else os.getenv("OPENAI_API_KEY")

# ---------------------------
# UTILITY FUNCTIONS
# ---------------------------

def extract_text_from_pdf(uploaded_file):
    pdf_reader = PdfReader(uploaded_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""
    return text

def calculate_similarity(text, jd_text):
    vectorizer = TfidfVectorizer(stop_words='english')
    vectors = vectorizer.fit_transform([jd_text, text])
    return cosine_similarity(vectors[0:1], vectors[1:2])[0][0] * 100

def generate_ai_summary(text):
    prompt = f"""
    Analyze the following resume content:

    {text[:4000]}

    Provide a summary with:
    - Key strengths
    - Weaknesses
    - Technologies used
    - Project experience
    Keep it clear and concise.
    """
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a professional technical recruiter."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"AI Summary failed: {e}"

# ---------------------------
# STREAMLIT INTERFACE
# ---------------------------

def main():
    st.set_page_config(page_title="Resume Matcher & AI Analyzer", layout="centered")
    st.title("📄 Resume Matcher with AI Insights")

    with st.sidebar:
        st.info("Upload a Job Description and multiple PDF resumes.")
        jd_file = st.file_uploader("Job Description (TXT)", type="txt")
        resume_files = st.file_uploader("Resume PDFs", type="pdf", accept_multiple_files=True)

    if jd_file and resume_files:
        jd_text = jd_file.read().decode("utf-8")
        results = []

        with st.spinner("Processing resumes..."):
            for resume in resume_files:
                try:
                    text = extract_text_from_pdf(resume)
                    match_score = calculate_similarity(text, jd_text)
                    ai_summary = generate_ai_summary(text)

                    results.append({
                        "Candidate": resume.name,
                        "Match %": round(match_score, 2),
                        "AI Summary": ai_summary
                    })
                except Exception as e:
                    st.error(f"Error processing {resume.name}: {e}")

        if results:
            df = pd.DataFrame(results)
            st.subheader("📊 Match Scores")
            st.dataframe(df[["Candidate", "Match %"]].sort_values(by="Match %", ascending=False))

            st.subheader("🤖 AI Summaries")
            for res in results:
                st.markdown(f"### {res['Candidate']}")
                st.markdown(res['AI Summary'])

if __name__ == "__main__":
    main()
