# AI-VIRTUAL-INTERVIEWER-SYSTEM
# An AI-powered virtual interviewer that generates questions, analyzes voice/text answers, and gives instant feedback using Google Gemini, NLP, and WebRTC. Built with Python and Streamlit to automate interview practice and initial screening efficiently.
import streamlit as st
import google.generativeai as genai
import speech_recognition as sr
import tempfile
import os
import json
import logging
import subprocess
from pathlib import Path

# Stage 2: Page Configuration
logging.basicConfig(level=logging.INFO)
st.set_page_config(page_title="AI Virtual Interviewer", page_icon="🎤", layout="wide")
st.title("🤖 AI Virtual Interviewer")

# Stage 3: Gemini API Setup
try:
    # Try to get key from secrets, otherwise handle gracefully
    if "GEMINI_API_KEY" in st.secrets:
        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
        model = genai.GenerativeModel("gemini-2.5-flash")
        GEMINI_READY = True
    else:
        st.error("GEMINI_API_KEY not found in secrets.")
        GEMINI_READY = False
except Exception as e:
    GEMINI_READY = False
    st.error(f"Gemini setup failed: {e}")

# Stage 4: Session State Initialization
if "questions" not in st.session_state:
    st.session_state.questions = []
if "current_q" not in st.session_state:
    st.session_state.current_q = 0
if "feedback" not in st.session_state:
    st.session_state.feedback = None
if "setup_done" not in st.session_state:
    st.session_state.setup_done = False
if "mode" not in st.session_state:
    st.session_state.mode = "Text ✎"  # Default
if "name" not in st.session_state:
    st.session_state.name = ""

# Stage 5: Helper Functions
def generate_questions(role, difficulty):
    """Generates interview questions using Gemini."""
    prompt = f"""
    Generate 4 professional interview questions for a candidate applying for a '{role}' role at '{difficulty}' difficulty.
    Return output as a JSON array with 'id' and 'text' fields.
    """
    if not GEMINI_READY:
        return [{"id": "q1", "text": "Tell me about yourself."}]
    try:
        response = model.generate_content(prompt)
        clean_json = response.text.strip().lstrip("```json").rstrip("```")
        return json.loads(clean_json)
    except Exception as e:
        st.error(f"Error generating questions: {e}")
        return [{"id": "q1", "text": "Tell me about yourself."}]

def transcribe_audio(audio_file_path):
    """
    Transcribes audio using Google Speech Recognition.
    Converts to compatible WAV format first using ffmpeg.
    """
    recognizer = sr.Recognizer()
    
    # Define a temp path for the converted file
    tmp_wav = f"{audio_file_path}_converted.wav"
    
    try:
        # Use ffmpeg to convert input to 16k mono wav (best for recognition)
        subprocess.run([
            "ffmpeg", "-y", 
            "-i", audio_file_path, 
            "-ac", "1", 
            "-ar", "16000", 
            tmp_wav
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # Use the converted file
        process_path = tmp_wav
    except Exception as e:
        logging.warning(f"FFmpeg conversion failed: {e}. Trying original file.")
        process_path = audio_file_path

    try:
        with sr.AudioFile(process_path) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data)
            
        # Cleanup converted file
        if os.path.exists(tmp_wav) and tmp_wav != audio_file_path:
            os.remove(tmp_wav)
            
        return text
    except sr.UnknownValueError:
        return None # Signal that audio wasn't understood
    except Exception as e:
        return f"Error: {str(e)}"

def analyze_with_gemini(answer_text, question_text):
    """Sends transcript to Gemini for feedback."""
    st.info("Analyzing your response...")
    prompt = f"""
    You are an expert interview coach.
    **Question:** "{question_text}"
    **Candidate Answer:** "{answer_text}"

    Provide feedback in Markdown:
    1. **Clarity & Confidence**
    2. **Content Quality**
    3. **Improvement Tips**
    4. ** Answer I want from candiadate **
    """
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error getting feedback: {e}"

# Stage 6: Main UI Logic
if not st.session_state.setup_done:
    st.subheader("Interview Setup")
    
    col1, col2 = st.columns(2)
    with col1:
        st.session_state.name = st.text_input("Full Name")
        field = st.selectbox("Field", ["Student", "Professional"])
    with col2:
        role = st.selectbox("Role", ["Software Engineer", "Data Analyst", "Product Manager"])
        difficulty = st.selectbox("Difficulty", ["Beginner", "Intermediate", "Advanced"])

    # Mode Selection
    mode = st.radio("Response Mode", ["Text ✎", "Voice 🎙️ (Real-time)"], index=1)

    if st.button("🚀 Start Interview"):
        if not st.session_state.name:
            st.error("Please enter your name.")
        elif not GEMINI_READY:
            st.error("Gemini API key missing.")
        else:
            with st.spinner("Generating questions..."):
                st.session_state.questions = generate_questions(role, difficulty)
            st.session_state.mode = mode
            st.session_state.setup_done = True
            st.rerun()

else:
    # Stage 7: Question & Answer Loop
    q_idx = st.session_state.current_q
    questions = st.session_state.questions
    
    if q_idx < len(questions):
        q_data = questions[q_idx]
        st.markdown(f"### 🎯 Question {q_idx + 1}/{len(questions)}")
        st.info(f"**{q_data['text']}**")
        
        # Check if we already have feedback for this question
        if st.session_state.feedback:
            st.success("Analysis Complete!")
            st.markdown("### 🤖 Gemini Feedback")
            st.markdown(st.session_state.feedback)
            
            if st.button("Next Question ➡️"):
                st.session_state.feedback = None
                st.session_state.current_q += 1
                st.rerun()
        else:
            # Input Area based on Mode
            if "Voice" in st.session_state.mode:
                st.write("🎙️ **Record your answer below:**")
                st.caption("Click the mic to start. Click again to stop. Feedback generates automatically.")
                
                # THE NEW NATIVE WIDGET - No WebRTC complexity needed
                audio_value = st.audio_input("Record Answer", key=f"audio_{q_idx}")
                
                if audio_value:
                    # Logic: Audio exists = User finished recording
                    with st.spinner("Transcribing audio..."):
                        # Save to temp file
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                            tmp_file.write(audio_value.read())
                            tmp_path = tmp_file.name

                        # Transcribe
                        transcript = transcribe_audio(tmp_path)
                        
                        # Clean up temp file
                        os.remove(tmp_path)

                        if transcript:
                            st.write("🗣️ **You said:**")
                            st.info(f'"{transcript}"')
                            
                            # Generate Feedback
                            feedback = analyze_with_gemini(transcript, q_data['text'])
                            st.session_state.feedback = feedback
                            st.rerun()
                        else:
                            st.warning("Could not understand audio. Please try again.")
            
            else:
                # Text Mode
                answer = st.text_area("Type your answer here...")
                if st.button("Submit"):
                    if answer.strip():
                        feedback = analyze_with_gemini(answer, q_data['text'])
                        st.session_state.feedback = feedback
                        st.rerun()
                    else:
                        st.warning("Please write an answer first.")

    else:
        st.balloons()
        st.success(f"Interview Complete! Great job, {st.session_state.name}.")
        if st.button("Start Over"):
            st.session_state.clear()
            st.rerun()

# Sidebar Info
st.sidebar.markdown("### Debug Info")
st.sidebar.text(f"Mode: {st.session_state.get('mode', 'Not set')}")
st.sidebar.text(f"Gemini: {'Active' if GEMINI_READY else 'Inactive'}")
