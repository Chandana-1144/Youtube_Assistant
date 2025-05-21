import streamlit as st
import whisper
import os
import re
import tempfile
import time
import traceback
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from pytube import YouTube
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
import groq
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set page configuration
st.set_page_config(
    page_title="YouTube Video Assistant",
    page_icon="🎬",
    layout="wide"
)

# Initialize session state variables
if 'transcript' not in st.session_state:
    st.session_state.transcript = None
if 'db' not in st.session_state:
    st.session_state.db = None
if 'video_title' not in st.session_state:
    st.session_state.video_title = None
if 'transcript_source' not in st.session_state:
    st.session_state.transcript_source = None
if 'debug_info' not in st.session_state:
    st.session_state.debug_info = []

# Load whisper model - only when needed
@st.cache_resource
def load_whisper_model():
    try:
        with st.spinner("Loading Whisper model (this may take a while on first run)..."):
            return whisper.load_model("base")
    except Exception as e:
        st.error(f"Failed to load Whisper model: {str(e)}")
        st.session_state.debug_info.append(f"Whisper model load error: {str(e)}")
        return None

# Function to extract YouTube ID
def get_video_id(url):
    # Handle various YouTube URL formats
    patterns = [
        r"(?:v=|\/)([0-9A-Za-z_-]{11}).*",  # Standard and shortened
        r"(?:embed\/)([0-9A-Za-z_-]{11})",  # Embed URLs
        r"(?:youtu\.be\/)([0-9A-Za-z_-]{11})"  # youtu.be URLs
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    
    return None

# Function to download audio
def download_audio(youtube_url):
    try:
        with st.spinner("Downloading audio from YouTube..."):
            yt = YouTube(youtube_url)
            audio_stream = yt.streams.filter(only_audio=True).first()
            
            if not audio_stream:
                st.error("No audio stream available for this video")
                st.session_state.debug_info.append("No audio stream available")
                return None, None
            
            # Use a temporary file
            temp_dir = tempfile.gettempdir()
            temp_path = os.path.join(temp_dir, "audio.mp4")
            
            # Ensure previous files are cleaned up
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
            audio_stream.download(output_path=temp_dir, filename="audio.mp4")
            
            if not os.path.exists(temp_path):
                st.error("Failed to save audio file")
                st.session_state.debug_info.append("Audio file not saved")
                return None, None
                
            return temp_path, yt.title
    except Exception as e:
        st.error(f"Error downloading audio: {str(e)}")
        st.session_state.debug_info.append(f"Audio download error: {str(e)}\n{traceback.format_exc()}")
        return None, None

# Function to transcribe with Whisper
def transcribe_with_whisper(audio_path):
    try:
        with st.spinner("Transcribing audio with Whisper AI (this may take a while)..."):
            model = load_whisper_model()
            if not model:
                return None
                
            result = model.transcribe(audio_path)
            return result["text"]
    except Exception as e:
        st.error(f"Whisper transcription error: {str(e)}")
        st.session_state.debug_info.append(f"Whisper error: {str(e)}\n{traceback.format_exc()}")
        return None

# Function to get transcript
def get_transcript(youtube_url):
    video_id = get_video_id(youtube_url)
    if not video_id:
        st.error("Invalid YouTube URL or couldn't extract video ID")
        st.session_state.debug_info.append(f"Invalid URL: {youtube_url}")
        return None, None, "Error"
    
    st.session_state.debug_info.append(f"Extracted video ID: {video_id}")
    
    # First try: YouTube Transcript API
    try:
        with st.spinner("Fetching transcript from YouTube..."):
            transcript = YouTubeTranscriptApi.get_transcript(video_id)
            full_text = " ".join([t["text"] for t in transcript])
            
            # Only get title if transcript was successful
            try:
                yt = YouTube(youtube_url)
                video_title = yt.title
            except Exception as e:
                st.warning(f"Got transcript but couldn't get video title: {str(e)}")
                video_title = f"Video {video_id}"
                
            st.session_state.debug_info.append("Successfully retrieved YouTube transcript")
            return full_text, video_title, "YouTube Transcript API"
            
    except TranscriptsDisabled:
        st.warning("Transcripts are disabled for this video. Falling back to Whisper AI...")
        st.session_state.debug_info.append("TranscriptsDisabled error")
    except NoTranscriptFound:
        st.warning("No transcript found for this video. Falling back to Whisper AI...")
        st.session_state.debug_info.append("NoTranscriptFound error")
    except Exception as e:
        st.warning(f"Error getting YouTube transcript: {str(e)}. Falling back to Whisper AI...")
        st.session_state.debug_info.append(f"YouTube transcript error: {str(e)}")
    
    # Second try: Whisper AI
    try:
        audio_file, video_title = download_audio(youtube_url)
        if not audio_file:
            return None, None, "Error"
            
        full_text = transcribe_with_whisper(audio_file)
        
        # Clean up the temporary file
        if os.path.exists(audio_file):
            try:
                os.remove(audio_file)
            except Exception as e:
                st.session_state.debug_info.append(f"Failed to remove temp file: {str(e)}")
        
        if full_text:
            st.session_state.debug_info.append("Successfully transcribed with Whisper")
            return full_text, video_title, "Whisper AI"
        else:
            st.error("Whisper transcription failed")
            return None, None, "Error"
    except Exception as ex:
        st.error(f"Failed to transcribe with Whisper: {str(ex)}")
        st.session_state.debug_info.append(f"Whisper process error: {str(ex)}\n{traceback.format_exc()}")
        return None, None, "Error"

# Function to process transcript
def process_transcript(transcript_text):
    try:
        with st.spinner("Processing transcript..."):
            # Split text into chunks
            text_splitter = CharacterTextSplitter(
                separator="\n",
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
            )
            
            texts = text_splitter.split_text(transcript_text)
            
            # Initialize embedding model
            embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
            
            # Create FAISS vector store
            db = FAISS.from_texts(texts, embedding_model)
            
            return db, texts
    except Exception as e:
        st.error(f"Error processing transcript: {str(e)}")
        st.session_state.debug_info.append(f"Transcript processing error: {str(e)}\n{traceback.format_exc()}")
        return None, None

# Initialize Groq client
def get_groq_client():
    try:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            st.error("GROQ_API_KEY environment variable not found. Please check your .env file.")
            st.session_state.debug_info.append("GROQ_API_KEY not found in environment variables")
            return None
            
        client = groq.Client(api_key=api_key)
        # Test the API key with a simple request
        test_completion = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[
                {"role": "user", "content": "Hello, testing connection"}
            ],
            max_tokens=10,
        )
        return client
    except Exception as e:
        st.error(f"Error initializing Groq client: {str(e)}")
        st.session_state.debug_info.append(f"Groq client error: {str(e)}")
        return None

# Function to ask question using Groq directly
def ask_question_with_groq(query, context, client):
    try:
        prompt = f"""
You are a helpful YouTube video assistant. Your goal is to answer questions based on the transcript of a YouTube video.

Context from the video transcript:
{context}

Human Question: {query}

When answering, use only information from the transcript context. If the transcript doesn't contain the answer, politely state that the information isn't available in the video. Be concise and accurate.

Answer:
"""
        
        completion = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[
                {"role": "system", "content": "You are a helpful YouTube video assistant that answers questions based on video transcripts."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=1024,
        )
        
        return completion.choices[0].message.content
    except Exception as e:
        st.error(f"Error getting answer from Groq: {str(e)}")
        st.session_state.debug_info.append(f"Groq answer error: {str(e)}")
        return "Sorry, there was an error processing your question. Please try again."

# UI Layout
st.title("🎬 YouTube Video Assistant")
st.markdown("""
This app helps you get answers from YouTube videos. 
1. Enter a YouTube URL
2. Wait for the transcript to be processed
3. Ask questions about the video content
""")

# Sidebar for URL input only (API key removed)
with st.sidebar:
    st.header("Configuration")
    youtube_url = st.text_input("YouTube URL:", placeholder="https://www.youtube.com/watch?v=...")
    
    if st.button("Process Video", disabled=not youtube_url):
        # Clear previous debug info
        st.session_state.debug_info = []
        
        if not youtube_url:
            st.error("Please enter a YouTube URL.")
        else:
            # Get Groq client
            client = get_groq_client()
            if not client:
                st.error("Failed to initialize Groq client. Please check your .env file.")
            else:
                # Get transcript
                transcript, video_title, source = get_transcript(youtube_url)
                
                if transcript:
                    st.session_state.transcript = transcript
                    st.session_state.video_title = video_title
                    st.session_state.transcript_source = source
                    
                    # Process transcript
                    db, texts = process_transcript(transcript)
                    if db and texts:
                        st.session_state.db = db
                        st.session_state.texts = texts
                        st.success("Video processed successfully!")
                    else:
                        st.error("Failed to process transcript.")
                else:
                    st.error("Failed to generate transcript. Check debug information.")
    
    if st.session_state.video_title:
        st.subheader("Current Video")
        st.write(f"**Title:** {st.session_state.video_title}")
        st.write(f"**Transcript source:** {st.session_state.transcript_source}")
    
    # Debug information section
    with st.expander("Debug Information"):
        if st.session_state.debug_info:
            for i, info in enumerate(st.session_state.debug_info):
                st.text(f"{i+1}. {info}")
        else:
            st.text("No debug information available.")

# Main area for Q&A
if st.session_state.transcript and st.session_state.db:
    # Show transcript and Q&A in tabs
    tab1, tab2 = st.tabs(["Ask Questions", "View Transcript"])
    
    with tab1:
        st.header("Ask a Question About the Video")

    user_question = st.text_input("Enter your question:", placeholder="What is the video about?")

    if st.button("Get Answer", disabled=not user_question):
        # Get top k relevant chunks using FAISS
        docs = st.session_state.db.similarity_search(user_question, k=4)
        context = "\n\n".join([doc.page_content for doc in docs])

        groq_client = get_groq_client()
        if groq_client:
            response = ask_question_with_groq(user_question, context, groq_client)
            st.markdown("### 🧠 Answer")
            st.write(response)
                        
    with tab2:
        st.header("Video Transcript")
        st.text_area("Full transcript:", value=st.session_state.transcript, height=400, disabled=True)
else:
    st.info("Enter a YouTube URL in the sidebar and click 'Process Video' to start.")

# Footer
st.divider()
st.caption("YouTube Video Assistant powered by Groq & Llama 3.1")