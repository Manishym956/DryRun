# AI-Powered Interview Assistant

An interactive web application that conducts AI-driven technical interviews using video, audio, and text responses. Built with Streamlit and Google's Gemini AI.

## Features

- 🎥 Real-time video streaming for face-to-face interviews(under-progress)
- 🎙️ Voice-to-text transcription support
- 🤖 AI-generated interview questions based on role and difficulty
- 📊 Automated evaluation and scoring
- 📝 Downloadable PDF interview reports
- 🔊 Text-to-speech question reading

## Requirements

```txt
# Core Dependencies
streamlit>=1.37.0
streamlit-webrtc>=0.47.7
google-generativeai>=0.8.3
python-dotenv>=1.0.1

# Media Processing
aiortc>=1.9.0
av>=12.3.0

# Optional: Speech-to-Text Providers
deepgram-sdk>=3.7.6
assemblyai>=0.35.0
vosk>=0.3.45
azure-cognitiveservices-speech>=1.40.0

# PDF Generation
reportlab>=4.2.2
```

## Voice Interviewer (new)

A live spoken interview: the AI listens, asks follow-ups, adapts difficulty to your answers, and tailors questions to your resume and the job description. It's built on the Gemini Live API.

- **Coding round:** a problem and an editor appear on screen, and the interviewer reads your code and asks about it.
- **Replay:** the interview is recorded. In the report, click any question or caption to jump to that moment.
- **History:** scores over time, every past interview, and an AI summary of your recurring weak spots.
- **Reliability:** if the connection to Gemini drops, the interview reconnects and continues.

```sh
pip install -r requirements.txt
uvicorn server:app --reload
```

Open http://localhost:8000 in Chrome or Edge, allow the microphone, and use headphones for the best results.

- `server.py`: FastAPI app. It plans the interview, relays audio to Gemini Live, and evaluates the transcript.
- `interview.py`: interview state (adaptive difficulty, notes, transcript) and the interviewer prompt. Tests are in `test_interview.py`.
- `static/`: `index.html` (setup and live interview), `report.html` (results and replay), `history.html` (progress dashboard).
- Sessions and recordings are saved in `sessions/`.
- Optional `.env` settings:
  - `MAX_INTERVIEW_MINUTES` (default 30): the interviewer wraps up at this point, and the call ends 2 minutes later.
  - `SESSIONS_PER_HOUR` (default 10): interviews allowed per IP address per hour.
  - `LIVE_MODEL`, `TEXT_MODEL`: override the Gemini models.

There are no logins: anyone who can reach the server can start interviews on your API key and see every past interview. Keep it on localhost.

The original Streamlit app (`app.py`) still works with `streamlit run app.py`.

## Setup

1. Clone the repository
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Create a `.env` file with required API keys:
   ```
   GEMINI_API_KEY=your_api_key_here
   # Optional STT providers:
   # DEEPGRAM_API_KEY=your_key_here
   # VOSK_MODEL_PATH=path/to/model
   ```

## Usage

1. Start the application:
   ```sh
   streamlit run app.py
   ```
2. Configure the interview:
   - Enter candidate name
   - Select role (SDE, Data Science, Product, Design, or Custom)
   - Choose difficulty level (Beginner, Intermediate, Advanced)
   - Click "Generate Questions"

3. During the interview:
   - Use webcam and microphone for face-to-face interaction
   - Click "Transcribe voice to text" to convert speech to text
   - Type or edit responses in the text area
   - Navigate questions using Previous/Next buttons
   - Save responses as you progress

4. Complete the interview:
   - Click "End Interview" for AI evaluation
   - Review scores and feedback
   - Download the interview report as PDF

## Features in Detail

### AI Question Generation
- Role-specific questions
- Three difficulty levels
- Mix of technical and behavioral questions
- Fallback questions if API key not configured

### Real-time Media
- WebRTC-based video streaming
- Audio capture and processing
- Multiple speech-to-text provider options

### Evaluation
- Technical knowledge assessment
- Behavioral fit scoring
- Communication skills evaluation
- Detailed justifications
- Hiring recommendations

## Environment Variables

- `GEMINI_API_KEY`: Required for AI features
- `DEEPGRAM_API_KEY`: Optional for speech-to-text
- `VOSK_MODEL_PATH`: Optional for offline speech recognition

## License

MIT License

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request