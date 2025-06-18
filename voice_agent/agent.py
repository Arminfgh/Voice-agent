"""
voice_agent.py

Author: Your Name
Date: 2025-06-16
Description: Main Voice Agent component with LiveKit integration for TTS system.
Handles voice conversations, reminders, and real-time audio processing.
"""
# from livekit import Room
#await websocket.send("from_python:Hallo! Ich bin dein Bot.")
import asyncio
import logging
import os
import tempfile
import time
from typing import Optional, Dict, Any, Callable
from dataclasses import dataclass
from enum import Enum

import speech_recognition as sr
from TTS.api import TTS
import soundfile as sf
import numpy as np
#from livekit import api, rtc
#from livekit.rtc import Room, TrackKind, AudioFrame, TrackPublishOptions
import webrtcvad

from dataset_curation_coquiai import DatasetCurationCoquiAI
from finetuning_coquiai import FinetuningCoquiAi


class CallType(Enum):
    NORMAL = "normal"
    REMINDER = "reminder"
    EMERGENCY = "emergency"


@dataclass
class ConversationContext:
    user_id: str
    session_id: str
    call_type: CallType
    start_time: float
    last_activity: float
    context_data: Dict[str, Any]
    conversation_history: list


class VoiceAgent:
    """
    Main Voice Agent class that handles voice conversations using TTS/STT
    with LiveKit for real-time communication.
    """
    
    def __init__(self, 
                 model_path: Optional[str] = None,
                 livekit_url: str = "ws://localhost:7880",
                 livekit_api_key: str = "devkey",
                 livekit_secret: str = "secret"):
        
        self.logger = self._setup_logging()
        
        # TTS Setup using existing fine-tuned model
        self.tts_model = self._initialize_tts(model_path)
        
        # STT Setup
        self.stt_recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        
        # LiveKit Configuration
        self.livekit_url = livekit_url
        self.livekit_api_key = livekit_api_key
        self.livekit_secret = livekit_secret
        
        # Voice Activity Detection
        self.vad = webrtcvad.Vad(2)  # Aggressiveness level 0-3
        
        # Active conversations
        self.active_conversations: Dict[str, ConversationContext] = {}
        
        # Callbacks for different events
        self.on_conversation_start: Optional[Callable] = None
        self.on_conversation_end: Optional[Callable] = None
        self.on_user_speech: Optional[Callable] = None
        
        self.logger.info("Voice Agent initialized successfully")

    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)

    def _initialize_tts(self, model_path: Optional[str] = None) -> TTS:
        """Initialize TTS model using the fine-tuned Thorsten model"""
        try:
            if model_path and os.path.exists(model_path):
                # Use custom fine-tuned model
                tts = TTS(model_path=model_path)
                self.logger.info(f"Loaded custom TTS model from {model_path}")
            else:
                # Use pre-trained Thorsten model
                tts = TTS("tts_models/de/thorsten/tacotron2-DDC")
                self.logger.info("Loaded pre-trained Thorsten TTS model")
            
            return tts
        except Exception as e:
            self.logger.error(f"Failed to initialize TTS: {e}")
            raise

    async def create_room_session(self, room_name: str, participant_name: str):

        """Create and join a LiveKit room session"""
        try:
            # Generate access token
            token = api.AccessToken(self.livekit_api_key, self.livekit_secret) \
                .with_identity(participant_name) \
                .with_name(participant_name) \
                .with_grants(api.VideoGrants(
                    room_join=True,
                    room=room_name,
                    can_publish=True,
                    can_subscribe=True
                )).to_jwt()

            # Create room connection
            room = rtc.Room()
            
            # Setup event handlers
            room.on("participant_connected", self._on_participant_connected)
            room.on("participant_disconnected", self._on_participant_disconnected)
            room.on("track_subscribed", self._on_track_subscribed)
            
            # Connect to room
            await room.connect(self.livekit_url, token)
            self.logger.info(f"Connected to room: {room_name}")
            
            return room
            
        except Exception as e:
            self.logger.error(f"Failed to create room session: {e}")
            raise

    def _on_participant_connected(self, participant):
        """Handle participant connection"""
        self.logger.info(f"Participant connected: {participant.identity}")

    def _on_participant_disconnected(self, participant):
        """Handle participant disconnection"""
        self.logger.info(f"Participant disconnected: {participant.identity}")
        # Clean up conversation context
        if participant.identity in self.active_conversations:
            del self.active_conversations[participant.identity]

    async def _on_track_subscribed(self, track, publication, participant):
        """Handle incoming audio track"""
        if track.kind == TrackKind.KIND_AUDIO:
            self.logger.info(f"Subscribed to audio track from {participant.identity}")
            # Start processing audio from this track
            asyncio.create_task(self._process_audio_track(track, participant))

    async def _process_audio_track(self, track, participant):
        """Process incoming audio track for speech recognition"""
        audio_buffer = []
        
        async for frame in track:
            if isinstance(frame, AudioFrame):
                # Convert frame to numpy array
                audio_data = np.frombuffer(frame.data, dtype=np.int16)
                audio_buffer.extend(audio_data)
                
                # Process buffer when it reaches certain size
                if len(audio_buffer) >= 16000:  # ~1 second at 16kHz
                    await self._process_audio_chunk(
                        np.array(audio_buffer), 
                        participant.identity
                    )
                    audio_buffer = []

    async def _process_audio_chunk(self, audio_data: np.ndarray, user_id: str):
        """Process audio chunk for speech recognition"""
        try:
            # Check for voice activity
            if self._has_voice_activity(audio_data):
                # Convert to speech recognition format
                text = await self._speech_to_text(audio_data)
                
                if text and text.strip():
                    self.logger.info(f"Recognized speech from {user_id}: {text}")
                    
                    # Process the recognized text
                    await self._handle_user_input(user_id, text)
                    
        except Exception as e:
            self.logger.error(f"Error processing audio chunk: {e}")

    def _has_voice_activity(self, audio_data: np.ndarray, sample_rate: int = 16000) -> bool:
        """Check if audio data contains voice activity"""
        try:
            # Convert to bytes for VAD
            audio_bytes = audio_data.astype(np.int16).tobytes()
            
            # VAD expects 10, 20, or 30ms frames
            frame_duration = 30  # ms
            frame_length = int(sample_rate * frame_duration / 1000)
            
            # Process in frames
            for i in range(0, len(audio_data) - frame_length, frame_length):
                frame = audio_data[i:i + frame_length].astype(np.int16).tobytes()
                if len(frame) == frame_length * 2:  # 2 bytes per sample
                    if self.vad.is_speech(frame, sample_rate):
                        return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Voice activity detection error: {e}")
            return True  # Default to processing if VAD fails

    async def _speech_to_text(self, audio_data: np.ndarray) -> Optional[str]:
        """Convert audio data to text using speech recognition"""
        try:
            # Save audio to temporary file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                sf.write(temp_file.name, audio_data, 16000)
                
                # Use speech recognition
                with sr.AudioFile(temp_file.name) as source:
                    audio = self.stt_recognizer.record(source)
                    
                # Try German recognition first, then English
                try:
                    text = self.stt_recognizer.recognize_google(audio, language="de-DE")
                except sr.UnknownValueError:
                    try:
                        text = self.stt_recognizer.recognize_google(audio, language="en-US")
                    except sr.UnknownValueError:
                        text = None
                
                # Clean up temporary file
                os.unlink(temp_file.name)
                
                return text
                
        except Exception as e:
            self.logger.error(f"Speech to text error: {e}")
            return None

    async def _handle_user_input(self, user_id: str, text: str):
        """Handle recognized user speech input"""
        try:
            # Get or create conversation context
            if user_id not in self.active_conversations:
                await self._start_conversation(user_id, CallType.NORMAL)
            
            context = self.active_conversations[user_id]
            context.last_activity = time.time()
            context.conversation_history.append({"role": "user", "text": text})
            
            # Generate response based on call type and context
            response = await self._generate_response(context, text)
            
            if response:
                # Convert response to speech and send
                await self._text_to_speech_and_send(user_id, response)
                
                # Update conversation history
                context.conversation_history.append({"role": "agent", "text": response})
                
        except Exception as e:
            self.logger.error(f"Error handling user input: {e}")

    async def _start_conversation(self, user_id: str, call_type: CallType):
        """Start a new conversation session"""
        session_id = f"{user_id}_{int(time.time())}"
        
        context = ConversationContext(
            user_id=user_id,
            session_id=session_id,
            call_type=call_type,
            start_time=time.time(),
            last_activity=time.time(),
            context_data={},
            conversation_history=[]
        )
        
        self.active_conversations[user_id] = context
        
        # Send initial greeting
        greeting = await self._get_initial_greeting(call_type)
        await self._text_to_speech_and_send(user_id, greeting)
        
        if self.on_conversation_start:
            self.on_conversation_start(context)
        
        self.logger.info(f"Started {call_type.value} conversation with {user_id}")

    async def _get_initial_greeting(self, call_type: CallType) -> str:
        """Get initial greeting based on call type"""
        greetings = {
            CallType.NORMAL: "Hallo! Wie kann ich Ihnen heute helfen?",
            CallType.REMINDER: "Hallo! Ich rufe Sie wegen Ihrer Erinnerung an.",
            CallType.EMERGENCY: "Hallo! Dies ist ein wichtiger Anruf. Bitte hören Sie zu."
        }
        return greetings.get(call_type, greetings[CallType.NORMAL])

    async def _generate_response(self, context: ConversationContext, user_input: str) -> str:
        """Generate appropriate response based on context and input"""
        try:
            # Simple rule-based responses (can be enhanced with AI/ML)
            user_input_lower = user_input.lower()
            
            # Handle common interactions
            if any(word in user_input_lower for word in ["hallo", "hi", "guten tag"]):
                return "Hallo! Schön, Sie zu hören. Wie geht es Ihnen?"
            
            elif any(word in user_input_lower for word in ["hilfe", "help", "unterstützung"]):
                return "Gerne helfe ich Ihnen! Was möchten Sie wissen?"
            
            elif any(word in user_input_lower for word in ["zeit", "uhrzeit", "wie spät"]):
                current_time = time.strftime("%H:%M", time.localtime())
                return f"Es ist jetzt {current_time} Uhr."
            
            elif any(word in user_input_lower for word in ["tschüss", "auf wiedersehen", "beenden"]):
                await self._end_conversation(context.user_id)
                return "Auf Wiedersehen! Schönen Tag noch!"
            
            elif context.call_type == CallType.REMINDER:
                return await self._handle_reminder_response(context, user_input)
            
            else:
                # Default response
                return "Das ist interessant. Können Sie mir mehr dazu erzählen?"
                
        except Exception as e:
            self.logger.error(f"Error generating response: {e}")
            return "Entschuldigung, ich habe das nicht verstanden. Können Sie das wiederholen?"

    async def _handle_reminder_response(self, context: ConversationContext, user_input: str) -> str:
        """Handle responses in reminder calls"""
        user_input_lower = user_input.lower()
        
        if any(word in user_input_lower for word in ["ja", "okay", "verstanden", "danke"]):
            return "Perfekt! Die Erinnerung wurde bestätigt. Haben Sie noch Fragen?"
        elif any(word in user_input_lower for word in ["nein", "später", "nicht jetzt"]):
            return "Verstanden. Soll ich Sie später noch einmal erinnern?"
        else:
            return "Haben Sie die Erinnerung verstanden? Sagen Sie ja oder nein."

    async def _text_to_speech_and_send(self, user_id: str, text: str):
        """Convert text to speech and send via LiveKit"""
        try:
            # Generate speech using TTS
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                self.tts_model.tts_to_file(
                    text=text,
                    file_path=temp_file.name,
                    speaker_wav=None  # Use default voice
                )
                
                # Load generated audio
                audio_data, sample_rate = sf.read(temp_file.name)
                
                # Convert to the format expected by LiveKit
                if audio_data.dtype != np.int16:
                    audio_data = (audio_data * 32767).astype(np.int16)
                
                # Send audio via LiveKit (this would need the room reference)
                await self._send_audio_to_user(user_id, audio_data, sample_rate)
                
                # Clean up
                os.unlink(temp_file.name)
                
                self.logger.info(f"Sent speech response to {user_id}: {text[:50]}...")
                
        except Exception as e:
            self.logger.error(f"Error in text to speech: {e}")

    async def _send_audio_to_user(self, user_id: str, audio_data: np.ndarray, sample_rate: int):
        """Send audio data to user via LiveKit"""
        # This would require access to the room and audio track
        # Implementation depends on your LiveKit setup
        #await websocket.send("from_python:Hallo! Ich bin dein Bot.")

        self.logger.info(f"Sending audio to {user_id} (length: {len(audio_data)} samples)")

    async def _end_conversation(self, user_id: str):
        """End conversation session"""
        if user_id in self.active_conversations:
            context = self.active_conversations[user_id]
            
            if self.on_conversation_end:
                self.on_conversation_end(context)
            
            del self.active_conversations[user_id]
            self.logger.info(f"Ended conversation with {user_id}")

    # Public methods for call management
    
    async def start_normal_call(self, user_id: str, room_name: str):
        """Start a normal voice call"""
        room = await self.create_room_session(room_name, f"agent_{user_id}")
        await self._start_conversation(user_id, CallType.NORMAL)
        return room

    async def start_reminder_call(self, user_id: str, reminder_data: Dict[str, Any]):
        """Start a reminder call"""
        room_name = f"reminder_{user_id}_{int(time.time())}"
        room = await self.create_room_session(room_name, f"reminder_agent_{user_id}")
        
        await self._start_conversation(user_id, CallType.REMINDER)
        
        # Store reminder data in context
        if user_id in self.active_conversations:
            self.active_conversations[user_id].context_data["reminder"] = reminder_data
        
        return room

    def get_active_conversations(self) -> Dict[str, ConversationContext]:
        """Get all active conversations"""
        return self.active_conversations.copy()

    async def cleanup_inactive_conversations(self, timeout_seconds: int = 300):
        """Clean up conversations that have been inactive for too long"""
        current_time = time.time()
        inactive_users = []
        
        for user_id, context in self.active_conversations.items():
            if current_time - context.last_activity > timeout_seconds:
                inactive_users.append(user_id)
        
        for user_id in inactive_users:
            await self._end_conversation(user_id)
            self.logger.info(f"Cleaned up inactive conversation: {user_id}")


# Example usage and testing
async def main():
    """Example usage of the Voice Agent"""
    agent = VoiceAgent()
    
    # Example: Start a normal call
    try:
        room = await agent.start_normal_call("user123", "test_room")
        print("Voice agent started successfully!")
        
        # Keep the agent running
        await asyncio.sleep(60)  # Run for 1 minute
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())