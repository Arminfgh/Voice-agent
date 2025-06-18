"""
livekit_handler.py

Author: Your Name
Date: 2025-06-16
Description: LiveKit server management and real-time audio streaming handler.
Manages rooms, participants, and audio streams for the voice agent system.
"""

import asyncio
import logging
import time
from typing import Dict, Optional, Callable, List
from dataclasses import dataclass

#from livekit import api, rtc
#from livekit.rtc import Room, TrackKind, AudioFrame, VideoFrame, TrackPublishOptions
import numpy as np


@dataclass
class RoomSession:
    room: Room
    room_name: str
    participants: Dict[str, rtc.Participant]
    created_at: float
    audio_tracks: Dict[str, rtc.Track]
    active: bool = True


class LiveKitHandler:
    """
    Handles LiveKit server operations, room management, and real-time audio streaming.
    """
    
    def __init__(self, 
                 server_url: str = "ws://localhost:7880",
                 api_key: str = "devkey",
                 api_secret: str = "secret"):
        
        self.server_url = server_url
        self.api_key = api_key
        self.api_secret = api_secret
        
        self.logger = self._setup_logging()
        
        # Active room sessions
        self.active_rooms: Dict[str, RoomSession] = {}
        
        # Callbacks
        self.on_participant_joined: Optional[Callable] = None
        self.on_participant_left: Optional[Callable] = None
        self.on_audio_received: Optional[Callable] = None
        self.on_room_disconnected: Optional[Callable] = None
        
        self.logger.info("LiveKit Handler initialized")

    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(f"{__name__}.LiveKitHandler")

    def generate_access_token(self, 
                            room_name: str, 
                            participant_identity: str,
                            participant_name: Optional[str] = None,
                            can_publish: bool = True,
                            can_subscribe: bool = True) -> str:
        """Generate access token for room participation"""
        try:
            #token = api.AccessToken(self.api_key, self.api_secret)
            token = token.with_identity(participant_identity)
            
            if participant_name:
                token = token.with_name(participant_name)
            
            grants = api.VideoGrants(
                room_join=True,
                room=room_name,
                can_publish=can_publish,
                can_subscribe=can_subscribe
            )
            
            token = token.with_grants(grants)
            return token.to_jwt()
            
        except Exception as e:
            self.logger.error(f"Failed to generate access token: {e}")
            raise

    async def create_room(self, 
                         room_name: str, 
                         max_participants: int = 10) -> bool:
        """Create a new room on the LiveKit server"""
        try:
            room_service = api.RoomService(self.server_url, self.api_key, self.api_secret)
            
            room_info = await room_service.create_room(
                api.CreateRoomRequest(
                    name=room_name,
                    max_participants=max_participants,
                    empty_timeout=30 * 60,  # 30 minutes
                    max_duration=2 * 60 * 60  # 2 hours
                )
            )
            
            self.logger.info(f"Created room: {room_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create room {room_name}: {e}")
            return False

    async def join_room(self, 
                       room_name: str, 
                       participant_identity: str,
                       participant_name: Optional[str] = None) -> Optional[RoomSession]:
        """Join a room as a participant"""
        try:
            # Generate access token
            token = self.generate_access_token(
                room_name, 
                participant_identity, 
                participant_name
            )
            
            # Create room connection
            room = rtc.Room()
            
            # Setup event handlers
            room.on("participant_connected", self._on_participant_connected)
            room.on("participant_disconnected", self._on_participant_disconnected)
            room.on("track_subscribed", self._on_track_subscribed)
            room.on("track_unsubscribed", self._on_track_unsubscribed)
            room.on("disconnected", self._on_room_disconnected)
            room.on("connection_quality_changed", self._on_connection_quality_changed)
            
            # Connect to room
            await room.connect(self.server_url, token)
            
            # Create room session
            session = RoomSession(
                room=room,
                room_name=room_name,
                participants={},
                created_at=time.time(),
                audio_tracks={}
            )
            
            self.active_rooms[room_name] = session
            
            self.logger.info(f"Joined room: {room_name} as {participant_identity}")
            return session
            
        except Exception as e:
            self.logger.error(f"Failed to join room {room_name}: {e}")
            return None

    def _on_participant_connected(self, participant: rtc.Participant):
        """Handle participant connection"""
        self.logger.info(f"Participant connected: {participant.identity}")
        
        # Find the room this participant belongs to
        for session in self.active_rooms.values():
            if participant in session.room.participants.values():
                session.participants[participant.identity] = participant
                break
        
        # Call external callback if set
        if self.on_participant_joined:
            self.on_participant_joined(participant)

    def _on_participant_disconnected(self, participant: rtc.Participant):
        """Handle participant disconnection"""
        self.logger.info(f"Participant disconnected: {participant.identity}")
        
        # Remove from session participants
        for session in self.active_rooms.values():
            if participant.identity in session.participants:
                del session.participants[participant.identity]
                break
        
        # Call external callback if set
        if self.on_participant_left:
            self.on_participant_left(participant)

    async def _on_track_subscribed(self, 
                                  track: rtc.Track, 
                                  publication: rtc.TrackPublication, 
                                  participant: rtc.Participant):
        """Handle track subscription"""
        self.logger.info(f"Subscribed to {track.kind} track from {participant.identity}")
        
        # Store audio tracks for processing
        if track.kind == TrackKind.KIND_AUDIO:
            for session in self.active_rooms.values():
                if participant.identity in session.participants:
                    session.audio_tracks[participant.identity] = track
                    # Start processing audio from this track
                    asyncio.create_task(self._process_audio_track(track, participant))
                    break

    def _on_track_unsubscribed(self, 
                              track: rtc.Track, 
                              publication: rtc.TrackPublication, 
                              participant: rtc.Participant):
        """Handle track unsubscription"""
        self.logger.info(f"Unsubscribed from {track.kind} track from {participant.identity}")
        
        # Remove from stored tracks
        for session in self.active_rooms.values():
            if participant.identity in session.audio_tracks:
                del session.audio_tracks[participant.identity]
                break

    def _on_room_disconnected(self, reason: str):
        """Handle room disconnection"""
        self.logger.warning(f"Room disconnected: {reason}")
        
        # Mark all sessions as inactive
        for session in self.active_rooms.values():
            session.active = False
        
        if self.on_room_disconnected:
            self.on_room_disconnected(reason)

    def _on_connection_quality_changed(self, quality, participant):
        """Handle connection quality changes"""
        self.logger.debug(f"Connection quality for {participant.identity}: {quality}")

    async def _process_audio_track(self, track: rtc.Track, participant: rtc.Participant):
        """Process incoming audio track"""
        try:
            async for frame in track:
                if isinstance(frame, AudioFrame):
                    # Convert audio frame to numpy array
                    audio_data = np.frombuffer(frame.data, dtype=np.int16)
                    
                    # Call external callback for audio processing
                    if self.on_audio_received:
                        await self.on_audio_received(audio_data, participant)
                        
        except Exception as e:
            self.logger.error(f"Error processing audio track: {e}")

    async def publish_audio(self, 
                           room_name: str, 
                           audio_data: np.ndarray, 
                           sample_rate: int = 16000) -> bool:
        """Publish audio data to a room"""
        try:
            if room_name not in self.active_rooms:
                self.logger.error(f"Room {room_name} not found")
                return False
            
            session = self.active_rooms[room_name]
            room = session.room
            
            # Create audio source
            audio_source = rtc.AudioSource(sample_rate, 1)  # mono
            
            # Create track
            track = rtc.LocalAudioTrack.create_audio_track("agent_audio", audio_source)
            
            # Publish track
            options = TrackPublishOptions()
            options.source = rtc.TrackSource.SOURCE_MICROPHONE
            
            publication = await room.local_participant.publish_track(track, options)
            
            # Convert numpy array to AudioFrame and capture
            frame = AudioFrame(
                data=audio_data.astype(np.int16).tobytes(),
                sample_rate=sample_rate,
                num_channels=1,
                samples_per_channel=len(audio_data)
            )
            
            await audio_source.capture_frame(frame)
            
            self.logger.info(f"Published audio to room {room_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to publish audio: {e}")
            return False

    async def send_audio_stream(self, 
                               room_name: str, 
                               audio_generator, 
                               sample_rate: int = 16000):
        """Send continuous audio stream to room"""
        try:
            if room_name not in self.active_rooms:
                self.logger.error(f"Room {room_name} not found")
                return
            
            session = self.active_rooms[room_name]
            room = session.room
            
            # Create audio source
            audio_source = rtc.AudioSource(sample_rate, 1)
            track = rtc.LocalAudioTrack.create_audio_track("agent_stream", audio_source)
            
            # Publish track
            options = TrackPublishOptions()
            options.source = rtc.TrackSource.SOURCE_MICROPHONE
            await room.local_participant.publish_track(track, options)
            
            # Stream audio data
            async for audio_chunk in audio_generator:
                if not session.active:
                    break
                
                frame = AudioFrame(
                    data=audio_chunk.astype(np.int16).tobytes(),
                    sample_rate=sample_rate,
                    num_channels=1,
                    samples_per_channel=len(audio_chunk)
                )
                await audio_source.capture_frame(frame)

            self.logger.info(f"Finished streaming audio to {room_name}")
        except Exception as e:
            self.logger.error(f"Failed to stream audio to room {room_name}: {e}")