// index.js - Final version (ESM, LiveKit, WebSocket)
import express from "express";
import * as livekit from "livekit-client";
import { WebSocketServer } from "ws";

// === CONFIG ===
const LIVEKIT_URL = "wss://armin-s9egncm3.livekit.cloud";
const TOKEN = "PASTE_YOUR_TOKEN_HERE"; // ← insert your generated token
const PORT = 5000;

// === WebSocket + Express server ===
const app = express();
const server = app.listen(PORT, () => {
  console.log(`🛰 WebSocket + LiveKit bridge running at ws://localhost:${PORT}`);
});

const wss = new WebSocketServer({ server });
let botSocket = null;

wss.on("connection", (ws) => {
  console.log("✅ Python bot connected to WebSocket");

  botSocket = ws;

  ws.on("message", (msg) => {
    console.log("📩 Message from Python bot:", msg.toString());
  });
});

// === Connect to LiveKit ===
async function start() {
  const room = await livekit.connect(LIVEKIT_URL, TOKEN, {
    autoSubscribe: true,
  });

  console.log("🎧 Connected to LiveKit room:", room.name);

  room.on("trackSubscribed", (track, publication, participant) => {
    if (track.kind === "audio") {
      console.log("🎤 Subscribed to audio from:", participant.identity);

      if (botSocket) {
        botSocket.send(`🎤 Receiving audio from ${participant.identity}`);
      }
    }
  });
}

start().catch(console.error);
