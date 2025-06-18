# voice_agent/ws_agent.py

import asyncio
import websockets

async def main():
    uri = "ws://localhost:5000"

    async with websockets.connect(uri) as websocket:
        print("✅ Connected to WebSocket server!")

        # Listen for messages from the bridge
        while True:
            message = await websocket.recv()
            print("📩 Message from WebSocket:", message)

            # Send a response back
            await websocket.send("🧠 Python bot says hi!")

asyncio.run(main())
