import logging
logging.basicConfig(level=logging.INFO)
import asyncio
import logging
from voice_agent import VoiceAgent

logging.basicConfig(level=logging.INFO)

async def main():
    logging.info("🔥 Voice Agent test starting...")

    agent = VoiceAgent()
    await agent.start_normal_call("user_1", "conversation")
    await asyncio.sleep(5)

    logging.info("✅ Voice Agent call started.")

if __name__ == "__main__":
    asyncio.run(main())
