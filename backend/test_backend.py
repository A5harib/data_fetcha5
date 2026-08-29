"""
Client verification script to test the live WebSocket streaming endpoint.
"""
import asyncio
import json
import websockets

async def test_ws_client(symbol: str = "BTCUSDT"):
    uri = f"ws://localhost:8000/ws/analytics/{symbol}"
    print(f"[*] Connecting to {uri}...")
    
    async with websockets.connect(uri) as ws:
        print("[+] Connected! Listening for real-time order flow signals...")
        for i in range(10):
            msg = await ws.recv()
            data = json.loads(msg)
            print(f"\n--- Packet #{i+1} received ---")
            print(json.dumps(data, indent=2))
            
if __name__ == "__main__":
    asyncio.run(test_ws_client())
