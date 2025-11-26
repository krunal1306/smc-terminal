import MetaTrader5 as mt5
import websockets
import asyncio
import json

# List your symbols here (match those in the dashboard)
SYMBOLS = [
    'BTCUSDm', 'ETHUSDm', 'SOLUSDm', 'XAUUSDm',
    'USTECm', 'US30m', 'DXYm',
    'EURUSDm', 'GBPUSDm', 'AUDUSDm',
    'DE30m', 'JP225m', 'UK100m'
]

async def handler(websocket):
    while True:
        if not mt5.initialize():
            print("MT5 initialization failed")
            await asyncio.sleep(5)
            continue
        
        for symbol in SYMBOLS:
            tick = mt5.symbol_info_tick(symbol)
            if tick:
                data = {
                    'symbol': symbol,
                    'ask': float(tick.ask),
                    'bid': float(tick.bid),
                    'time': tick.time
                }
                try:
                    await websocket.send(json.dumps(data))
                except:
                    pass  # Client disconnected
        
        await asyncio.sleep(1)  # Adjust for update frequency (1s = high frequency)

async def main():
    print("Starting MT5 WebSocket bridge on ws://localhost:8765")
    async with websockets.serve(handler, "localhost", 8765):
        await asyncio.Future()  # Run forever

if __name__ == "__main__":
    asyncio.run(main())