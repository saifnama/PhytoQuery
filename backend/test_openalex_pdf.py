import asyncio, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))
from services.openalex.client import OpenAlexClient
import json

async def test():
    result = await OpenAlexClient.fetch_paper("10.1104/pp.109.138990")
    print("pdfUrl in result:", result.get("pdfUrl"))
    print("has pdfUrl:", bool(result.get("pdfUrl")))

asyncio.run(test())
