from fastmcp import FastMCP
import requests
import json

# 1. Initialize the MCP Server
mcp = FastMCP("Climate Risk and Ag-Insurance Server")

# 2. Add Tools using the @mcp.tool decorator
@mcp.tool
def get_current_weather(latitude: float, longitude: float) -> str:
    """Get the current real-time weather (temperature and precipitation) for a specific latitude and longitude."""
    url = f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current=temperature_2m,precipitation&timezone=auto"
    try:
        response = requests.get(url)
        data = response.json()
        current = data.get("current", {})
        return json.dumps({
            "temperature_celsius": current.get("temperature_2m"),
            "precipitation_mm": current.get("precipitation")
        })
    except Exception as e:
        return json.dumps({"error": str(e)})

@mcp.tool
def calculate_claim(threshold_yield: float, actual_yield: float, sum_insured: float) -> str:
    """Calculates the exact crop insurance claim payout based on Threshold Yield, Actual Yield, and Sum Insured."""
    if actual_yield >= threshold_yield:
        return json.dumps({
            "claim_payout": 0, 
            "message": "Actual yield is greater than or equal to threshold yield. No payout required."
        })
    
    shortfall_ratio = (threshold_yield - actual_yield) / threshold_yield
    payout = shortfall_ratio * sum_insured
    
    return json.dumps({
        "claim_payout": round(payout, 2),
        "shortfall_percentage": round(shortfall_ratio * 100, 2)
    })

# 3. Run the server using stdio transport by default
if __name__ == "__main__":
    mcp.run()