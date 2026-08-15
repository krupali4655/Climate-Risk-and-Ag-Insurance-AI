from fastmcp import FastMCP
from urllib.parse import quote
import requests
import json

# 1. Initialize the MCP Server
mcp = FastMCP("Climate Risk and Ag-Insurance Server")


# 2. Add Tools using the @mcp.tool decorator
def _geocode(name: str):
    """Query Open-Meteo's geocoding API and return its raw 'results' list (or [])."""
    geocode_url = (
        "https://geocoding-api.open-meteo.com/v1/search"
        f"?name={quote(name)}&count=5&language=en&format=json"
    )
    geo_response = requests.get(geocode_url, timeout=10)
    geo_response.raise_for_status()
    return geo_response.json().get("results") or []


@mcp.tool
def get_current_weather(location_name: str) -> str:
    """Get the current real-time weather (temperature and precipitation) for a named place — prefer a plain place name (e.g. 'Talala') over a full address; state/country will be resolved automatically."""
 
    try:
        results = _geocode(location_name)

        if not results:
            simplified = location_name.split(",")[0].strip()
            if simplified and simplified.lower() != location_name.lower():
                results = _geocode(simplified)

        if not results:
            return json.dumps({"error": f"Could not find a location matching '{location_name}'."})

        match = next((r for r in results if r.get("country_code") == "IN"), results[0])
        latitude = match["latitude"]
        longitude = match["longitude"]
        resolved_name = ", ".join(
            part for part in [match.get("name"), match.get("admin1"), match.get("country")] if part
        )

        weather_url = (
            f"https://api.open-meteo.com/v1/forecast"
            f"?latitude={latitude}&longitude={longitude}"
            f"&current=temperature_2m,precipitation&timezone=auto"
        )
        weather_response = requests.get(weather_url, timeout=10)
        weather_response.raise_for_status()
        weather_data = weather_response.json()
        current = weather_data.get("current", {})

        return json.dumps({
            "resolved_location": resolved_name,
            "latitude": latitude,
            "longitude": longitude,
            "temperature_celsius": current.get("temperature_2m"),
            "precipitation_mm": current.get("precipitation")
        })
    except Exception as e:
        return json.dumps({"error": str(e)})

        weather_url = (
            f"https://api.open-meteo.com/v1/forecast"
            f"?latitude={latitude}&longitude={longitude}"
            f"&current=temperature_2m,precipitation&timezone=auto"
        )
        weather_response = requests.get(weather_url, timeout=10)
        weather_response.raise_for_status()
        weather_data = weather_response.json()
        current = weather_data.get("current", {})

        return json.dumps({
            "resolved_location": resolved_name,
            "latitude": latitude,
            "longitude": longitude,
            "temperature_celsius": current.get("temperature_2m"),
            "precipitation_mm": current.get("precipitation")
        })
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool
def calculate_claim(threshold_yield: float, actual_yield: float, sum_insured: float) -> str:
    """Calculates the exact crop insurance claim payout based on Threshold Yield, Actual Yield, and Sum Insured."""
  
    if threshold_yield <= 0:
        return json.dumps({"error": "threshold_yield must be greater than 0"})

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


if __name__ == "__main__":
    mcp.run(transport="http", host="127.0.0.1", port=8931)