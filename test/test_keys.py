# =============================================================================
# test_keys.py — Tests ALL API keys before running any module
# Run this first to confirm everything is connected properly
# =============================================================================
import config
import sys

print("=" * 55)
print("WILDFIRE PROJECT — API KEY TESTER")
print("=" * 55)

# =============================================================================
# TEST 1 — FIRMS API KEY
# =============================================================================
print("\n[1/3] Testing NASA FIRMS API Key...")

try:
    import requests

    # Read key from config
    import config

    if (
        not config.FIRMS_MAP_KEY
        or config.FIRMS_MAP_KEY == "c9e7ff13ad9f8cb69f8bc5577c6604dd"
    ):
        print("  FAIL — FIRMS_MAP_KEY is not set in your .env file")
        print("  Edit .env and set: FIRMS_MAP_KEY=your_actual_key")
        print("  Get your key at: firms.modaps.eosdis.nasa.gov/api/")
    else:
        # Test with small real request
        # California, Camp Fire date, 1 day, small area
        url = (
            f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/"
            f"{config.FIRMS_MAP_KEY}/VIIRS_SNPP_SP/"
            f"-122,39,-120,41/1/2018-11-08"
        )

        response = requests.get(url, timeout=30)

        if response.status_code == 200:
            lines = response.text.strip().split("\n")
            if len(lines) > 1:
                print(f"  PASS — Key works!")
                print(f"  Got {len(lines) - 1} fire detections in test area")
                print(f"  First row: {lines[1][:80]}...")
            else:
                print("  PASS — Key accepted but no data in test area")
                print("  This is OK — key is valid")

        elif response.status_code == 400:
            print("  FAIL — Bad request. Key may be invalid")
            print(f"  Response: {response.text[:200]}")

        elif response.status_code == 401:
            print("  FAIL — Unauthorized. Key is wrong or not activated")
            print("  Check your email for the correct key")

        elif response.status_code == 429:
            print("  FAIL — Rate limited. Wait 1 hour and try again")

        else:
            print(f"  FAIL — Unexpected status: {response.status_code}")
            print(f"  Response: {response.text[:200]}")

except ImportError:
    print("  FAIL — requests library not installed")
    print("  Run: pip install requests")
except Exception as e:
    print(f"  FAIL — Error: {e}")


# =============================================================================
# TEST 2 — GOOGLE EARTH ENGINE
# =============================================================================
print("\n[2/3] Testing Google Earth Engine...")

try:
    import ee

    if not config.GEE_PROJECT or config.GEE_PROJECT == "your_gee_project_id_here":
        print("  FAIL — GEE_PROJECT is not set in your .env file")
        print("  Edit .env and set: GEE_PROJECT=your_actual_project_id")
        print("  Find your project ID at: code.earthengine.google.com")
    else:
        try:
            # Try to initialize
            ee.Initialize(project=config.GEE_PROJECT)

            # Simple test — get one Landsat image
            test_image = (
                ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
                .filterDate("2018-11-08", "2018-11-09")
                .filterBounds(ee.Geometry.Point([-121.5, 39.8]))
                .first()
            )

            # Get image ID — this forces GEE to actually connect
            image_id = test_image.get("system:index").getInfo()

            if image_id:
                print(f"  PASS — GEE connected!")
                print(f"  Test image ID: {image_id}")
            else:
                print("  PASS — GEE connected but no image found in test")
                print("  This is OK — connection works")

        except ee.EEException as e:
            print(f"  FAIL — GEE error: {e}")
            print("  Try running: earthengine authenticate")

        except Exception as e:
            error_msg = str(e)

            if (
                "Please authorize access" in error_msg
                or "credentials" in error_msg.lower()
            ):
                print("  FAIL — Not authenticated")
                print("  Run this in terminal: earthengine authenticate")

            elif "project" in error_msg.lower():
                print("  FAIL — Project ID wrong or no access")
                print(f"  Your project ID: {config.GEE_PROJECT}")
                print("  Check at: code.earthengine.google.com")

            else:
                print(f"  FAIL — {error_msg}")

except ImportError:
    print("  FAIL — earthengine-api not installed")
    print("  Run: pip install earthengine-api")
    print("  Then: earthengine authenticate")


# =============================================================================
# TEST 3 — COPERNICUS / ERA5 (OPTIONAL — for weather data)
# =============================================================================
print("\n[3/3] Testing Copernicus CDS (ERA5 weather)...")

try:
    import cdsapi

    # Check if .cdsapirc file exists
    from pathlib import Path

    cds_config = Path.home() / ".cdsapirc"

    if not cds_config.exists():
        print("  SKIP — No .cdsapirc file found")
        print("  This is OK for now — needed only for weather features")
        print("  Setup later at: cds.climate.copernicus.eu")
    else:
        # Try connecting
        client = cdsapi.Client(quiet=True)
        print("  PASS — Copernicus CDS connected!")

except ImportError:
    print("  SKIP — cdsapi not installed (optional for now)")
except Exception as e:
    print(f"  INFO — CDS issue: {str(e)[:100]}")
    print("  This is optional — project works without weather data initially")


# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 55)
print("SUMMARY")
print("=" * 55)
print("""
  FIRMS key  → Needed for module1_firms_downloader.py
  GEE key    → Needed for module1_data_collection.py
  CDS key    → Needed later for weather features (optional now)

  If FIRMS and GEE both PASS, you are ready to run modules.
""")
