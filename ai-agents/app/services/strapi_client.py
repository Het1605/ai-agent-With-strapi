import requests
import time
import json
from typing import Dict, Any, List, Optional

class StrapiClient:
    """
    Service for robust communication with the Strapi Bridge.
    Handles networking boilerplate, retries, and hot-reload timing.
    """
    
    def __init__(self, base_url: str = "http://strapi:1337", stabilization_delay: int = 10):
        self.base_url = base_url
        self.stabilization_delay = stabilization_delay
        self.endpoints = {
            "create": f"{base_url}/api/ai-schema/create-collection",
            "modify": f"{base_url}/api/ai-schema/modify-schema"
        }

    def post_payload(self, payload: Dict[str, Any], max_retries: int = 3) -> Dict[str, Any]:
        """
        Sends a payload to the appropriate Strapi bridge endpoint.
        """
        operation = payload.get("operation")
        endpoint = self.endpoints["modify"] if operation in {"add_column", "update_collection", "update_field", "delete_field"} else self.endpoints["create"]
        
        last_error = ""
        for attempt in range(1, max_retries + 1):
            try:
                print(f"[StrapiClient] Attempt {attempt}/{max_retries}: {operation} -> {endpoint}")
                response = requests.post(endpoint, json=payload, timeout=30)
                
                # Success
                if response.status_code == 200:
                    data = response.json()
                    print(f"[StrapiClient] Success: {data.get('message', 'OK')}")
                    
                    # Wait for Strapi hot-reload to stabilize
                    print(f"[StrapiClient] Waiting {self.stabilization_delay}s for hot-reload...")
                    time.sleep(self.stabilization_delay)
                    return {"status": "success", "data": data}

                # Duplicate / Client Error
                if response.status_code == 400:
                    body = response.text.lower()
                    if "already exists" in body:
                        print(f"[StrapiClient] Skipped: Collection already exists.")
                        return {"status": "skipped", "reason": "already_exists"}
                    return {"status": "error", "error": f"HTTP 400: {response.text}"}

                # Server Error (Retry)
                if response.status_code >= 500:
                    last_error = f"HTTP {response.status_code}: {response.text}"
                    print(f"[StrapiClient] Server Error. Retrying in 2s...")
                    time.sleep(2)
                    continue

            except Exception as e:
                last_error = str(e)
                print(f"[StrapiClient] Network Error: {last_error}")
                time.sleep(2)
        
        return {"status": "error", "error": f"Failed after {max_retries} attempts. Last error: {last_error}"}

# Global instance for shared use
strapi_client = StrapiClient()
