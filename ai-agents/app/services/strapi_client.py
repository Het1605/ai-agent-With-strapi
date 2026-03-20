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

    def wait_for_registry(self, expected_uid: str, timeout_s: int = 30) -> bool:
        """
        Polls the field-registry until the expected UID is present and loaded.
        Ensures the hot-reload is complete and the registry is consistent.
        """
        print(f"[StrapiClient] Verifying registry readiness for: {expected_uid}...")
        start_time = time.time()
        registry_url = f"{self.base_url}/api/ai-schema/field-registry"
        
        while time.time() - start_time < timeout_s:
            try:
                # We use a short timeout for the registry check itself
                res = requests.get(registry_url, timeout=5)
                if res.status_code == 200:
                    data = res.json()
                    collections = data.get("collection_uids", [])
                    if expected_uid in collections:
                        print(f"[StrapiClient] Registry Ready! {expected_uid} found.")
                        # Small extra buffer to allow internal Strapi services to catch up
                        time.sleep(2)
                        return True
                
                print(f"[StrapiClient] Registry not ready yet. Retrying in 2s...")
                time.sleep(2)
            except Exception as e:
                print(f"[StrapiClient] Waiting for server to reboot... ({str(e)})")
                time.sleep(2)
        
        print(f"[StrapiClient] WARNING: Registry verification timed out after {timeout_s}s for {expected_uid}")
        return False

    def post_payload(self, payload: Dict[str, Any], max_retries: int = 3) -> Dict[str, Any]:
        """
        Sends a payload to the appropriate Strapi bridge endpoint.
        """
        operation = payload.get("operation")
        # Extract expected UID if this is a creation
        expected_uid = None
        if operation not in {"add_column", "update_collection", "update_column", "delete_column"}:
            s_name = (payload.get("singularName") or payload.get("collectionName") or "").lower()
            if s_name:
                expected_uid = f"api::{s_name}.{s_name}"

        endpoint = self.endpoints["modify"] if operation in {"add_column", "update_collection", "update_column", "delete_column"} else self.endpoints["create"]
        
        last_error = ""
        for attempt in range(1, max_retries + 1):
            try:
                print(f"[StrapiClient] Attempt {attempt}/{max_retries}: {operation} -> {endpoint}")
                response = requests.post(endpoint, json=payload, timeout=30)
                
                # Success
                if response.status_code == 200:
                    data = response.json()
                    print(f"[StrapiClient] Success: {data.get('message', 'OK')}")
                    
                    # If we just created a collection, WAIT for it to appear in the registry
                    if expected_uid:
                        # This replaces the blind 10s sleep with a smart poll
                        self.wait_for_registry(expected_uid)
                    else:
                        # For modifications, we still need a small buffer for hot-reload
                        print(f"[StrapiClient] Waiting 5s for modification sync...")
                        time.sleep(5)
                        
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
                    # If we hit a 500 suggesting "not found", it's likely a stale registry. 
                    # We wait longer on retry to allow reboot.
                    print(f"[StrapiClient] Server Error (Stale Registry?). Retrying in 5s...")
                    time.sleep(5)
                    continue

            except Exception as e:
                last_error = str(e)
                print(f"[StrapiClient] Network Error: {last_error}")
                time.sleep(3)
        
        return {"status": "error", "error": f"Failed after {max_retries} attempts. Last error: {last_error}"}

# Global instance for shared use
strapi_client = StrapiClient()
