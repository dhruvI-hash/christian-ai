import json
import urllib.request
import urllib.error
import time
import sys

# Ensure UTF-8 output for Windows console
sys.stdout.reconfigure(encoding='utf-8')

def run_test():
    url = "http://127.0.0.1:8000/api/chat"
    
    # Payload for the chat request
    payload = {
        "message": "What is the Test of Abraham?",
        "conversation_id": f"test_rag_{int(time.time())}",
        "denomination": "Non-denominational",
        "use_rag": True,
        "vector_db": "qdrant",  # Set to the DB you ingested into
        "llm_config": {
            "provider": "gemini",
            "model": "gemini-3-flash-preview"
        }
    }
    
    data = json.dumps(payload).encode("utf-8")
    headers = {
        "Content-Type": "application/json"
    }
    
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    
    print("Sending chat request to the AI...")
    print(f"Question: {payload['message']}")
    
    try:
        start_time = time.time()
        with urllib.request.urlopen(req) as response:
            result = json.loads(response.read().decode("utf-8"))
            elapsed = time.time() - start_time
            
            print(f"\nResponse received in {elapsed:.2f} seconds!\n")
            print("-" * 50)
            print("AGENT RESPONSE:")
            print(result.get("response", ""))
            print("-" * 50)
            
            # Check if RAG was used
            rag_used = result.get("rag_used", False)
            print(f"\nRAG Tool successfully called: {rag_used}")
            
            citations = result.get("citations", [])
            if citations:
                print("\nCitations extracted:")
                for citation in citations:
                    print(f"- {citation}")
            else:
                print("\nNo citations extracted.")
                
            if rag_used:
                print("\n✅ SUCCESS: The agent successfully called the RAG tool and used it to generate the answer!")
            else:
                print("\n❌ FAILURE: The RAG tool was not called or its output wasn't recognized.")
                
    except urllib.error.HTTPError as e:
        print(f"HTTP Error: {e.code} - {e.reason}")
        print(e.read().decode("utf-8"))
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    run_test()
