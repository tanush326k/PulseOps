import json
import urllib.request
import time

url = "http://127.0.0.1:8080/api/run-incident"
data = json.dumps({"incident_id": "INC-001"}).encode("utf-8")
req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})

print("Sending request to PulseOps /api/run-incident...")
t0 = time.time()
with urllib.request.urlopen(req, timeout=120) as resp:
    result = json.loads(resp.read().decode("utf-8"))
latency = time.time() - t0

print(f"Success! Response in {latency:.2f}s:")
print("Action chosen by Decider 2B:", result["decider"]["action"])
print("Confidence:", result["decider"]["confidence"])
print("Probabilities:", result["decider"]["probabilities"])
print("Tool result finding:", result["tool_result"]["key_finding"])
print("Assessment:", result["assessment"]["assessment"][:120], "...")
