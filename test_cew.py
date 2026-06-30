import requests

token = "8819583453:AAFpX10MgHhRH0VWi9a0eIySIZ_rrlWSP3k"
chat_id = input("Chat ID خودت رو وارد کن (از @userinfobot): ")

url = f"https://api.telegram.org/bot{token}/sendMessage"
payload = {
    "chat_id": chat_id,
    "text": "🚗 Test message from Fatigue Detection System!"
}

response = requests.post(url, json=payload)
print(f"Status: {response.status_code}")
print(f"Response: {response.text}")