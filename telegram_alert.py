import requests

TOKEN = ""
CHAT_ID = ""

def send_alert(message):
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"

    data = {
        "chat_id": CHAT_ID,
        "text": message
    }

    response = requests.post(url, data=data, timeout=10)
    response.raise_for_status()

    return response.json()


def send_photo(image_path, caption=""):
    url = f"https://api.telegram.org/bot{TOKEN}/sendPhoto"

    with open(image_path, "rb") as image:
        response = requests.post(
            url,
            data={
                "chat_id": CHAT_ID,
                "caption": caption
            },
            files={"photo": image},
            timeout=20
        )

    response.raise_for_status()
    return response.json()
