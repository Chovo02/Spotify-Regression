import base64
import requests
import json
import dotenv
import os

dotenv.load_dotenv()

client_id = os.getenv("SPOTIFY_CLIENT_ID")
client_secret = os.getenv("SPOTIFY_CLIENT_SECRET")

def get_token():
    auth_string = f"{client_id}:{client_secret}"
    auth_bytes = auth_string.encode("utf-8")
    auth_base64 = str(base64.b64encode(auth_bytes), "utf-8")

    url = "https://accounts.spotify.com/api/token"
    headers = {
        "Authorization": f"Basic {auth_base64}",
        "Content-Type": "application/x-www-form-urlencoded"
    }
    data = {
        "grant_type": "client_credentials"
        }
    result = requests.post(url, headers=headers, data=data)
    json_result = json.loads(result.text)
    return json_result["access_token"]

def get_auth_header(token):
    return {"Authorization": f"Bearer {token}"}

def get_feat(track_name, token):
    track_name = track_name.replace(" ", "%20")
    url = f"https://api.spotify.com/v1/search?q={track_name}&type=track&limit=1"
    result = requests.get(url, headers=get_auth_header(token))
    json_result = json.loads(result.content)
    feat = []
    for artist in json_result["tracks"]["items"][0]["artists"][1:]:
        feat.append(artist["name"])
    return feat