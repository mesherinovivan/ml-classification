import base64
import io
from PIL import Image
import json
import requests

import base64
with open("image.jpg", "rb") as image_file:
    base64str = base64.b64encode(image_file.read()).decode("utf-8")


payload = json.dumps({
  "base64str": base64str,
  "threshold": 0.6
})

response = requests.put("http://127.0.0.1:8000/predict",data = payload)
data_dict = response.json()
print(data_dict)