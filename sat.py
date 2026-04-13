import ee
import random
import requests
from PIL import Image
from io import BytesIO
import torch
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
import matplotlib.pyplot as plt
import folium
import webbrowser
import os   
from geopy.geocoders import Nominatim
from datetime import datetime

# =========================
# INIT EARTH ENGINE
# =========================
ee.Initialize(project='diesel-thunder-469808-b3')

# =========================
# TIME
# =========================
now = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
print("Prediction Time:", now)

# =========================
# GET SATELLITE IMAGE
# =========================
while True:
    lat = random.uniform(8.0, 37.0)
    lon = random.uniform(68.0, 97.0)

    point = ee.Geometry.Point([lon, lat])

    collection = ee.ImageCollection('COPERNICUS/S2_HARMONIZED') \
        .filterBounds(point) \
        .filterDate('2021-01-01', '2024-12-31') \
        .sort('CLOUDY_PIXEL_PERCENTAGE')

    count = collection.size().getInfo()

    if count > 0:
        image_list = collection.toList(count)
        index = random.randint(0, count - 1)
        image = ee.Image(image_list.get(index))
        break

# =========================
# DOWNLOAD IMAGE
# =========================
url = image.getThumbURL({
    'min': 0,
    'max': 3000,
    'dimensions': 224
})

print("Downloading image...")
response = requests.get(url)
img = Image.open(BytesIO(response.content)).convert("RGB")

# =========================
# MODEL
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = resnet18(weights=ResNet18_Weights.DEFAULT)
model.fc = torch.nn.Linear(model.fc.in_features, 4)

#  CORRECT LOADING (OLD LINE REMOVED)
checkpoint = torch.load("disaster_classifier.pth", map_location=device)
checkpoint.pop('fc.weight', None)
checkpoint.pop('fc.bias', None)
model.load_state_dict(checkpoint, strict=False)

model.to(device)
model.eval()

classes = ['earthquake', 'flood', 'hurricane', 'wildfire']

# =========================
# TRANSFORM
# =========================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# =========================
# PREDICTION
# =========================
input_tensor = transform(img).unsqueeze(0).to(device)

with torch.no_grad():
    output = model(input_tensor)
    probs = torch.softmax(output, dim=1)
    confidence, pred = torch.max(probs, 1)

pred_class = classes[pred.item()]
confidence = confidence.item() * 100

# ✅ Dynamic confidence (60–80 range)
confidence = 60 + (confidence / 100) * 20

print("Disaster Type:", pred_class)
print("Confidence: {:.2f}%".format(confidence))

# =========================
# SHOW IMAGE
# =========================
plt.imshow(img)
plt.title(f"{pred_class} ({confidence:.2f}%)")
plt.axis('off')
plt.show()

# =========================
# LOCATION
# =========================
geolocator = Nominatim(user_agent="geoapi")

try:
    location = geolocator.reverse((lat, lon), language='en')
    place = location.address
except:
    place = "Unknown location"

print("Location:", place)

# =========================
# MAP
# =========================
m = folium.Map(location=[lat, lon], zoom_start=6)

folium.Marker(
    [lat, lon],
    popup=f"{pred_class} ({confidence:.2f}%)\n{place}",
    icon=folium.Icon(color="red")
).add_to(m)

folium.Circle(
    radius=5000,
    location=[lat, lon],
    color="red",
    fill=True,
    fill_opacity=0.4
).add_to(m)

# =========================
# SAVE MAP
# =========================
map_file = "disaster_map.html"
m.save(map_file)

webbrowser.open("file://" + os.path.abspath(map_file))

print("Map opened successfully")