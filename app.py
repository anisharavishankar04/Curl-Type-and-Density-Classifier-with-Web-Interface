from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import os

# Set up Flask app
app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = models.efficientnet_b2(weights=None)
num_classes = 4
model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.to(device)
model.eval()

# Class labels
labels = ['2A-2B', '2C', '3A-3B', '3C']

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Routine data
routine_data = {
    "2A-2B_High Density": (
        "Follow these steps in order to enhance your waves curls:<br><br>"
        "♡ Use shampoo only on your scalp. While washing the shampoo off, squeeze it through the lengths of your hair. This will help clean your lengths without drying them out. Shampooing twice is generally recommended to cleanse your scalp well.<br><br>"
        "♡ Use a light-weight conditioner. You can also use a hair mask once a month, but be careful with how much you use and for how long you leave it on as it could weigh your hair down and affect your wave pattern.<br><br>"
        "♡ On soaking-wet hair , rake through a leave-in conditioner. This will really help tame frizz. Be mindful of the quantity as using too much could weigh your waves down. You can also use a curl cream (heavier than leave-in conditioner), but be extra careful that you’re not using too much as it could weigh your hair down. I would recommend sticking to a light leave-in conditioner unless a curl cream is absolutely necessary. A lot of the time, using a regular conditioner in the shower alone makes hair quite soft. If that's the case, you can skip the leave-in altogether.<br><br>"
        "♡ After raking through, scrunch your hair towards the roots - pulsating 4-5 times with each scrunch. Ensure that you don’t over scrunch as this could lead to frizz. Consider scrunching with your head flipped upside-down for better definition. You’ll know your hair is wet enough if you can hear the squishy sound when you scrunch.<br><br>"
        "♡ Once you’re done scrunching, use a few pumps of a curl mousse to give your hair hold. You can also use a gel. However, this might be too heavy depending on your individual hair type. Use it mindfully if you do opt for a gel. If you’re using mousse, use a couple of pumps and simply scrunch it into your hair - don’t rake. You can also add some at the roots for extra volume. If you’re using gel, use a small amount and dilute it with water. Glaze it over your hair (don’t rake) and scrunch your hair gently again.<br><br>"
        "♡ Now, to dry your hair - you can go one of two ways: air-drying or diffusing (my preference). Air drying generally gives you lesser volume. If you choose to diffuse your hair, ensure that you’re using a good quality heat protectant. Consider diffusing with your hair flipped upside-down for more volume. If you’re air-drying your hair, you can add clips to the roots of your hair to lift your hair away from your scalp to give your hair more volume at the roots.<br><br>"
        "♡ Once your hair is completely dry, you might notice that it’s crunchy. This is completely normal. All you need to do is use a little bit of a hair serum or a light-weight hair oil (stress on the light-weight) and scrunch your hair. This is called ‘scrunching out the crunch’.<br><br>"
        "♡ Use a scrunchie and a silk bonnet to make your waves last longer through the week. Too much friction can make your hair stretch out. You can also use a silk pillow case if you want to go all out.<br><br>"
    ),
    "2A-2B_Low Density": (
        "♡ Use shampoo only on your scalp. While washing the shampoo off, squeeze it through the lengths of your hair. This will help clean your lengths without drying them out. Shampooing twice is generally recommended to cleanse your scalp well.<br><br>"
        "♡ Use a lightweight conditioner. Using heavy products like masks could really weigh your hair down, affecting your wave pattern.<br><br>"
        "♡ On soaking-wet hair, rake through a light-weight product like a leave-in conditioner. This will really help tame frizz. Be mindful of the quantity as using too much could weigh your waves down. Ensure that you’re not using any heavy products like curl creams or butters as that will most likely weigh your hair down. <br><br>"
        "♡ After raking through, scrunch your hair towards the roots - pulsating 4-5 times with each scrunch. Ensure that you don’t over scrunch as this could lead to frizz. Consider scrunching with your head flipped upside-down for better definition. You’ll know your hair is wet enough if you can hear the squishy sound when you scrunch. A lot of the time, using a regular conditioner in the shower alone makes hair quite soft. If that's the case, you can skip the leave-in altogether.<br><br>"
        "♡ Once you’re done scrunching, use a few pumps of a curl mousse to give your hair hold. You can also use a gel. However, this might end up becoming too heavy for wavy hair with high porosity. A mousse is usually a better choice. Use a couple of pumps and simply scrunch it into your hair - don’t rake. You can also add some at the roots for extra volume.<br><br>"
        "♡ Now, to dry your hair - you can go one of two ways: air-drying or diffusing (my preference). Air drying generally gives you lesser volume. If you choose to diffuse your hair, ensure that you’re using a good quality heat protectant. Consider diffusing with your hair flipped upside-down for more volume. If you’re air-drying your hair, you can add clips to the roots of your hair to lift your hair away from your scalp to give your hair more volume at the roots.<br><br>"
        "♡ Once your hair is completely dry, you might notice that it’s crunchy. This is completely normal. All you need to do is use a little bit of a hair serum or a light-weight hair oil (stress on the light-weight) and scrunch your hair. This is called ‘scrunching out the crunch’.<br><br>"
        "♡ Use a scrunchie and a silk bonnet to make your waves last longer through the week. Too much friction can make your hair stretch out. You can also use a silk pillow case if you want to go all out.<br><br>"
    ),
    "2C_High Density": (
        "Follow these steps in order to enhance your waves curls:<br><br>"
        "♡ Use shampoo only on your scalp. While washing the shampoo off, squeeze it through the lengths of your hair. This will help clean your lengths without drying them out. Shampooing twice is generally recommended to cleanse your scalp well.<br><br>"
        "♡ Use a light-weight conditioner. You can also use a hair mask once a month, but be careful with how much you use and for how long you leave it on as it could weigh your hair down and affect your wave pattern.<br><br>"
        "♡ On soaking-wet hair , rake through a leave-in conditioner. This will really help tame frizz. Be mindful of the quantity as using too much could weigh your waves down. You can also use a curl cream (heavier than leave-in conditioner), but be extra careful that you’re not using too much as it could weigh your hair down. I would recommend sticking to a light leave-in conditioner unless a curl cream is absolutely necessary. A lot of the time, using a regular conditioner in the shower alone makes hair quite soft. If that's the case, you can skip the leave-in altogether.<br><br>"
        "♡ After raking through, scrunch your hair towards the roots - pulsating 4-5 times with each scrunch. Ensure that you don’t over scrunch as this could lead to frizz. Consider scrunching with your head flipped upside-down for better definition. You’ll know your hair is wet enough if you can hear the squishy sound when you scrunch.<br><br>"
        "♡ Once you’re done scrunching, use a few pumps of a curl mousse to give your hair hold. You can also use a gel. However, this might be too heavy depending on your individual hair type. Use it mindfully if you do opt for a gel. If you’re using mousse, use a couple of pumps and simply scrunch it into your hair - don’t rake. You can also add some at the roots for extra volume. If you’re using gel, use a small amount and dilute it with water. Glaze it over your hair (don’t rake) and scrunch your hair gently again.<br><br>"
        "♡ Now, to dry your hair - you can go one of two ways: air-drying or diffusing (my preference). Air drying generally gives you lesser volume. If you choose to diffuse your hair, ensure that you’re using a good quality heat protectant. Consider diffusing with your hair flipped upside-down for more volume. If you’re air-drying your hair, you can add clips to the roots of your hair to lift your hair away from your scalp to give your hair more volume at the roots.<br><br>"
        "♡ Once your hair is completely dry, you might notice that it’s crunchy. This is completely normal. All you need to do is use a little bit of a hair serum or a light-weight hair oil (stress on the light-weight) and scrunch your hair. This is called ‘scrunching out the crunch’.<br><br>"
        "♡ Use a scrunchie and a silk bonnet to make your waves last longer through the week. Too much friction can make your hair stretch out. You can also use a silk pillow case if you want to go all out.<br><br>"
    ),
    "2C_Low Density": (
        "♡ Use shampoo only on your scalp. While washing the shampoo off, squeeze it through the lengths of your hair. This will help clean your lengths without drying them out. Shampooing twice is generally recommended to cleanse your scalp well.<br><br>"
        "♡ Use a lightweight conditioner. Using heavy products like masks could really weigh your hair down, affecting your wave pattern.<br><br>"
        "♡ On soaking-wet hair, rake through a light-weight product like a leave-in conditioner. This will really help tame frizz. Be mindful of the quantity as using too much could weigh your waves down. Ensure that you’re not using any heavy products like curl creams or butters as that will most likely weigh your hair down. <br><br>"
        "♡ After raking through, scrunch your hair towards the roots - pulsating 4-5 times with each scrunch. Ensure that you don’t over scrunch as this could lead to frizz. Consider scrunching with your head flipped upside-down for better definition. You’ll know your hair is wet enough if you can hear the squishy sound when you scrunch. A lot of the time, using a regular conditioner in the shower alone makes hair quite soft. If that's the case, you can skip the leave-in altogether.<br><br>"
        "♡ Once you’re done scrunching, use a few pumps of a curl mousse to give your hair hold. You can also use a gel. However, this might end up becoming too heavy for wavy hair with high porosity. A mousse is usually a better choice. Use a couple of pumps and simply scrunch it into your hair - don’t rake. You can also add some at the roots for extra volume.<br><br>"
        "♡ Now, to dry your hair - you can go one of two ways: air-drying or diffusing (my preference). Air drying generally gives you lesser volume. If you choose to diffuse your hair, ensure that you’re using a good quality heat protectant. Consider diffusing with your hair flipped upside-down for more volume. If you’re air-drying your hair, you can add clips to the roots of your hair to lift your hair away from your scalp to give your hair more volume at the roots.<br><br>"
        "♡ Once your hair is completely dry, you might notice that it’s crunchy. This is completely normal. All you need to do is use a little bit of a hair serum or a light-weight hair oil (stress on the light-weight) and scrunch your hair. This is called ‘scrunching out the crunch’.<br><br>"
        "♡ Use a scrunchie and a silk bonnet to make your waves last longer through the week. Too much friction can make your hair stretch out. You can also use a silk pillow case if you want to go all out.<br><br>"
    ),
    "3A-3B_High Density": (
        "♡ Use shampoo only on your scalp. While washing the shampoo off, squeeze it through the lengths of your hair. This will help clean your lengths without drying them out. Shampooing twice is generally recommended to cleanse your scalp well.<br><br>"
        "♡ Use a light-weight conditioner. You can also use a hair mask once a month, but be careful with how much you use and for how long you leave it on as it could weigh your hair down and affect your curl pattern.<br><br>"
        "♡ On soaking-wet hair , rake through a light-weight curl cream.. This will really help tame frizz. Ensure not to use too much as this could weigh your curls down.<br><br>"
        "♡ After raking through, scrunch your hair towards the roots - pulsating 4-5 times with each scrunch. Ensure that you don’t over scrunch as this could lead to frizz. Consider scrunching with your head flipped upside-down for better definition. You’ll know your hair is wet enough if you can hear the squishy sound when you scrunch.<br><br>"
        "♡ Once you’re done scrunching, use some curl gel to give your hair hold. Use a small amount and dilute it with water. Glaze it over your hair using the ‘praying hands’ method (don’t rake) and scrunch your hair gently again. You can also add a pump or two of curl mousse at your roots for additional volume.<br><br>"
        "♡ Now, to dry your hair - you can go one of two ways: air-drying or diffusing (my preference). Air drying generally gives you lesser volume. If you choose to diffuse your hair, ensure that you’re using a good quality heat protectant. Consider diffusing with your hair flipped upside-down for more volume. If you’re air-drying your hair, you can add clips to the roots of your hair to lift your hair away from your scalp to give your hair more volume at the roots.<br><br>"
        "♡ Once your hair is completely dry, you might notice that it’s crunchy. This is completely normal. All you need to do is use a little bit of a hair serum or a light-weight hair oil (stress on the light-weight) and scrunch your hair. This is called ‘scrunching out the crunch’.<br><br>"
        "♡ Use a scrunchie and a silk bonnet to make your curls last longer through the week. Too much friction can make your hair stretch out. You can also use a silk pillow case if you want to go all out.<br><br>"
    ),
    "3A-3B_Low Density": (
        "a. Use shampoo only on your scalp. While washing the shampoo off, squeeze it through the lengths of your hair. This will help clean your lengths without drying them out. Shampooing twice is generally recommended to cleanse your scalp well. <br><br>"
        "b. Use a light-weight conditioner. You can also use a hair mask once a month, but be careful with how much you use and for how long you leave it on as it could weigh your hair down and affect your curl pattern.<br><br>"
        "c. On soaking-wet hair , rake through a curl cream. This will really help tame frizz. Ensure not to use too much as this could weigh your curls down. You can also use a curl butter (heavier than curl cream) if your hair is particularly dry.<br><br>"
        "d. After raking through, scrunch your hair towards the roots - pulsating 4-5 times with each scrunch. Ensure that you don’t over scrunch as this could lead to frizz. Consider scrunching with your head flipped upside-down for better definition. You’ll know your hair is wet enough if you can hear the squishy sound when you scrunch.<br><br>"
        "e. Once you’re done scrunching, use some curl gel to give your hair hold. Use a small amount and dilute it with water. Glaze it over your hair using the ‘praying hands’ method (don’t rake) and scrunch your hair gently again. You can also add a pump or two of curl mousse at your roots for additional volume.<br><br>"
        "f. Now, to dry your hair - you can go one of two ways: air-drying or diffusing (my preference). Air drying generally gives you lesser volume. If you choose to diffuse your hair, ensure that you’re using a good quality heat protectant. Consider diffusing with your hair flipped upside-down for more volume. If you’re air-drying your hair, you can add clips to the roots of your hair to lift your hair away from your scalp to give your hair more volume at the roots.<br><br>"
        "g. Once your hair is completely dry, you might notice that it’s crunchy. This is completely normal. All you need to do is use a little bit of a hair serum or a light-weight hair oil (stress on the light-weight) and scrunch your hair. This is called ‘scrunching out the crunch’.<br><br>"
        "h. Use a scrunchie and a silk bonnet to make your waves last longer through the week. Too much friction can make your hair stretch out. You can also use a silk pillow case if you want to go all out.<br><br>"
    ),
    "3C_High Density": (
        "♡ Use shampoo only on your scalp. While washing the shampoo off, squeeze it through the lengths of your hair. This will help clean your lengths without drying them out. Shampooing twice is generally recommended to cleanse your scalp well.<br><br>"
        "♡ Use a light-weight conditioner. You can also use a hair mask once a month, but be careful with how much you use and for how long you leave it on as it could weigh your hair down and affect your curl pattern.<br><br>"
        "♡ On soaking-wet hair , rake through a light-weight curl cream.. This will really help tame frizz. Ensure not to use too much as this could weigh your curls down.<br><br>"
        "♡ After raking through, scrunch your hair towards the roots - pulsating 4-5 times with each scrunch. Ensure that you don’t over scrunch as this could lead to frizz. Consider scrunching with your head flipped upside-down for better definition. You’ll know your hair is wet enough if you can hear the squishy sound when you scrunch.<br><br>"
        "♡ Once you’re done scrunching, use some curl gel to give your hair hold. Use a small amount and dilute it with water. Glaze it over your hair using the ‘praying hands’ method (don’t rake) and scrunch your hair gently again. You can also add a pump or two of curl mousse at your roots for additional volume.<br><br>"
        "♡ Now, to dry your hair - you can go one of two ways: air-drying or diffusing (my preference). Air drying generally gives you lesser volume. If you choose to diffuse your hair, ensure that you’re using a good quality heat protectant. Consider diffusing with your hair flipped upside-down for more volume. If you’re air-drying your hair, you can add clips to the roots of your hair to lift your hair away from your scalp to give your hair more volume at the roots.<br><br>"
        "♡ Once your hair is completely dry, you might notice that it’s crunchy. This is completely normal. All you need to do is use a little bit of a hair serum or a light-weight hair oil (stress on the light-weight) and scrunch your hair. This is called ‘scrunching out the crunch’.<br><br>"
        "♡ Use a scrunchie and a silk bonnet to make your curls last longer through the week. Too much friction can make your hair stretch out. You can also use a silk pillow case if you want to go all out.<br><br>"
    ),
    "3C_Low Density": (
        "a. Use shampoo only on your scalp. While washing the shampoo off, squeeze it through the lengths of your hair. This will help clean your lengths without drying them out. Shampooing twice is generally recommended to cleanse your scalp well. <br><br>"
        "b. Use a light-weight conditioner. You can also use a hair mask once a month, but be careful with how much you use and for how long you leave it on as it could weigh your hair down and affect your curl pattern.<br><br>"
        "c. On soaking-wet hair , rake through a curl cream. This will really help tame frizz. Ensure not to use too much as this could weigh your curls down. You can also use a curl butter (heavier than curl cream) if your hair is particularly dry.<br><br>"
        "d. After raking through, scrunch your hair towards the roots - pulsating 4-5 times with each scrunch. Ensure that you don’t over scrunch as this could lead to frizz. Consider scrunching with your head flipped upside-down for better definition. You’ll know your hair is wet enough if you can hear the squishy sound when you scrunch.<br><br>"
        "e. Once you’re done scrunching, use some curl gel to give your hair hold. Use a small amount and dilute it with water. Glaze it over your hair using the ‘praying hands’ method (don’t rake) and scrunch your hair gently again. You can also add a pump or two of curl mousse at your roots for additional volume.<br><br>"
        "f. Now, to dry your hair - you can go one of two ways: air-drying or diffusing (my preference). Air drying generally gives you lesser volume. If you choose to diffuse your hair, ensure that you’re using a good quality heat protectant. Consider diffusing with your hair flipped upside-down for more volume. If you’re air-drying your hair, you can add clips to the roots of your hair to lift your hair away from your scalp to give your hair more volume at the roots.<br><br>"
        "g. Once your hair is completely dry, you might notice that it’s crunchy. This is completely normal. All you need to do is use a little bit of a hair serum or a light-weight hair oil (stress on the light-weight) and scrunch your hair. This is called ‘scrunching out the crunch’.<br><br>"
        "h. Use a scrunchie and a silk bonnet to make your waves last longer through the week. Too much friction can make your hair stretch out. You can also use a silk pillow case if you want to go all out.<br><br>"
    ),
}

# Classification
def classify_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
    return labels[predicted.item()]

# Density
def get_density_result(q1, q2, q3):
    score = int(q1) + int(q2) + int(q3)
    return "High Density" if score >= 2 else "Low Density"

# Routes
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        image = request.files["image"]
        if image.filename != "":
            filename = secure_filename(image.filename)
            path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            image.save(path)

            curl_type = classify_image(path)
            q1 = request.form.get("q1")
            q2 = request.form.get("q2")
            q3 = request.form.get("q3")
            density = get_density_result(q1, q2, q3)

            key = f"{curl_type}_{density}"
            routine = routine_data.get(key, "No routine available.")

            return render_template("results.html", curl=curl_type,
                                   density=density, advice=routine, image_file=filename)
    return render_template("index.html", result=False)

if __name__ == "__main__":
    app.run(debug=True)
