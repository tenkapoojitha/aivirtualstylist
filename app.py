from flask import Flask, render_template, request, jsonify
import cv2
import mediapipe as mp
import numpy as np
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "static/uploads"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

mp_face_mesh = mp.solutions.face_mesh
mp_pose = mp.solutions.pose

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def analyze_image(image_path):
    try:
        img = cv2.imread(image_path)
        if img is None:
            return {"error": "Invalid image"}

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, _ = img.shape

        # --- FACE LANDMARKS ---
        face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, refine_landmarks=True)
        face_result = face_mesh.process(rgb)

        # --- BODY LANDMARKS ---
        pose = mp_pose.Pose(static_image_mode=True)
        pose_result = pose.process(rgb)

        # ---- SKIN TONE ----
        small = cv2.resize(img, (100, 100))
        avg_color = np.mean(small, axis=(0, 1))
        brightness = np.mean(avg_color)

        if brightness > 180:
            skin_tone = "Fair"
        elif brightness > 130:
            skin_tone = "Medium"
        else:
            skin_tone = "Dark"

        # --- GUARANTEED UNDERTONE ---
        undertone = "Neutral"  # default fallback
        b, g, r = avg_color
        if (r - g > 20) and (r - b > 20):
            undertone = "Warm"
        elif (b - r > 20) and (b - g > 20):
            undertone = "Cool"
        undertone = undertone.capitalize()  # "Warm", "Cool", "Neutral"

        # --- FACE SHAPE ---
        face_shape = "Not detected"
        if face_result.multi_face_landmarks:
            lm = face_result.multi_face_landmarks[0].landmark
            chin = lm[152]
            forehead = lm[10]
            left_cheek = lm[234]
            right_cheek = lm[454]

            face_length = abs(forehead.y - chin.y)
            face_width = abs(right_cheek.x - left_cheek.x)
            ratio = face_length / face_width

            jaw_width = abs(lm[172].x - lm[397].x)
            forehead_width = abs(lm[108].x - lm[338].x)

            if ratio > 1.5:
                face_shape = "Heart" if forehead_width > jaw_width else "Long"
            elif 1.3 < ratio <= 1.5:
                if forehead_width < jaw_width:
                    face_shape = "Triangle"
                elif abs(forehead_width - jaw_width) < 0.05:
                    face_shape = "Rectangle"
                else:
                    face_shape = "Oval"
            elif ratio < 1.1:
                face_shape = "Round" if abs(forehead_width - jaw_width) < 0.02 else "Diamond"
            else:
                face_shape = "Square"

        # --- BODY TYPE (with APPLE) ---
        body_type = "Not detected"
        if pose_result.pose_landmarks:
            lmk = pose_result.pose_landmarks.landmark
            shoulder = abs(lmk[mp_pose.PoseLandmark.LEFT_SHOULDER].x - lmk[mp_pose.PoseLandmark.RIGHT_SHOULDER].x)
            hip = abs(lmk[mp_pose.PoseLandmark.LEFT_HIP].x - lmk[mp_pose.PoseLandmark.RIGHT_HIP].x)
            waist = abs(lmk[mp_pose.PoseLandmark.LEFT_ELBOW].x - lmk[mp_pose.PoseLandmark.RIGHT_ELBOW].x) * 0.6

            if shoulder > hip * 1.2:
                body_type = "Inverted Triangle"
            elif hip > shoulder * 1.2:
                body_type = "Pear"
            elif waist > shoulder and waist > hip:
                body_type = "Apple"
            elif abs(shoulder - hip) < 0.05:
                body_type = "Rectangle"
            else:
                body_type = "Hourglass"

        # --- HAIR TEXTURE ---
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.Laplacian(gray, cv2.CV_64F).var()
        if blur < 40:
            hair_texture = "Straight"
        elif 40 <= blur < 80:
            hair_texture = "Wavy"
        else:
            hair_texture = "Curly"

        # --- SKIN TYPE ---
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        contrast = np.std(gray_img)
        if contrast < 25:
            skin_type = "Dry"
        elif 25 <= contrast < 50:
            skin_type = "Normal"
        elif 50 <= contrast < 70:
            skin_type = "Combination"
        else:
            skin_type = "Oily"

        # --- HAIR TYPE ---
        avg_brightness = np.mean(gray_img)
        if avg_brightness > 160:
            hair_type = "Oily"
        elif avg_brightness < 80:
            hair_type = "Dry"
        else:
            hair_type = "Normal"

        # --- MAKEUP RECOMMENDATIONS (Detailed) ---
        makeup = {
            "Warm": {
                "Lipstick": ["Peach", "Coral", "Terracotta", "Warm Nude"],
                "Foundation": ["Golden Beige", "Warm Honey", "Caramel Glow"],
                "Concealer": ["Yellow Undertone", "Golden Neutral"],
                "Blush": ["Apricot", "Peach", "Coral"],
                "Contour": ["Golden Brown", "Warm Bronze"],
                "Eyeshadow": ["Gold", "Copper", "Bronze", "Champagne"]
            },
            "Cool": {
                "Lipstick": ["Rose", "Berry", "Mauve", "Pink Nude"],
                "Foundation": ["Rosy Beige", "Cool Ivory", "Soft Almond"],
                "Concealer": ["Pink Undertone", "Porcelain Cool"],
                "Blush": ["Rosy Pink", "Soft Plum", "Berry"],
                "Contour": ["Ash Brown", "Cool Taupe"],
                "Eyeshadow": ["Silver", "Lilac", "Cool Brown", "Icy Blue"]
            },
            "Neutral": {
                "Lipstick": ["Soft Pink", "Nude", "Rosewood"],
                "Foundation": ["Neutral Beige", "Honey Sand", "Light Tan"],
                "Concealer": ["Neutral Undertone", "Beige Neutral"],
                "Blush": ["Neutral Pink", "Peachy Nude"],
                "Contour": ["Neutral Brown", "Soft Bronze"],
                "Eyeshadow": ["Champagne", "Rose Gold", "Soft Brown", "Taupe"]
            }
        }

        # --- DRESS STYLES ---
        dress_styles = {
            "Inverted Triangle": ["A-line dresses", "V-neck tops", "Flared skirts"],
            "Pear": ["Off-shoulder tops", "Wide-leg pants", "Fit and flare dresses"],
            "Rectangle": ["Peplum tops", "Belted dresses", "Layered outfits"],
            "Hourglass": ["Bodycon dresses", "Wrap tops", "High-waist skirts"],
            "Apple": ["Empire waist dresses", "Flowy tops", "Straight-cut pants"],
            "Not detected": ["Classic styles that suit all shapes"],
        }

        # --- SKINCARE RECOMMENDATIONS ---
        skincare_tips = {
            "Dry": ["Use hydrating cleanser and rich moisturizer.", "Avoid hot water; use lukewarm.", "Apply facial oil before sleeping."],
            "Oily": ["Use salicylic acid or clay-based cleanser.", "Moisturize with oil-free gel.", "Blot excess oil during the day."],
            "Combination": ["Use gentle cleanser and hydrate dry areas.", "Toner on T-zone to control oil.", "Weekly exfoliation helps balance skin."],
            "Normal": ["Stick to gentle cleanser and SPF daily.", "Use light moisturizer.", "Exfoliate 1–2 times weekly."],
            "Sensitive": ["Avoid fragrance-based products.", "Use calming aloe or chamomile toner.", "Do patch tests before new products."]
        }

        # --- HAIRCARE RECOMMENDATIONS ---
        haircare_tips = {
            "Dry": ["Apply coconut or argan oil weekly.", "Avoid heat styling tools.", "Use moisturizing shampoo."],
            "Oily": ["Shampoo regularly with clarifying formula.", "Avoid heavy conditioners on scalp.", "Rinse thoroughly to prevent buildup."],
            "Normal": ["Use mild shampoo and conditioner.", "Trim regularly for healthy ends.", "Protect from sun exposure."]
        }

        # ✅ Always have valid makeup shades
        makeup_shades = makeup.get(undertone, makeup["Neutral"])

        return {
            "face_type": face_shape,
            "body_type": body_type,
            "hair_texture": hair_texture,
            "hair_type": hair_type,
            "skin_tone": skin_tone,
            "skin_type": skin_type,
            "undertone": undertone,
            "makeup_shades": makeup_shades,  # always returned explicitly
            "recommendations": {
                "makeup": makeup_shades,
                "dress_colors": {
                    "Warm": ["Coral", "Olive", "Gold", "Rust"],
                    "Cool": ["Lavender", "Blue", "Emerald", "Silver"],
                    "Neutral": ["Beige", "Gray", "Cream", "Soft pastels"],
                }.get(undertone, ["Beige", "Soft pink"]),
                "jewellery": {
                    "Warm": ["Gold", "Bronze", "Copper"],
                    "Cool": ["Silver", "Platinum", "Pearl"],
                    "Neutral": ["Rose gold", "Mixed metals"],
                }.get(undertone, ["Gold"]),
                "dress_style": dress_styles.get(body_type, ["Casual classic fits"]),
                "skincare": skincare_tips.get(skin_type, ["Maintain basic skin hydration."]),
                "haircare": haircare_tips.get(hair_type, ["Maintain scalp cleanliness."]),
                "hairstyle": {
                    "Oval": ["Long waves", "Bob cut", "Side-swept bangs"],
                    "Round": ["Layered cut", "High ponytail", "Long straight hair"],
                    "Square": ["Soft curls", "Feathered layers", "Side fringe"],
                    "Heart": ["Chin-length bob", "Textured lob", "Wavy fringe"],
                    "Diamond": ["Side part waves", "Chin-length bob", "Soft layers"],
                    "Triangle": ["Curly top volume", "Layered side bangs", "Rounded bob"],
                    "Rectangle": ["Wavy lob", "Soft curls", "Layered ends"],
                    "Long": ["Curtain bangs", "Soft waves", "Long layered cut"],
                }.get(face_shape, ["Classic layered style"]),
            },
        }

    except Exception as e:
        print("Error:", e)
        return {"error": str(e)}


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        file = request.files.get("file")
        if not file or not allowed_file(file.filename):
            return jsonify({"error": "Please upload a valid image."})

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        result = analyze_image(filepath)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    print("🚀 Running AI Stylist at: http://127.0.0.1:5000")
    app.run(debug=True)