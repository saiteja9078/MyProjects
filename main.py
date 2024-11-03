import streamlit as st
import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model("trained_plant_disease_model.keras")
# Tensorflow Model Prediction
def model_prediction(test_image):
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions)  # return index of max element

# Sidebar
st.sidebar.title("Plant Disease Dashboard")
app_mode = st.sidebar.radio(
    "Navigate",
    ["Home", "About", "Disease Recognition"],
    index=0,
    help="Use the navigation to explore the app!"
)

# Styling for the app
st.markdown(
    """
    <style>
    .main-header {
        text-align: center;
        color: #2E8B57;
        font-size: 36px;
        font-weight: bold;
        margin-bottom: 20px;
    }
    .sub-header {
        color: #4682B4;
        font-size: 24px;
        font-weight: semi-bold;
    }
    .custom-button {
        background-color: #4682B4;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 8px 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Main Page
if app_mode == "Home":
    st.markdown('<div class="main-header">AgroVision: Deep learning-based detection of plant diseases</div>', unsafe_allow_html=True)
    st.image("home1.jpg", use_column_width=True)
    st.markdown("""
    Welcome to the **Plant Disease Recognition System**! 🌿🔍

    **How It Works:**
    - Upload an image of a plant with suspected disease.
    - The system will analyze it to detect potential diseases.
    - Get results and recommendations instantly!

    ### Features
    - **High Accuracy:** Using advanced ML algorithms.
    - **User-Friendly:** Intuitive interface for easy use.
    - **Fast Results:** Quick analysis for prompt action.

    Navigate to the **Disease Recognition** page to start.
    """)

# About Project
elif app_mode == "About":
    st.markdown('<div class="main-header">About the Project</div>', unsafe_allow_html=True)
    st.markdown("""
    ### Dataset Information
    - The dataset is derived from a public repository, containing 87K images of healthy and diseased crops.
    - Categorized into 38 classes, split into training (80%) and validation (20%).
    
    **Dataset Content:**
    - **Train:** 70,295 images
    - **Test:** 33 images
    - **Validation:** 17,572 images
    """)

# Disease Recognition
elif app_mode == "Disease Recognition":
    st.markdown('<div class="main-header">Disease Recognition</div>', unsafe_allow_html=True)
    test_image = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
    
    if test_image:
        st.image(test_image, use_column_width=True, caption="Uploaded Image")
    
    if st.button("Predict", key="predict_button", help="Click to predict disease"):

        st.snow()
        st.write("Analyzing Image...")
        result_index = model_prediction(test_image)
        class_name = [
    'Apple: Apple scab', 
    'Apple: Black rot', 
    'Apple: Cedar apple rust', 
    'Apple: Healthy',
    'Blueberry: Healthy', 
    'Cherry (including sour): Powdery mildew', 
    'Cherry (including sour): Healthy',
    'Corn (maize): Cercospora leaf spot, Gray leaf spot', 
    'Corn (maize): Common rust', 
    'Corn (maize): Northern Leaf Blight', 
    'Corn (maize): Healthy',
    'Grape: Black rot', 
    'Grape: Esca (Black Measles)', 
    'Grape: Leaf blight (Isariopsis Leaf Spot)', 
    'Grape: Healthy',
    'Orange: Huanglongbing (Citrus greening)', 
    'Peach: Bacterial spot', 
    'Peach: Healthy',
    'Pepper (bell): Bacterial spot', 
    'Pepper (bell): Healthy',
    'Potato: Early blight', 
    'Potato: Late blight', 
    'Potato: Healthy',
    'Raspberry: Healthy', 
    'Soybean: Healthy', 
    'Squash: Powdery mildew',
    'Strawberry: Leaf scorch', 
    'Strawberry: Healthy',
    'Tomato: Bacterial spot', 
    'Tomato: Early blight', 
    'Tomato: Late blight', 
    'Tomato: Leaf Mold',
    'Tomato: Septoria leaf spot', 
    'Tomato: Spider mites (Two-spotted spider mite)',
    'Tomato: Target Spot', 
    'Tomato: Tomato Yellow Leaf Curl Virus', 
    'Tomato: Tomato mosaic virus',
    'Tomato: Healthy'
        ]
        plant_disease_treatments = {
    "Apple: Apple scab": "Apply fungicides during the early season. Remove and destroy fallen leaves and infected fruit. Use resistant apple varieties.",
    "Apple: Black rot": "Prune and destroy infected twigs, branches, and fruits. Apply fungicides preventatively. Maintain tree health with proper fertilization and watering.",
    "Apple: Cedar apple rust": "Apply fungicides early in the season. Remove nearby cedar trees if possible, as they are alternate hosts.",
    "Apple: Healthy": "No treatment needed. Maintain regular care to keep the plant healthy.",
    "Blueberry: Healthy": "No treatment needed. Ensure proper soil conditions and adequate watering.",
    "Cherry (including sour): Powdery mildew": "Apply sulfur-based or fungicidal sprays. Increase air circulation and avoid overhead watering.",
    "Cherry (including sour): Healthy": "No treatment needed. Continue regular disease prevention practices.",
    "Corn (maize): Cercospora leaf spot, Gray leaf spot": "Use resistant varieties. Apply fungicides if necessary and practice crop rotation.",
    "Corn (maize): Common rust": "Plant resistant varieties and apply fungicides if needed. Maintain good field hygiene.",
    "Corn (maize): Northern Leaf Blight": "Use resistant seeds and rotate crops. Apply fungicides if required.",
    "Corn (maize): Healthy": "No treatment needed. Maintain regular care.",
    "Grape: Black rot": "Prune and remove infected vines. Apply fungicides in early spring. Ensure good air circulation.",
    "Grape: Esca (Black Measles)": "Remove infected wood. Ensure vines are not stressed and avoid trunk or root injuries.",
    "Grape: Leaf blight (Isariopsis Leaf Spot)": "Apply fungicides and prune affected leaves. Increase ventilation around vines.",
    "Grape: Healthy": "No treatment needed. Maintain proper care.",
    "Orange: Huanglongbing (Citrus greening)": "No known cure. Remove and destroy infected trees. Control psyllid vectors and plant disease-free stock.",
    "Peach: Bacterial spot": "Apply copper-based bactericides. Use resistant varieties if available. Prune and remove affected parts.",
    "Peach: Healthy": "No treatment needed. Ensure good tree health.",
    "Pepper (bell): Bacterial spot": "Apply copper-based sprays. Practice crop rotation and use certified disease-free seeds.",
    "Pepper (bell): Healthy": "No treatment needed. Maintain regular care.",
    "Potato: Early blight": "Apply fungicides and remove infected plant debris. Practice crop rotation.",
    "Potato: Late blight": "Use resistant varieties. Apply fungicides and destroy infected plants.",
    "Potato: Healthy": "No treatment needed. Continue regular monitoring.",
    "Raspberry: Healthy": "No treatment needed. Maintain proper care and disease prevention practices.",
    "Soybean: Healthy": "No treatment needed. Practice regular crop management.",
    "Squash: Powdery mildew": "Apply fungicides. Improve air circulation and avoid overhead watering.",
    "Strawberry: Leaf scorch": "Remove and destroy infected leaves. Apply appropriate fungicides.",
    "Strawberry: Healthy": "No treatment needed. Maintain regular care.",
    "Tomato: Bacterial spot": "Use copper sprays and disease-resistant seeds. Practice crop rotation.",
    "Tomato: Early blight": "Apply fungicides and remove affected plant parts.",
    "Tomato: Late blight": "Apply fungicides and remove and destroy infected plants. Use resistant varieties.",
    "Tomato: Leaf Mold": "Increase air circulation and apply fungicides. Prune lower leaves.",
    "Tomato: Septoria leaf spot": "Apply fungicides and remove infected leaves. Ensure good ventilation.",
    "Tomato: Spider mites (Two-spotted spider mite)": "Use insecticidal soap or horticultural oil. Increase humidity around plants.",
    "Tomato: Target Spot": "Apply fungicides and remove infected plant debris.",
    "Tomato: Tomato Yellow Leaf Curl Virus": "Control whitefly populations. Use resistant varieties and remove infected plants.",
    "Tomato: Tomato mosaic virus": "Remove and destroy infected plants. Sanitize tools and avoid tobacco products.",
    "Tomato: Healthy": "No treatment needed. Maintain regular plant care."
}
        st.success(f'''
                   Disease is {class_name[result_index]} , 
                   Diagnosis:
                   \n
                   {plant_disease_treatments[class_name[result_index]]}
''')
