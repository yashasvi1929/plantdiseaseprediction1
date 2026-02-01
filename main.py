import streamlit as st
from streamlit.components.v1 import html
import tensorflow as tf
import numpy as np
from PIL import Image
import time

#Tensorflow Model Prediction
# def model_prediction(test_image):
#   model = tf.keras.models.load_model('trained_model.keras')
#   image = tf.keras.preprocessing.image.load_img(image_path,target_size(128,128))
#   input_arr = tf.keras.preprocessing.image.img_to_array(image)
#   input_arr = np.array([input_arr])
#   prediction = model.predict(input_arr)
#   result_index = np.argmax(prediction) 
#   return result_index

def model_prediction(test_image):
    # Load model
    model = tf.keras.models.load_model('trained_model.keras')
    
    # Convert the uploaded file to an image using PIL
    image = Image.open(test_image)  # 'test_image' is a file-like object, not a file path
    
    # Resize the image to match the input shape expected by the model (128x128 in your case)
    image = image.resize((128, 128))
    
    # Convert image to numpy array and add batch dimension
    input_arr = np.array(image)  # Convert to numpy array
    input_arr = np.expand_dims(input_arr, axis=0)  # Add batch dimension
    
    # Normalize the image if needed (based on your model's training)
    input_arr = input_arr / 255.0  # Assuming model was trained with normalized images
    
    # Predict
    prediction = model.predict(input_arr)
    
    # Get the index of the highest prediction score
    result_index = np.argmax(prediction)
    
    return result_index



#Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page",["Home","About","Disease Recognition"])

#Home Page
if(app_mode == "Home"):
  st.header("PLANT DISEASE RECOGNITION SYSTEM")
  image_path = "home_page.jpeg"
  st.image(image_path,use_container_width=True)
  st.markdown("""
    Welcome to the Plant Disease Recognition System! 🌿🔍
    
    Our mission is to help in identifying plant diseases efficiently. Upload an image of a plant, and our system will analyze it to detect any signs of diseases. Together, let's protect our crops and ensure a healthier harvest!

    ### How It Works
    1. *Upload Image:* Go to the *Disease Recognition* page and upload an image of a plant with suspected diseases.
    2. *Analysis:* Our system will process the image using advanced algorithms to identify potential diseases.
    3. *Results:* View the results and recommendations for further action.

    ### Why Choose Us?
    - *Accuracy:* Our system utilizes state-of-the-art machine learning techniques for accurate disease detection.
    - *User-Friendly:* Simple and intuitive interface for seamless user experience.
    - *Fast and Efficient:* Receive results in seconds, allowing for quick decision-making.

    ### Get Started
    Click on the *Disease Recognition* page in the sidebar to upload an image and experience the power of our Plant Disease Recognition System!

    ### About Us
    Learn more about the project, our team, and our goals on the *About* page.
    """)
  
#About Page
elif(app_mode == "About"):
  st.header("About")
  image_path = "about_page.jpeg"
  st.image(image_path,use_container_width=True)
  st.markdown("""
  #### About Dataset
  This dataset is recreated using offline augmentation from the original dataset. The original dataset can be found on 
  #### Content
  1. Train (70295 images)
  2. Valid (17572 images)
  3. Test (33 images)
""")
  st.markdown(""" 
####  Dataset Structure
              Typically, datasets are structured into training, validation, and testing subsets, where:

Training set: A large portion of the dataset used to train the model.
Validation set: A smaller portion used to tune model parameters and prevent overfitting.
Testing set: A separate portion to evaluate the performance of the trained model.
The images in the dataset are often organized into folders where each folder corresponds to a specific disease or plant health status.
              """)
  
  st.markdown("""
####  Applications
              Disease Detection: The main goal of these datasets is to develop machine learning models that can automatically detect plant diseases from images.
Precision Agriculture: By automating the detection of plant diseases, these models can help farmers identify and treat plant diseases early, improving crop yield and quality.
Research: These datasets are used in research to develop new algorithms, improve diagnostic systems, and explore new ways to detect and classify plant diseases.
              """)
  st.markdown("""
#### Challenges in Plant Disease Detection
              Image Quality: The quality of the images can vary, and inconsistent lighting, angles, or image resolution can make detection harder.
Data Imbalance: Some diseases might have fewer images than others, which can affect model training and performance.
Complexity of Symptoms: Different diseases can have similar symptoms, making it difficult for models to distinguish between them.
Environmental Factors: Plant health can also be influenced by environmental factors such as soil quality, weather conditions, and watering habits, which are not captured in the image dataset but may affect disease symptoms.
              """)
  
#Prediction Page
# elif(app_mode == "Disease Recognition"):
#   st.header("Disease Recognition")
#   test_image = st.file_uploader("Choose an Image:")
#   if(st.button("Show Image")):
#     st.image(test_image,use_container_width=True)
#   #Predict Button
#   if(st.button("Predict")):
#     st.snow()
#     st.write("Our Prediction")
#     result_index = model_prediction(test_image)
#     #Define Class
#     class_name = ['Apple__Apple_scab', 'Apple_Black_rot', 'Apple_Cedar_apple_rust', 'Apple__healthy',
#                     'Blueberry__healthy', 'Cherry(including_sour)_Powdery_mildew', 
#                     'Cherry_(including_sour)healthy', 'Corn(maize)_Cercospora_leaf_spot Gray_leaf_spot', 
#                     'Corn_(maize)Common_rust', 'Corn_(maize)Northern_Leaf_Blight', 'Corn(maize)_healthy', 
#                     'Grape__Black_rot', 'Grape_Esca(Black_Measles)', 'Grape__Leaf_blight(Isariopsis_Leaf_Spot)', 
#                     'Grape__healthy', 'Orange_Haunglongbing(Citrus_greening)', 'Peach___Bacterial_spot',
#                     'Peach__healthy', 'Pepper,_bell_Bacterial_spot', 'Pepper,_bell__healthy', 
#                     'Potato__Early_blight', 'Potato_Late_blight', 'Potato__healthy', 
#                     'Raspberry__healthy', 'Soybean_healthy', 'Squash__Powdery_mildew', 
#                     'Strawberry__Leaf_scorch', 'Strawberry_healthy', 'Tomato__Bacterial_spot', 
#                     'Tomato__Early_blight', 'Tomato_Late_blight', 'Tomato__Leaf_Mold', 
#                     'Tomato__Septoria_leaf_spot', 'Tomato__Spider_mites Two-spotted_spider_mite', 
#                     'Tomato__Target_Spot', 'Tomato_Tomato_Yellow_Leaf_Curl_Virus', 'Tomato__Tomato_mosaic_virus',
#                     'Tomato___healthy']
#     st.success("Model is Predicting it's a {}".format(class_name[result_index]))

elif(app_mode == "Disease Recognition"):
    st.markdown("""
<style>
                body{
                background-image: url('/static/disease_recognitionpage.jpeg');
                background-size: cover;
                backround-repeat: no-repeat;
                background-attachment: fixed;
                }
                </style>
                """,
                unsafe_allow_html=True
                )
    st.header("Disease Recognition")
    
    test_image = st.file_uploader("Choose an Image:")
    
    if test_image is not None:
        st.image(test_image, use_container_width=True)
        
        # Predict button
        if st.button("Predict"):
            st.snow()
            st.write("Our Prediction")
            result_index = model_prediction(test_image)
            
            # Define class names
            class_name = ['Apple__Apple_scab', 'Apple_Black_rot', 'Apple_Cedar_apple_rust', 'Apple__healthy',
                          'Blueberry__healthy', 'Cherry(including_sour)_Powdery_mildew', 
                          'Cherry_(including_sour)healthy', 'Corn(maize)_Cercospora_leaf_spot_Gray_leaf_spot', 
                          'Corn_(maize)Common_rust', 'Corn_(maize)Northern_Leaf_Blight', 'Corn(maize)_healthy', 
                          'Grape__Black_rot', 'Grape_Esca(Black_Measles)', 'Grape__Leaf_blight(Isariopsis_Leaf_Spot)', 
                          'Grape__healthy', 'Orange_Haunglongbing(Citrus_greening)', 'Peach___Bacterial_spot',
                          'Peach__healthy', 'Pepper,_bell_Bacterial_spot', 'Pepper,_bell__healthy', 
                          'Potato__Early_blight', 'Potato_Late_blight', 'Potato__healthy', 
                          'Raspberry__healthy', 'Soybean_healthy', 'Squash__Powdery_mildew:  is a fungal disease that affects squash plants (and other members of the gourd family, such as pumpkins and cucumbers). It is caused by several species of the fungus Podosphaera xanthii (formerly known as Sphaerotheca fuliginea), which primarily attacks the leaves, stems, fruits, roots, seeds, branch and flowers of squash plants. Symptoms of Squash Powdery Mildew:White, Powdery Growth, Leaf Yellowing, Leaf Curling,Reduced Plant Growth, Premature Leaf Drop', 
                          'Strawberry__Leaf_scorch', 'Strawberry_healthy', 'Tomato__Bacterial_spot', 
                          'Tomato__Early_blight', 'Tomato_Late_blight', 'Tomato__Leaf_Mold', 
                          'Tomato__Septoria_leaf_spot', 'Tomato__Spider_mites_Two-spotted_spider_mite', 
                          'Tomato__Target_Spot', 'Tomato_Tomato_Yellow_Leaf_Curl_Virus', 'Tomato__Tomato_mosaic_virus',
                          'Tomato___healthy']
                          
            st.success(f"Model is predicting: {class_name[result_index]}")












# import streamlit as st
# import tensorflow as tf
# import numpy as np
# from PIL import Image
# import time

# # Tensorflow Model Prediction
# def model_prediction(image_path):
#     model = tf.keras.models.load_model('trained_model.keras')
#     image = tf.keras.preprocessing.image.load_img(image_path, target_size=(128, 128))
#     input_arr = tf.keras.preprocessing.image.img_to_array(image) / 255.0  # Normalize
#     input_arr = np.expand_dims(input_arr, axis=0)
#     prediction = model.predict(input_arr)
#     result_index = np.argmax(prediction)
#     confidence = np.max(prediction) * 100  # Confidence in percentage
#     return result_index, confidence

# # Class Names and Remedies
# class_name = [
#     'Apple__Apple_scab', 'Apple_Black_rot', 'Apple_Cedar_apple_rust', 'Apple__healthy',
#     'Blueberry__healthy', 'Cherry(including_sour)_Powdery_mildew', 
#     'Cherry_(including_sour)healthy', 'Corn(maize)_Cercospora_leaf_spot Gray_leaf_spot', 
#     'Corn_(maize)Common_rust', 'Corn_(maize)Northern_Leaf_Blight', 'Corn(maize)_healthy', 
#     'Grape__Black_rot', 'Grape_Esca(Black_Measles)', 'Grape__Leaf_blight(Isariopsis_Leaf_Spot)', 
#     'Grape__healthy', 'Orange_Haunglongbing(Citrus_greening)', 'Peach___Bacterial_spot',
#     'Peach__healthy', 'Pepper,_bell_Bacterial_spot', 'Pepper,_bell__healthy', 
#     'Potato__Early_blight', 'Potato_Late_blight', 'Potato__healthy', 
#     'Raspberry__healthy', 'Soybean_healthy', 'Squash__Powdery_mildew', 
#     'Strawberry__Leaf_scorch', 'Strawberry_healthy', 'Tomato__Bacterial_spot', 
#     'Tomato__Early_blight', 'Tomato_Late_blight', 'Tomato__Leaf_Mold', 
#     'Tomato__Septoria_leaf_spot', 'Tomato__Spider_mites Two-spotted_spider_mite', 
#     'Tomato__Target_Spot', 'Tomato_Tomato_Yellow_Leaf_Curl_Virus', 'Tomato__Tomato_mosaic_virus',
#     'Tomato___healthy'
# ]

# remedies = {
#     'Apple___Apple_scab': ["Use fungicides", "Prune affected leaves"],
#     'Tomato___Early_blight': ["Apply copper-based fungicides", "Ensure crop rotation"],
#     # Add more remedies here as needed
# }

# disease_info = {
#     'Apple___Apple_scab': "A fungal disease causing dark blotches on apple leaves and fruit.",
#     'Tomato___Early_blight': "A common tomato disease caused by the fungus Alternaria solani.",
#     # Add more disease descriptions here
# }

# # Sidebar
# st.sidebar.title("Dashboard")
# app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Disease Recognition"])

# # Home Page
# if app_mode == "Home":
#     st.header("🌱 Plant Disease Recognition System")
#     image_path = "home_page.jpeg"  # Replace with an actual image path
#     st.image(image_path, use_column_width=True)
#     st.markdown("""
#     Welcome to the Plant Disease Recognition System! 🌿🔍

#     *How It Works:*
#     1. Upload an image of a plant.
#     2. Our system analyzes the image to detect diseases.
#     3. View the results and suggested remedies.

#     Protect your crops and ensure healthier harvests!
#     """)

# # About Page
# elif app_mode == "About":
#     st.header("About")
#     st.markdown("""
#     *Project Overview:*
#     This system uses deep learning models to detect plant diseases. It's trained on a large dataset with over 70,000 images, enabling accurate recognition of 38 plant diseases.

#     *Goal:*
#     To assist farmers and researchers in identifying plant diseases and taking preventive actions early.
#     """)

# # Disease Recognition Page
# elif app_mode == "Disease Recognition":
#     st.header("Disease Recognition")
#     test_image = st.file_uploader("Choose an Image:", type=["jpg", "jpeg", "png"])
#     if test_image:
#         st.image(test_image, caption="Uploaded Image", use_column_width=True)
    
#     # Predict Button
#     if st.button("Predict"):
#         if test_image:
#             with st.spinner("Analyzing the image..."):
#                 result_index, confidence = model_prediction(test_image)
#             st.success(f"Prediction: {class_name[result_index]} with {confidence:.2f}% confidence")
            
#             # Disease Information
#             if class_name[result_index] in disease_info:
#                 st.markdown(f"*Disease Information:* {disease_info[class_name[result_index]]}")
            
#             # Remedies
#             if class_name[result_index] in remedies:
#                 st.markdown("### Suggested Remedies:")
#                 for remedy in remedies[class_name[result_index]]:
#                     st.write(f"- {remedy}")
#             else:
#                 st.write("No specific remedies found.")
#         else:
#             st.warning("Please upload an image before predicting.")

# # Footer
# st.markdown("---")
# st.write("🌍 Powered by AI | Plant Disease Recognition System © 2024")