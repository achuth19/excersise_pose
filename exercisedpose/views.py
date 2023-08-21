from django.http import HttpResponse, HttpResponseRedirect
from django.shortcuts import render
import base64
from django.core.files.storage import FileSystemStorage
from io import BytesIO
import io
from PIL import Image
# import streamlit as st
from PIL import Image
import cv2
import numpy as np
# import math
import cv2
# import numpy as np
from time import time
import mediapipe as mp
import pandas as pd
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
import sklearn.preprocessing
from sklearn.preprocessing import normalize
from pandas.core.common import random_state
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
# from IPython.display import HTML
# import modell
def predict_image(image):
    
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder, MinMaxScaler
    from sklearn.utils.class_weight import compute_class_weight
    from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split
    from sklearn.linear_model import LogisticRegression, SGDClassifier
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import MinMaxScaler
    import sklearn.preprocessing
    from sklearn.preprocessing import normalize
    from pandas.core.common import random_state
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    import numpy as np

    import math
    import cv2
    import numpy as np
    from time import time
    import mediapipe as mp
    import matplotlib.pyplot as plt
    from IPython.display import HTML



    file_1 = pd.read_csv(r'C:\Users\achut\OneDrive\Documents\posedetection\exercisedpose\modules\3d_distances.csv')
    file_2 = pd.read_csv(r'C:\Users\achut\OneDrive\Documents\posedetection\exercisedpose\modules\angles.csv')
    file_3 = pd.read_csv(r'C:\Users\achut\OneDrive\Documents\posedetection\exercisedpose\modules\labels.csv')
    file_4 = pd.read_csv(r'C:\Users\achut\OneDrive\Documents\posedetection\exercisedpose\modules\landmarks.csv')
    file_5 = pd.read_csv(r'C:\Users\achut\OneDrive\Documents\posedetection\exercisedpose\modules\xyz_distances.csv')

    merged_data = pd.merge(file_4,file_5,on='pose_id')
    merged_data = pd.merge(merged_data,file_1,on='pose_id')
    merged_data = pd.merge(merged_data,file_2,on='pose_id')

    merged_data = pd.merge(merged_data,file_3,on='pose_id')

    merged_data = merged_data.drop('pose_id',axis=1)
    merged_data = merged_data.drop('right_knee_mid_hip_left_knee',axis=1)

    x= merged_data.drop(['pose'], axis='columns')
    y = merged_data['pose']

    encoder = LabelEncoder()
    y = merged_data['pose']
    y = encoder.fit_transform(y)
    
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y), y=y)
    class_weights_dict = {}
    for idx, k in enumerate(class_weights):
        class_weights_dict[idx] = k 

    scaler = MinMaxScaler()
    x = scaler.fit_transform(x)

    X_train,X_test,Y_train,Y_test = train_test_split(x,y,test_size=0.2,random_state=42)

    classifier = RandomForestClassifier(random_state=42)

    classifier.fit(X_train,Y_train)

    # y_pred = classifier.predict(X_test)
    # accuracy_rf = accuracy_score(Y_test,y_pred)   
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.3, model_complexity=2)
    mp_drawing = mp.solutions.drawing_utils

    # sample_img_array  = cv2.imread(r"OIP.jpeg")
    # sample_img_array = np.array(image)


    points = mp_pose.PoseLandmark   
    data = []

    for p in points:
            x = str(p)[13:]
            data.append(x + "_x")
            data.append(x + "_y")
            data.append(x + "_z")
            data.append(x + "_vis")
    data = pd.DataFrame(columns = data) 

    count = 0
    temp = []
    img_array = np.array(image)

    # img_array = cv2.imread(r"OIP.jpeg")

        # imageWidth, imageHeight = img_array.shape[:2]

    img_arrayRGB = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        # results = pose.process(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB))

    blackie = np.zeros(img_array.shape) # Blank image

    results = pose.process(img_arrayRGB)

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(blackie, results.pose_landmarks, mp_pose.POSE_CONNECTIONS) # draw landmarks on blackie

        landmarks = results.pose_landmarks.landmark


        dict = {}
        for point, j in zip(points, landmarks):
            temp = temp + [j.x, j.y, j.z]
    # temp.extend([landmark.x, landmark.y, landmark.z])
            key = point.name
            value = [temp[count], temp[count+1], temp[count+2]]
            dict[key] = value
            count = count + 1

    #xyz_distances
    c1 = dict['LEFT_WRIST'][0] - dict['LEFT_SHOULDER'][0]
    c2 = dict['LEFT_WRIST'][1] - dict['LEFT_SHOULDER'][1]
    c3 =  dict['LEFT_WRIST'][2] - dict['LEFT_SHOULDER'][2]

    c4 = dict['RIGHT_WRIST'][0] - dict['RIGHT_SHOULDER'][0]
    c5 = dict['RIGHT_WRIST'][1] - dict['RIGHT_SHOULDER'][1]
    c6 = dict['RIGHT_WRIST'][2] - dict['RIGHT_SHOULDER'][2]

    c7	= dict['LEFT_ANKLE'][0] - dict['LEFT_HIP'][0]
    c8	= dict['LEFT_ANKLE'][1] - dict['LEFT_HIP'][1]
    c9	= dict['LEFT_ANKLE'][2] - dict['LEFT_HIP'][2]

    c10	=  dict['RIGHT_ANKLE'][0] - dict['RIGHT_HIP'][0]
    c11	= dict['RIGHT_ANKLE'][1] - dict['RIGHT_HIP'][1]
    c12 = dict['RIGHT_ANKLE'][2] - dict['RIGHT_HIP'][2]

    c13	= dict['LEFT_WRIST'][0] - dict['LEFT_HIP'][0]
    c14	=  dict['LEFT_WRIST'][1] - dict['LEFT_HIP'][1]
    c15=  dict['LEFT_WRIST'][2] - dict['LEFT_HIP'][2]

    c16	= dict['RIGHT_WRIST'][0] - dict['RIGHT_HIP'][0]
    c17	= dict['RIGHT_WRIST'][1] - dict['RIGHT_HIP'][1]
    c18= dict['RIGHT_WRIST'][2] - dict['RIGHT_HIP'][2]

    c19	= dict['LEFT_ANKLE'][0] - dict['LEFT_SHOULDER'][0]
    c20 = dict['LEFT_ANKLE'][1] - dict['LEFT_SHOULDER'][1]
    c21 = dict['LEFT_ANKLE'][2] - dict['LEFT_SHOULDER'][2]

    c22 = dict['RIGHT_ANKLE'][0] - dict['RIGHT_SHOULDER'][0]
    c23 = dict['RIGHT_ANKLE'][1] - dict['RIGHT_SHOULDER'][1]
    c24 = dict['RIGHT_ANKLE'][2] - dict['RIGHT_SHOULDER'][2]

    c25 = dict['RIGHT_WRIST'][0] - dict['LEFT_HIP'][0]
    c26 = dict['RIGHT_WRIST'][1] - dict['LEFT_HIP'][1]
    c27 = dict['RIGHT_WRIST'][2] - dict['LEFT_HIP'][2]

    c28 = dict['LEFT_WRIST'][0] - dict['RIGHT_HIP'][0]
    c29	= dict['LEFT_WRIST'][1] - dict['RIGHT_HIP'][1]
    c30	= dict['LEFT_WRIST'][2] - dict['RIGHT_HIP'][2]

    c31 = dict['RIGHT_ELBOW'][0] - dict['LEFT_ELBOW'][0]
    c32 = dict['RIGHT_ELBOW'][1] - dict['LEFT_ELBOW'][1]
    c33 = dict['RIGHT_ELBOW'][2] - dict['LEFT_ELBOW'][2]

    c34 = dict['RIGHT_KNEE'][0] - dict['LEFT_KNEE'][0]
    c35	= dict['RIGHT_KNEE'][1] - dict['LEFT_KNEE'][1]
    c36	= dict['RIGHT_KNEE'][2] - dict['LEFT_KNEE'][2]

    c37 = dict['RIGHT_WRIST'][0] - dict['LEFT_WRIST'][0]
    c38 = dict['RIGHT_WRIST'][1] - dict['LEFT_WRIST'][1]
    c39 = dict['RIGHT_WRIST'][2] - dict['LEFT_WRIST'][2]

    c40 = dict['RIGHT_ANKLE'][0] - dict['LEFT_ANKLE'][0]
    c41	= dict['RIGHT_ANKLE'][1] - dict['LEFT_ANKLE'][1]
    c42= dict['RIGHT_ANKLE'][2] - dict['LEFT_ANKLE'][2]

    c43 = dict['LEFT_HIP'][0]-(dict['LEFT_WRIST'][0] + dict['LEFT_ANKLE'][0])/2
    c44 = dict['LEFT_HIP'][1]-(dict['LEFT_WRIST'][1] + dict['LEFT_ANKLE'][1])/2
    c45 = dict['LEFT_HIP'][2]-(dict['LEFT_WRIST'][2] + dict['LEFT_ANKLE'][2])/2

    c46 = dict['RIGHT_HIP'][0]-(dict['RIGHT_WRIST'][0] + dict['RIGHT_ANKLE'][0])/2
    c47	= dict['RIGHT_HIP'][1]-(dict['RIGHT_WRIST'][1] + dict['RIGHT_ANKLE'][1])/2
    c48 = dict['RIGHT_HIP'][2]-(dict['RIGHT_WRIST'][2] + dict['RIGHT_ANKLE'][2])/2



    #3d_distances

    c49 = np.sqrt((dict['LEFT_WRIST'][0] - dict['LEFT_SHOULDER'][0])**2 + (dict['LEFT_WRIST'][1] - dict['LEFT_SHOULDER'][1])**2 + (dict['LEFT_WRIST'][2] - dict['LEFT_SHOULDER'][2])**2)
    c50 = np.sqrt((dict['RIGHT_WRIST'][0] - dict['RIGHT_SHOULDER'][0])**2 + (dict['RIGHT_WRIST'][1] - dict['RIGHT_SHOULDER'][1])**2 + (dict['RIGHT_WRIST'][2] - dict['RIGHT_SHOULDER'][2])**2)
    c51= np.sqrt((dict['LEFT_ANKLE'][0] - dict['LEFT_HIP'][0])**2 + (dict['LEFT_ANKLE'][1] - dict['LEFT_HIP'][1])**2 + (dict['LEFT_ANKLE'][2] - dict['LEFT_HIP'][2])**2)
    c52= np.sqrt((dict['RIGHT_ANKLE'][0] - dict['RIGHT_HIP'][0])**2 + (dict['RIGHT_ANKLE'][1] - dict['RIGHT_HIP'][1])**2 + (dict['RIGHT_ANKLE'][2] - dict['RIGHT_HIP'][2])**2)
    c53	= np.sqrt((dict['LEFT_WRIST'][0] - dict['LEFT_HIP'][0])**2 + (dict['LEFT_WRIST'][1] - dict['LEFT_HIP'][1])**2 + (dict['LEFT_WRIST'][2] - dict['LEFT_HIP'][2])**2)
    c54	= np.sqrt((dict['RIGHT_WRIST'][0] - dict['RIGHT_HIP'][0])**2 + (dict['RIGHT_WRIST'][1] - dict['RIGHT_HIP'][1])**2 + (dict['RIGHT_WRIST'][2] - dict['RIGHT_HIP'][2])**2)
    c55 = np.sqrt((dict['LEFT_ANKLE'][0] - dict['LEFT_SHOULDER'][0])**2 + (dict['LEFT_ANKLE'][1] - dict['LEFT_SHOULDER'][1])**2 + (dict['LEFT_ANKLE'][2] - dict['LEFT_SHOULDER'][2])**2)
    c56 = np.sqrt((dict['RIGHT_ANKLE'][0] - dict['RIGHT_SHOULDER'][0])**2 + (dict['RIGHT_ANKLE'][1] - dict['RIGHT_SHOULDER'][1])**2 + (dict['RIGHT_ANKLE'][2] - dict['RIGHT_SHOULDER'][2])**2)
    c57 = np.sqrt((dict['RIGHT_WRIST'][0] - dict['LEFT_HIP'][0])**2 + (dict['RIGHT_WRIST'][1] - dict['LEFT_HIP'][1])**2 + (dict['RIGHT_WRIST'][2] - dict['LEFT_HIP'][2])**2)
    c58 = np.sqrt((dict['LEFT_WRIST'][0] - dict['RIGHT_HIP'][0])**2 + (dict['LEFT_WRIST'][1] - dict['RIGHT_HIP'][1])**2 + (dict['LEFT_WRIST'][2] - dict['RIGHT_HIP'][2])**2)
    c59= np.sqrt((dict['RIGHT_ELBOW'][0] - dict['LEFT_ELBOW'][0])**2 + (dict['RIGHT_ELBOW'][1] - dict['LEFT_ELBOW'][1])**2 + (dict['RIGHT_ELBOW'][2] - dict['LEFT_ELBOW'][2])**2)
    c60 = np.sqrt((dict['RIGHT_KNEE'][0] - dict['LEFT_KNEE'][0])**2 + (dict['RIGHT_KNEE'][1] - dict['LEFT_KNEE'][1])**2 + (dict['RIGHT_KNEE'][2] - dict['LEFT_KNEE'][2])**2)
    c61 = np.sqrt((dict['RIGHT_WRIST'][0] - dict['LEFT_WRIST'][0])**2 + (dict['RIGHT_WRIST'][1] - dict['LEFT_WRIST'][1])**2 + (dict['RIGHT_WRIST'][2] - dict['LEFT_WRIST'][2])**2)
    c62 = np.sqrt((dict['RIGHT_ANKLE'][0] - dict['LEFT_ANKLE'][0])**2 + (dict['RIGHT_ANKLE'][1] - dict['LEFT_ANKLE'][1])**2 + (dict['RIGHT_ANKLE'][2] - dict['LEFT_ANKLE'][2])**2)

    x_avg = (dict['LEFT_WRIST'][0] + dict['LEFT_ANKLE'][0])/2
    y_avg = (dict['LEFT_WRIST'][1] + dict['LEFT_ANKLE'][1])/2
    z_avg = (dict['LEFT_WRIST'][2] + dict['LEFT_ANKLE'][2])/2

    c63 = np.sqrt((x_avg - dict['LEFT_HIP'][0])**2 + (y_avg - dict['LEFT_HIP'][1])**2 + (z_avg - dict['LEFT_HIP'][2])**2)



    x_avg = (dict['RIGHT_WRIST'][0] + dict['LEFT_ANKLE'][0])/2
    y_avg = (dict['RIGHT_WRIST'][1] + dict['LEFT_ANKLE'][1])/2
    z_avg = (dict['RIGHT_WRIST'][2] + dict['LEFT_ANKLE'][2])/2
    c64 = np.sqrt((x_avg - dict['RIGHT_HIP'][0])**2 + (y_avg - dict['RIGHT_HIP'][1])**2 + (z_avg - dict['RIGHT_HIP'][2])**2)

    #angles
    import math
    import numpy as np

    Ax = dict['RIGHT_SHOULDER'][0] - dict['RIGHT_ELBOW'][0]
    Ay = dict['RIGHT_SHOULDER'][1] - dict['RIGHT_ELBOW'][1]
    Az = dict['RIGHT_SHOULDER'][2] - dict['RIGHT_ELBOW'][2]
    Bx = dict['RIGHT_SHOULDER'][0] - dict['RIGHT_HIP'][0]
    By = dict['RIGHT_SHOULDER'][1] - dict['RIGHT_HIP'][1]
    Bz = dict['RIGHT_SHOULDER'][2] - dict['RIGHT_HIP'][2]
    AdotB = (Ax * Bx) + (Ay * By) + (Az * Bz)
    modA = np.sqrt(Ax**2 + Ay**2 + Az**2)
    modB = np.sqrt(Bx**2 + By**2 + Bz**2)
    cos_angle = AdotB / (modA * modB)
    angle = np.arccos(cos_angle)
    c65= np.degrees(angle)

    Ax = dict['LEFT_SHOULDER'][0] - dict['LEFT_ELBOW'][0]
    Ay = dict['LEFT_SHOULDER'][1] - dict['LEFT_ELBOW'][1]
    Az = dict['LEFT_SHOULDER'][2] - dict['LEFT_ELBOW'][2]
    Bx = dict['LEFT_SHOULDER'][0] - dict['LEFT_HIP'][0]
    By = dict['LEFT_SHOULDER'][1] - dict['LEFT_HIP'][1]
    Bz = dict['LEFT_SHOULDER'][2] - dict['LEFT_HIP'][2]
    AdotB = (Ax * Bx) + (Ay * By) + (Az * Bz)
    modA = np.sqrt(Ax**2 + Ay**2 + Az**2)
    modB = np.sqrt(Bx**2 + By**2 + Bz**2)
    cos_angle = AdotB / (modA * modB)
    angle = np.arccos(cos_angle)
    c66	= np.degrees(angle)

    Ax = (dict['LEFT_HIP'][0] + dict['RIGHT_HIP'][0])/2 - dict['RIGHT_KNEE'][0]
    Ay =(dict['LEFT_HIP'][1] + dict['RIGHT_HIP'][1])/2 - dict['RIGHT_KNEE'][1]
    Az = (dict['LEFT_HIP'][2] + dict['RIGHT_HIP'][2])/2 - dict['RIGHT_KNEE'][2]
    Bx = (dict['LEFT_HIP'][0] + dict['RIGHT_HIP'][0])/2 - dict['LEFT_KNEE'][0]
    By = (dict['LEFT_HIP'][1] + dict['RIGHT_HIP'][1])/2 - dict['LEFT_KNEE'][1]
    Bz = (dict['LEFT_HIP'][2] + dict['RIGHT_HIP'][2])/2 - dict['LEFT_KNEE'][2]
    AdotB = (Ax * Bx) + (Ay * By) + (Az * Bz)
    modA = np.sqrt(Ax**2 + Ay**2 + Az**2)
    modB = np.sqrt(Bx**2 + By**2 + Bz**2)
    cos_angle = AdotB / (modA * modB)
    angle = np.arccos(cos_angle)
    c67 = np.degrees(angle)
    # print(right_knee_mid_hip_left_knee )


    Ax = dict['RIGHT_KNEE'][0] - dict['RIGHT_HIP'][0]
    Ay = dict['RIGHT_KNEE'][1] - dict['RIGHT_HIP'][1]
    Az = dict['RIGHT_KNEE'][2] - dict['RIGHT_HIP'][2]
    Bx = dict['RIGHT_KNEE'][0] - dict['RIGHT_ANKLE'][0]
    By = dict['RIGHT_KNEE'][1] - dict['RIGHT_ANKLE'][1]
    Bz = dict['RIGHT_KNEE'][2] - dict['RIGHT_ANKLE'][2]
    AdotB = (Ax * Bx) + (Ay * By) + (Az * Bz)
    modA = np.sqrt(Ax**2 + Ay**2 + Az**2)
    modB = np.sqrt(Bx**2 + By**2 + Bz**2)
    cos_angle = AdotB / (modA * modB)
    angle = np.arccos(cos_angle)
    c68 = np.degrees(angle)

    Ax = dict['LEFT_KNEE'][0] - dict['LEFT_HIP'][0]
    Ay = dict['LEFT_KNEE'][1] - dict['LEFT_HIP'][1]
    Az = dict['LEFT_KNEE'][2] - dict['LEFT_HIP'][2]
    Bx = dict['LEFT_KNEE'][0] - dict['LEFT_ANKLE'][0]
    By = dict['LEFT_KNEE'][1] - dict['LEFT_ANKLE'][1]
    Bz = dict['LEFT_KNEE'][2] - dict['LEFT_ANKLE'][2]
    AdotB = (Ax * Bx) + (Ay * By) + (Az * Bz)
    modA = np.sqrt(Ax**2 + Ay**2 + Az**2)
    modB = np.sqrt(Bx**2 + By**2 + Bz**2)
    cos_angle = AdotB / (modA * modB)
    angle = np.arccos(cos_angle)
    c69=  np.degrees(angle)


    Ax = dict['RIGHT_ELBOW'][0] - dict['RIGHT_WRIST'][0]
    Ay = dict['RIGHT_ELBOW'][1] - dict['RIGHT_WRIST'][1]
    Az = dict['RIGHT_ELBOW'][2] - dict['RIGHT_WRIST'][2]
    Bx = dict['RIGHT_ELBOW'][0] - dict['RIGHT_SHOULDER'][0]
    By = dict['RIGHT_ELBOW'][1] - dict['RIGHT_SHOULDER'][1]
    Bz = dict['RIGHT_ELBOW'][2] - dict['RIGHT_SHOULDER'][2]
    AdotB = (Ax * Bx) + (Ay * By) + (Az * Bz)
    modA = np.sqrt(Ax**2 + Ay**2 + Az**2)
    modB = np.sqrt(Bx**2 + By**2 + Bz**2)
    cos_angle = AdotB / (modA * modB)
    angle = np.arccos(cos_angle)
    c70 = np.degrees(angle)

    Ax = dict['LEFT_ELBOW'][0] - dict['LEFT_WRIST'][0]
    Ay = dict['LEFT_ELBOW'][1] - dict['LEFT_WRIST'][1]
    Az = dict['LEFT_ELBOW'][2] - dict['LEFT_WRIST'][2]
    Bx = dict['LEFT_ELBOW'][0] - dict['LEFT_SHOULDER'][0]
    By = dict['LEFT_ELBOW'][1] - dict['LEFT_SHOULDER'][1]
    Bz = dict['LEFT_ELBOW'][2] - dict['LEFT_SHOULDER'][2]
    AdotB = (Ax * Bx) + (Ay * By) + (Az * Bz)
    modA = np.sqrt(Ax**2 + Ay**2 + Az**2)
    modB = np.sqrt(Bx**2 + By**2 + Bz**2)
    cos_angle = AdotB / (modA * modB)
    angle = np.arccos(cos_angle)
    c71 = np.degrees(angle)

    for i in range(1,72):
        if(i!=67):
            temp.append(locals()['c' + str(i)])

    temp = np.array(temp)         
    temp = temp.reshape(-1, 169)   

    y_pred = classifier.predict(temp)
    y_pred_class_names = encoder.inverse_transform(y_pred)
    return y_pred_class_names

def index(request):
    return render(request,"exercisedpose/base.html")

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.3, model_complexity=2)
mp_drawing = mp.solutions.drawing_utils

# Function to process the image and get the output
def process_image(image):
    print("Func")
    # Process the image using your .ipynb code here
    # For this example, we'll just convert it to grayscale using OpenCV
    sample_img = np.array(image)

    results = pose.process(cv2.cvtColor(sample_img, cv2.COLOR_BGR2RGB))

    img_copy = sample_img.copy()
    height, width, _ = img_copy.shape
    landmarks = []
    # Check if any landmarks are found.
    if results.pose_landmarks:

    # Draw Pose landmarks on the sample image.
        mp_drawing.draw_landmarks(image=img_copy, landmark_list=results.pose_landmarks, connections=mp_pose.POSE_CONNECTIONS)
        for landmark in results.pose_landmarks.landmark:

            # Append the landmark into the list.
                landmarks.append((int(landmark.x * width), int(landmark.y * height),
                                  (landmark.z * width)))
    # Specify a size of the figure.
        fig = plt.figure(figsize = [10, 10])


    return img_copy

def Blackie_image(image):
    img = np.array(image)
    # img = cv2.imread(r"OIP.jpeg")

        # imageWidth, imageHeight = img.shape[:2]

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # results = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    blackie = np.zeros(img.shape) # Blank image

    results = pose.process(imgRGB)

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(blackie, results.pose_landmarks, mp_pose.POSE_CONNECTIONS) # draw landmarks on blackie

    landmarks = results.pose_landmarks.landmark

    blackie_normalized = blackie.astype(float) / 255.0

    return blackie_normalized

# def record():
#     # Initialize MediaPipe Pose model
#     mp_pose = mp.solutions.pose
#     pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

#     points = mp_pose.PoseLandmark       

#     data = []

#     for p in points:
#             x = str(p)[13:]
#             data.append(x + "_x")
#             data.append(x + "_y")
#             data.append(x + "_z")
#             data.append(x + "_vis")
#     data = pd.DataFrame(columns = data) 


#     # Function to capture image and store pose landmarks
#     def capture_image_and_landmarks():
#         cap = cv2.VideoCapture(0)  # Use the appropriate camera index if you have multiple cameras

#         start_time = time.time()
#         last_frame = None

#         while time.time() - start_time <= 10:
#             ret, frame = cap.read()

#             if not ret:
#                 print("Failed to capture frame.")
#                 break

#             imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#             results = pose.process(imgRGB)

#             if results.pose_landmarks:
#                 mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

#             last_frame = frame  # Update the last_frame with the latest frame

#             cv2.imshow("Camera", frame)
#             cv2.waitKey(1)

#         frame_bytes = cv2.imencode(".jpg", frame)[1].tobytes()
#         st.image(frame_bytes, channels="BGR", use_column_width=True)

#         cap.release()
#         cv2.destroyAllWindows()

#         return last_frame

#     # Call the function to capture images for 10 seconds and get the last frame
#     last_frame = capture_image_and_landmarks()

#     # Now you have the last_frame, and you can do further processing or saving if needed.
#     if last_frame is not None:   
#         return last_frame                        # Display the image
#     else:
#         print("No frames captured.")
def image_to_base64(image_array):
    if len(image_array.shape) == 3 and image_array.shape[2] == 3:
        if image_array.dtype == np.float64:
            # Convert the floating-point array to uint8 if needed
            image_array = (image_array * 255).astype(np.uint8)
        elif image_array.dtype != np.uint8:
            raise ValueError("Unsupported data type for image conversion")
        
        # Convert the NumPy array to a PIL image
        image_pil = Image.fromarray(image_array)

        # Create an in-memory binary stream
        buffered = io.BytesIO()

        # Save the PIL image to the binary stream in JPEG format
        image_pil.save(buffered, format="JPEG")

        # Get the binary data from the stream
        image_binary = buffered.getvalue()

        # Convert the binary data to a base64-encoded string
        image_base64 = base64.b64encode(image_binary).decode()

        return image_base64
    else:
        raise ValueError("Invalid image shape")



def main(request):
    # st.title("Image and Video Analysis")

    # # Add a sidebar with options
    # option = st.sidebar.selectbox("Select Option", ("Upload Image", "Record Video"))

    # if option == "Upload Image":
    #     st.subheader("Upload an Image")
    if request.method == 'POST' and request.FILES['image']:
        uploaded_image = request.FILES['image']
        if uploaded_image is not None:
            # Display the uploaded image
           
            image = Image.open(uploaded_image)
            output_image = process_image(image)
            out_image = Blackie_image(image)
            final = predict_image(image)
            # print("Output Image:")
            # print("Shape:", output_image.shape)
            # print("Data Type:", output_image.dtype)
            # print("Min Value:", output_image.min())
            # print("Max Value:", output_image.max())

            # print("\nOut Image:")
            # print("Shape:", out_image.shape)
            # print("Data Type:", out_image.dtype)
            # print("Min Value:", out_image.min())
            # print("Max Value:", out_image.max())

            # print("\nFinal Image:")
            # print("Shape:", final.shape)
            # print("Data Type:", final.dtype)
            # print("Min Value:", final.min())
            # print("Max Value:", final.max())
            output_image_base64 = image_to_base64(output_image)
            out_image_base64 = image_to_base64(out_image)
            context = {
                'output_image_base64': output_image_base64,
                'out_image_base64': out_image_base64,
                'final_value': final
       
            }
            return render(request, 'exercisedpose/upload.html', context)