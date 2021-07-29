from logging import exception
from threading import Condition
from numpy.lib.type_check import imag
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import mediapipe as mp
import pandas as pd

import face_recognition
from streamlit.proto.Empty_pb2 import Empty

chris = face_recognition.load_image_file("chris.jpeg")
chris_encodings = face_recognition.face_encodings(chris)[0]

robert = face_recognition.load_image_file("robert.jpeg")
robert_encodings = face_recognition.face_encodings(robert)[0]

aai = face_recognition.load_image_file("aai.jpeg")
aai_encodings = face_recognition.face_encodings(aai)[0]

papa = face_recognition.load_image_file("papa.jpeg")
papa_encodings = face_recognition.face_encodings(papa)[0]

piyush = face_recognition.load_image_file("Piyush.jpeg")
piyush_encodings = face_recognition.face_encodings(piyush)[0]

varun = face_recognition.load_image_file("varun.jpeg")
varun_encodings = face_recognition.face_encodings(varun)[0]

known_face_encodings = [
    chris_encodings,
    robert_encodings,
    aai_encodings,
    papa_encodings,
    piyush_encodings,
    varun_encodings
]

known_face_names = [
    "Chris",
    "Robert",
    "Aai",
    "Papa",
    "Piyush",
    "Varun"
]

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

model_face_mesh = mp_face_mesh.FaceMesh()

mp_selfie_segmentation = mp.solutions.selfie_segmentation

model = mp_selfie_segmentation.SelfieSegmentation(model_selection = 1)


st.title("OpenCV Operations")
st.subheader("Image Processing")

st.write("This application performs various operations with OpenCV")


    
st.sidebar.title("OpenCV")
select_radio = st.sidebar.radio(
    "For doing OpenCV operations please choose one : ",
    ("Upload Image","Use Camera"))


image_file_path = None
if select_radio == "Upload Image":
    image_file_path = st.sidebar.file_uploader("Upload Image")  
    if image_file_path is not None:
        image = np.array(Image.open(image_file_path))
        sidebar_image = cv2.resize(image, (0, 0), fx=0.15, fy=0.15)
        st.sidebar.image(sidebar_image)
        add_selectbox = st.sidebar.selectbox(
            "What operations would you like to perform?",
            ("Select", "Change Color", "Meshing", "Image Background Change","Face Recognition")
        )
        if add_selectbox == "Select":
            st.write("This application is a demo for streamlit.")


        elif add_selectbox == "Change Color":
            add_selectbox1 = st.sidebar.radio(
                "Which color do you want to set for your image?",
                ("About","GrayScale", "Blue", "Green", "Red")
            )
            if add_selectbox1 == "About":
                st.write("You will be able to change color of an image.")
            elif add_selectbox1 == "GrayScale":
                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                st.image(gray_image)  
                user_input = st.text_input("File name to save",value="xyz.jpg")
                save = st.button("Download Image", help="Download")
                if save:
                    output_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    cv2.imwrite(user_input,output_image)
                    st.success("File Saved successfully!")  

            elif add_selectbox1 == "Blue":
                zeros = np.zeros(image.shape[:2], dtype="uint8")
                r, g, b = cv2.split(image)
                
                blue_image = cv2.merge([zeros, zeros, b])
                st.image(blue_image)
                user_input = st.text_input("File name to save",value="xyz.jpg")
                save = st.button("Download Image", help="Download")
                if save:
                    output_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    cv2.imwrite(user_input,output_image)
                    st.success("File Saved successfully!")
            elif add_selectbox1 == "Green":
                    zeros = np.zeros(image.shape[:2], dtype="uint8")
                    r, g, b = cv2.split(image)
                    green_image = cv2.merge([zeros, g, zeros])
                    st.image(green_image)
                    user_input = st.text_input("File name to save",value="xyz.jpg")
                    save = st.button("Download Image", help="Download")
                    if save:
                        output_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        cv2.imwrite(user_input,output_image)
                        st.success("File Saved successfully!")
            elif add_selectbox1 == "Red":
                    zeros = np.zeros(image.shape[:2], dtype="uint8")
                    r, g, b = cv2.split(image)
                    blue_image = cv2.merge([r, zeros, zeros])
                    st.image(blue_image)
                    user_input = st.text_input("File name to save",value="xyz.jpg")
                    save = st.button("Download Image", help="Download")
                    if save:
                        output_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        cv2.imwrite(user_input,output_image)
                        st.success("File Saved successfully!")

        elif add_selectbox == "Meshing":
                results = model_face_mesh.process(image)
                for face_landmarks in results.multi_face_landmarks:
                    mp_drawing.draw_landmarks(
                        image, face_landmarks
                    )
                st.image(image)
                user_input = st.text_input("File name to save",value="xyz.jpg")
                save = st.button("Download Image", help="Download")
                if save:
                    output_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    cv2.imwrite(user_input,output_image)
                    st.success("File Saved successfully!")
        elif add_selectbox == "Image Background Change":
            add_selectbox2 =  st.sidebar.radio(
                "Which background color you want to set for your image?",
                ("About","GrayScale", "Blue", "Green", "Red","White"))

            if add_selectbox2 == "About":
                st.write("Image with selected background color will be displayed here.")
            
            elif add_selectbox2 == "GrayScale":
                results = model.process(image)
                condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
                bg_image = np.zeros(image.shape, dtype=np.uint8)
                bg_image[:] = (128, 128, 128)
                bg_image = cv2.resize(bg_image, (image.shape[1], image.shape[0]))
                output_image = np.where(condition, image, bg_image)
                st.image(output_image)
                user_input = st.text_input("File name to save",value="xyz.jpg")
                save = st.button("Download Image", help="Download")
                if save:
                    output_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
                    cv2.imwrite(user_input,output_image)
                    st.success("Saved File")
            elif add_selectbox2 == "Blue":
                results = model.process(image)
                condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
                bg_image = np.zeros(image.shape, dtype=np.uint8)
                bg_image[:] = (0, 0, 255)
                bg_image = cv2.resize(bg_image, (image.shape[1], image.shape[0]))
                output_image = np.where(condition, image, bg_image)
                st.image(output_image)
                user_input = st.text_input("File name to save",value="xyz.jpg")
                save = st.button("Download Image", help="Download")
                if save:
                    output_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
                    cv2.imwrite(user_input,output_image)
                    st.success("Saved File")
            elif add_selectbox2 == "Green":
                results = model.process(image)
                condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
                bg_image = np.zeros(image.shape, dtype=np.uint8)
                bg_image[:] = (0, 255, 0)
                bg_image = cv2.resize(bg_image, (image.shape[1], image.shape[0]))
                output_image = np.where(condition, image, bg_image)
                st.image(output_image)
                user_input = st.text_input("File name to save",value="xyz.jpg")
                save = st.button("Download Image", help="Download")
                if save:
                    output_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
                    cv2.imwrite(user_input,output_image)
                    st.success("Saved File")
            elif add_selectbox2 == "Red":
                results = model.process(image)
                condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
                bg_image = np.zeros(image.shape, dtype=np.uint8)
                bg_image[:] = (255, 0, 0)
                bg_image = cv2.resize(bg_image, (image.shape[1], image.shape[0]))
                output_image = np.where(condition, image, bg_image)
                st.image(output_image)
                user_input = st.text_input("File name to save",value="xyz.jpg")
                save = st.button("Download Image", help="Download")
                if save:
                    output_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
                    cv2.imwrite(user_input,output_image)
                    st.success("Saved File")
            elif add_selectbox2 == "White":
                results = model.process(image)
                condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
                bg_image = np.zeros(image.shape, dtype=np.uint8)
                bg_image[:] = (255, 255, 255)
                bg_image = cv2.resize(bg_image, (image.shape[1], image.shape[0]))
                output_image = np.where(condition, image, bg_image)
                st.image(output_image)
                user_input = st.text_input("File name to save",value="xyz.jpg")
                save = st.button("Download Image", help="Download")
                if save:
                    output_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
                    cv2.imwrite(user_input,output_image)
                    st.success("Saved File")    
        elif add_selectbox == "Face Recognition":
            frame = image
            small_frame = cv2.resize(frame, (0, 0), fx=1/4, fy=1/4)
            rgb_small_frame = small_frame
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            face_names = []
            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index] 
                face_names.append(name)      
            for(top, right, bottom, left), name in zip(face_locations, face_names):
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4
                cv2.rectangle(frame, (left, top), (right, bottom), (255,0,0), 2)
                cv2.rectangle(frame, (left, bottom-35), (right, bottom), (255,0,0), cv2.FILLED)

                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left+6, bottom-6), font, 1.0, (255,255,255), 1)
            st.image(frame)

if select_radio == "Use Camera":
    cap =cv2.VideoCapture(0)
    add_selectbox = st.sidebar.selectbox(
            "What operations would you like to perform?",
            ("Select", "Change Color", "Meshing", "Image Background Change","Face Recognition")
        )
    if add_selectbox == "Select":
            st.write("This application is a demo for streamlit.")


    elif add_selectbox == "Change Color": 
            add_selectbox_frame1 = st.sidebar.radio(
                "Which color do you want to set for your image?",
                ("About","GrayScale", "Blue", "Green", "Red"), key="selectbox_for_changecolor"
            )
            if add_selectbox_frame1 == "About":
                st.write("You will be able to change color if an image.")
            if add_selectbox_frame1 == "GrayScale":
                while cap.isOpened():
                    flag, frame = cap.read()
                    if not flag:
                        st.write("Couldn't access camera.")
                        break
                    # b1_frame = st.sidebar.button("Show GrayScale Image", help="Displays image")
                    # if b1_frame:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    small_frame = cv2.resize(frame, (0,0), fx=0.25, fy=0.25)
                    gray_image = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                    break
                st.sidebar.image(small_frame)
                st.image(gray_image)
                cap.release()  
                user_input = st.text_input("File name to save",value="xyz.jpg")
                save = st.button("Download Image", help="Download")
                if save:
                    output_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
                    cv2.imwrite(user_input,output_image)
                    st.success("File Saved successfully!")

            elif add_selectbox_frame1 == "Blue":
                while cap.isOpened():
                    flag, frame = cap.read()
                    if not flag:
                        st.write("Couldn't access camera.")
                        break
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    zeros = np.zeros(frame.shape[:2], dtype="uint8")
                    r, g, b = cv2.split(frame)
                    small_frame = cv2.resize(frame, (0,0), fx=0.25, fy=0.25)
                    blue_image = cv2.merge([zeros, zeros, b])
                    break
                st.sidebar.image(small_frame)
                st.image(blue_image)
                cap.release()
                user_input = st.text_input("File name to save",value="xyz.jpg")
                save = st.button("Download Image", help="Download")
                if save:
                    output_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
                    cv2.imwrite(user_input,output_image)
                    st.success("File Saved successfully!")
            elif add_selectbox_frame1 == "Green":
                while cap.isOpened():
                    flag, frame = cap.read()
                    if not flag:
                        st.write("Couldn't access camera.")
                        break
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    zeros = np.zeros(frame.shape[:2], dtype="uint8")
                    r, g, b = cv2.split(frame)
                    small_frame = cv2.resize(frame, (0,0), fx=0.25, fy=0.25)
                    green_image = cv2.merge([zeros, g, zeros])
                    break
                st.sidebar.image(small_frame)
                st.image(green_image)
                cap.release()
                user_input = st.text_input("File name to save",value="xyz.jpg")
                save = st.button("Download Image", help="Download")
                if save:
                    output_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
                    cv2.imwrite(user_input,output_image)
                    st.success("File Saved successfully!")
            elif add_selectbox_frame1 == "Red":
                while cap.isOpened():
                    flag, frame = cap.read()
                    if not flag:
                        st.write("Couldn't access camera.")
                        break
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    zeros = np.zeros(frame.shape[:2], dtype="uint8")
                    r, g, b = cv2.split(frame)
                    small_frame = cv2.resize(frame, (0,0), fx=0.25, fy=0.25)
                    red_image = cv2.merge([r, zeros, zeros])
                    break
                st.sidebar.image(small_frame)
                st.image(red_image)
                cap.release()
                user_input = st.text_input("File name to save",value="xyz.jpg")
                save = st.button("Download Image", help="Download")
                if save:
                    output_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
                    cv2.imwrite(user_input,output_image)
                    st.success("File Saved successfully!")
    elif add_selectbox == "Meshing":
        while cap.isOpened():
            flag, frame = cap.read()
            if not flag:
                st.write("Couldn't access camera.")
                break
                    # b1_frame = st.sidebar.button("Show GrayScale Image", help="Displays image")
                    # if b1_frame:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            small_frame = cv2.resize(frame, (0,0), fx=0.25, fy=0.25)
            results = model_face_mesh.process(frame)
            for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    frame, face_landmarks
                )
            break
        st.sidebar.image(small_frame)
        st.image(frame)
        cap.release()
        user_input = st.text_input("File name to save",value="xyz.jpg")
        save = st.button("Download Image", help="Download")
        if save:
            output_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
            cv2.imwrite(user_input,output_image)
            st.success("File Saved successfully!")
    elif add_selectbox == "Image Background Change":
            add_selectbox2 =  st.sidebar.radio(
                "Which background color you want to set for your image?",
                ("About","GrayScale", "Blue", "Green", "Red"))

            if add_selectbox2 == "About":
                st.write("Image with selected background color will be displayed here.")
            
            elif add_selectbox2 == "GrayScale":
                while cap.isOpened():
                    flag, frame = cap.read()
                    if not flag:
                        st.write("Couldn't access camera.")
                        break
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    small_frame = cv2.resize(frame, (0,0), fx=0.25, fy=0.25)
                    results = model.process(frame)
                    condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
                    bg_image = np.zeros(frame.shape, dtype=np.uint8)
                    bg_image[:] = (128, 128, 128)
                    bg_image = cv2.resize(bg_image, (frame.shape[1], frame.shape[0]))
                    output_image = np.where(condition, frame, bg_image)
                    #st.image(output_image)
                    break
                st.sidebar.image(small_frame)
                st.image(output_image)
                cap.release()
                user_input = st.text_input("File name to save",value="xyz.jpg")
                save = st.button("Download Image", help="Download")
                if save:
                    output_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
                    cv2.imwrite(user_input,output_image)
                    st.success("Saved File")


            elif add_selectbox2 == "Blue":
                while cap.isOpened():
                    flag, frame = cap.read()
                    if not flag:
                        st.write("Couldn't access camera.")
                        break
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    small_frame = cv2.resize(frame, (0,0), fx=0.25, fy=0.25)
                    results = model.process(frame)
                    condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
                    bg_image = np.zeros(frame.shape, dtype=np.uint8)
                    bg_image[:] = (0, 0, 255)
                    bg_image = cv2.resize(bg_image, (frame.shape[1], frame.shape[0]))
                    output_image = np.where(condition, frame, bg_image)
                    #st.image(output_image)
                    break
                st.sidebar.image(small_frame)
                st.image(output_image)
                cap.release()

                user_input = st.text_input("File name to save",value="xyz.jpg")
                save = st.button("Download Image", help="Download")
                if save:
                    output_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
                    cv2.imwrite(user_input,output_image)
                    st.success("Saved File")


            elif add_selectbox2 == "Green":
                while cap.isOpened():
                    flag, frame = cap.read()
                    if not flag:
                        st.write("Couldn't access camera.")
                        break
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    small_frame = cv2.resize(frame, (0,0), fx=0.25, fy=0.25)
                    results = model.process(frame)
                    condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
                    bg_image = np.zeros(frame.shape, dtype=np.uint8)
                    bg_image[:] = (0, 255, 0)
                    bg_image = cv2.resize(bg_image, (frame.shape[1], frame.shape[0]))
                    output_image = np.where(condition, frame, bg_image)
                    #st.image(output_image)
                    break
                st.sidebar.image(small_frame)
                st.image(output_image)
                cap.release()

                user_input = st.text_input("File name to save",value="xyz.jpg")
                save = st.button("Download Image", help="Download")
                if save:
                    output_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
                    cv2.imwrite(user_input,output_image)
                    st.success("Saved File")

            elif add_selectbox2 == "Red":
                while cap.isOpened():
                    flag, frame = cap.read()
                    if not flag:
                        st.write("Couldn't access camera.")
                        break
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    small_frame = cv2.resize(frame, (0,0), fx=0.25, fy=0.25)
                    results = model.process(frame)
                    condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
                    bg_image = np.zeros(frame.shape, dtype=np.uint8)
                    bg_image[:] = (255, 0, 0)
                    bg_image = cv2.resize(bg_image, (frame.shape[1], frame.shape[0]))
                    output_image = np.where(condition, frame, bg_image)
                    #st.image(output_image)
                    break
                st.sidebar.image(small_frame)
                st.image(output_image)
                cap.release()

                user_input = st.text_input("File name to save",value="xyz.jpg")
                save = st.button("Download Image", help="Download")
                if save:
                    output_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
                    cv2.imwrite(user_input,output_image)
                    st.success("File Saved successfully!")
    elif add_selectbox == "Face Recognition":
        while cap.isOpened():
            flag, frame = cap.read()
            if not flag:
                print("Couldn't access camera")
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            small_frame = cv2.resize(frame, (0, 0), fx=1/4, fy=1/4)
            rgb_small_frame = small_frame
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            name = "Unknown"
            face_names = []
            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index] 
                else:
                    name = "Unknown"
                face_names.append(name)      
            for(top, right, bottom, left), name in zip(face_locations, face_names):
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4
                cv2.rectangle(frame, (left, top), (right, bottom), (255,0,0), 2)
                cv2.rectangle(frame, (left, bottom-35), (right, bottom), (255,0,0), cv2.FILLED)

                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left+6, bottom-6), font, 1.0, (255,255,255), 1)
            if face_encodings is not Empty:
                break
        but = st.sidebar.button("Detect Face", help = "Detects Faces available in front of camera.")
        if but:
            st.sidebar.image(small_frame)
            st.image(frame)   
        cap.release()
    cv2.destroyAllWindows()