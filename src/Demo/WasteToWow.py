import cv2
from ultralytics import YOLO
import googlemaps.client
from llamaapi import LlamaAPI
import json
import requests
import anvil.server
import anvil.media
import PIL
#from faker import Faker
# Create a Faker instance with 'en_US' locale for American cities
#from geopy.geocoders import Nominatim
import pandas as pd
import random

anvil.server.connect('Our API Key') # to connect to server using anvil key

# Replace with your Google Maps API key
api_key = "Our API Key"

# Replace with your Llama 2 API key
llama_api_key = "Our API Key"

# Initialize the LlamaAPI client
llama = LlamaAPI(llama_api_key)  # Commented out as LlamaAPI is not defined in the provided code

# Load your trained YOLO model
model = YOLO(r'C:\Users\bhati\DATA298-FinalProject\YOLO\YOLO_2500_gd_annotations_latest_wt\best.pt')

# Create a Google Maps client
gmaps = googlemaps.Client(key=api_key)

#Function to get user's location
def get_user_location():
    try:
        # Use Geolocation API to get user's location data
        location = gmaps.geolocate()
        latitude = location["latitude"]
        longitude = location["longitude"]
        return latitude, longitude
    except Exception as e:
        print(f"Geolocation failed. Default San Jose Location: {e}")
        #return 37.3347, -121.9087 # San Jose

def find_centers(class_name, latitude, longitude, keyword):
    print("keyword is ",keyword)
    print(f"class is {class_name}, latitude is : {latitude} and longitude is {longitude}")

    base_url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
    params = {
        "key": api_key,
        "keyword": keyword, #f"donation centers such as thrift and goodwill stores for {class_name}",
        "location": f"{latitude},{longitude}",  # Include user's location
        "radius": 5000,  # Search radius in meters
        "type": "establishment"
    }
    response = requests.get(base_url, params=params)
    results = response.json()

    centers_info = []  # Initialize an empty list to store info about each center

    if results["status"] == "OK":  # Check for successful search
        donation_centers = results["results"][:5]

        for center in donation_centers:
            center_dict = {
                "Name": center['name'],
                "Address": center['vicinity'],
                "Rating": center.get('rating', 'Not available'),
                "Place ID": center['place_id'],
                "Latitude": center['geometry']['location']['lat'],
                "Longitude": center['geometry']['location']['lng']
            }
            centers_info.append(center_dict)
    else:
        # Handle potential API request issues
        print(f"API request unsuccessful. Status: {results['status']}")
        if 'error_message' in results:  # Check for error message if available
            print(f"Error message: {results['error_message']}")
    print(centers_info)
    return centers_info

# Function to call Llama 2 API with recycling query
def call_llama_api(item):
    api_request_json = {
        "messages": [
            {"role": "user", "content": f"Do not start the response with sure.Give one way in 2 lines to upcycle or DIY {item} ?"}
        ],
        # Optionally adjust model and parameters:
        "model": "llama-13b-chat",  # Choose a suitable Llama 2 model
        "max_length": 100,
        "temperature": 0.7,
    }

    response = llama.run(api_request_json)  # Commented out as LlamaAPI is not defined in the provided code
    if response:
        response_text = response.json()['choices'][0]['message']['content'].strip()  # Commented out as LlamaAPI is not defined in the provided code
        print(response.json())  # Commented out as LlamaAPI is not defined in the provided code
    else:
        response_text = "Recycling information not available at the moment."
    return response_text

@anvil.server.callable # only to be used when running through app otherwise comment this
def process_image_new(image_path):
    with anvil.media.TempFile(image_path) as filename: # only to be used when running through app otherwise comment this
        img = cv2.imread(filename) # only to be used when running through app otherwise comment this

    #image_path = cv2.imread('/content/drive/MyDrive/Data298AandB/Grounding_DINO/train_val_test/data/val/images/organic_004012_photo.jpg')
    # Run prediction on the image
    results = model.predict(img) # (image_path, stream=True) - change for running in notebook #(img) -for running through app

    # Assuming results is a list of prediction results for multiple images
    for result in results:
        # Initialize variables to store the class with maximum confidence and its corresponding score
        max_conf_class = None
        max_conf_score = -1

        # Iterate through each detection box
        for i in range(len(result.boxes.cls)):
            # Get the class index and confidence score for the current box
            class_idx = int(result.boxes.cls[i].item())  # Extract the scalar value from the tensor
            confidence = float(result.boxes.conf[i].item())  # Extract the scalar value from the tensor

            # Check if the current box has higher confidence than the previous maximum
            if confidence > max_conf_score:
                max_conf_score = confidence
                max_conf_class = result.names[class_idx]

    # Print the class with the maximum confidence score for the current image
    print("Image Class with maximum confidence score:", max_conf_class)
    print("Maximum confidence score:", max_conf_score)

    #return (max_conf_class)

    class_name = max_conf_class
    confidence = max_conf_score
    gmaps_results = []

    results_dict = {"Class": class_name, "DIY": None, "GMaps": None}

    try:
        # Get user's location
        user_latitude, user_longitude = get_user_location()
    except Exception as e:
        print("Using Default Location")
        user_latitude, user_longitude = 37.3347, -121.9087


    print(f"Class: {class_name}, Confidence: {confidence:.2f}")

    if class_name in ['clothes', 'shoes', 'furniture']:
        print("Displaying donation centres")
        keyword_donation = f"donation centers such as thrift and goodwill stores for {class_name}"
        if user_latitude and user_longitude:
            # Search for donation centers considering location
            gmaps_results = find_centers(class_name, user_latitude, user_longitude, keyword_donation)
            results_dict["DIY"] = "Check out these pinned donation centers near you"
        else:
            print("Failed to retrieve user location. Using generic search.")
            gmaps_results = find_centers(class_name,user_latitude, user_longitude,keyword_donation)  # Fallback to generic search
            results_dict["DIY"] = "Check out these pinned donation centers near you"
    elif class_name in ['medical']:
        print("Displaying disposal centers for medical waste")
        keyword_recycle = f"recycle or disposal places for waste or expired materials related to {class_name}"
        if user_latitude and user_longitude:
            # Search for donation centers considering location
            gmaps_results = find_centers(class_name, user_latitude, user_longitude, keyword_recycle)
            results_dict["DIY"] = "Be careful while disposing off this hazardous material at these pinned locations near you"
        else:
            print("Failed to retrieve user location. Using generic search.")
            gmaps_results = find_centers(class_name, user_latitude, user_longitude, keyword_recycle)  # Fallback to generic search
            results_dict["DIY"] = "Be careful while disposing off this hazardous material at these pinned locations near you"
    elif class_name in ['e-waste']:
        print("Displaying disposal centers for e-waste materials")
        keyword_recycle = f"recycle or disposal places for waste related to electronics"
        if user_latitude and user_longitude:
            # Search for donation centers considering location
            gmaps_results = find_centers(class_name, user_latitude, user_longitude, keyword_recycle)
            results_dict["DIY"] = "Be careful while disposing off this hazardous material at these pinned locations near you"
        else:
            print("Failed to retrieve user location. Using generic search.")
            gmaps_results = find_centers(class_name, user_latitude, user_longitude, keyword_recycle)  # Fallback to generic search
            results_dict["DIY"] = "Be careful while disposing off this hazardous material at these pinned locations near you"
    elif class_name in ['glass', 'plastic', 'metal', 'cardboard', 'paper']:
        keyword_recycle = f"recycle or disposal places for {class_name}"
        recycling_info = call_llama_api(class_name)
        print(f"Recycling Info for {class_name}: {recycling_info}")
        results_dict["DIY"] = recycling_info
        #elif user_choice.lower() == 'r':
        if user_latitude and user_longitude:
            # Search for donation centers considering location
            gmaps_results = find_centers(class_name, user_latitude, user_longitude, keyword_recycle)
            results_dict["GMaps"] = gmaps_results
        else:
            print("Failed to retrieve user location. Using generic search.")
            gmaps_results = find_centers(class_name, user_latitude, user_longitude, keyword_recycle)  # Fallback to generic search
            results_dict["GMaps"] = gmaps_results
        #else:
        #    print("Invalid input. Please enter 'u' or 'r")

    elif class_name == 'biowaste':
        keyword_recycle = f"nearby municipal office"
        print("Displaying recommendations for recycling")
        recycling_info = call_llama_api(class_name)
        print(f"Recycling Info for {class_name}: {recycling_info}")
        results_dict["DIY"] = recycling_info
        gmaps_results = find_centers(class_name, user_latitude, user_longitude, keyword_recycle)  # Fallback to generic search

    else:
        print('Invalid category')

    if gmaps_results:
        results_dict["GMaps"] = gmaps_results

    return results_dict

    #exit()

# Display the output image
# from google.colab.patches import cv2_imshow
# cv2_imshow(img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

def process_video():
    # Set up video capture

    cap = cv2.VideoCapture(0)  # Use 0 for webcam, or a video file path

    while True:
        # Capture a frame
        ret, frame = cap.read()

        # Run prediction on the frame
        results = model.predict(frame, stream=True)

        # Assuming results is a list of prediction results for multiple images
        for result in results:
            # Initialize variables to store the class with maximum confidence and its corresponding score
            max_conf_class = None
            max_conf_score = -1

            # Iterate through each detection box
            for i in range(len(result.boxes.cls)):
                # Get the class index and confidence score for the current box
                class_idx = int(result.boxes.cls[i].item())  # Extract the scalar value from the tensor
                confidence = float(result.boxes.conf[i].item())  # Extract the scalar value from the tensor

                # Check if the current box has higher confidence than the previous maximum
                if confidence > max_conf_score:
                    max_conf_score = confidence
                    max_conf_class = result.names[class_idx]

            # Print the class with the maximum confidence score for the current image
            print("Image Class with maximum confidence score:", max_conf_class)
            print("Maximum confidence score:", max_conf_score)

            class_name = max_conf_class
            confidence = max_conf_score
            gmaps_results = []

            results_dict = {"Class": class_name, "DIY": None, "GMaps": None}

            try:
                # Get user's location
                user_latitude, user_longitude = get_user_location()
            except Exception as e:
                print("Using Default Location")
                user_latitude, user_longitude = 37.3347, -121.9087


            print(f"Class: {class_name}, Confidence: {confidence:.2f}")

            if class_name in ['clothes', 'shoes', 'furniture']:
                print("Displaying donation centres")
                keyword_donation = f"donation centers such as thrift and goodwill stores for {class_name}"
                if user_latitude and user_longitude:
                    # Search for donation centers considering location
                    gmaps_results = find_centers(class_name, user_latitude, user_longitude, keyword_donation)
                    results_dict["DIY"] = "Check out these pinned donation centers near you"
                else:
                    print("Failed to retrieve user location. Using generic search.")
                    gmaps_results = find_centers(class_name,user_latitude, user_longitude,keyword_donation)  # Fallback to generic search
                    results_dict["DIY"] = "Check out these pinned donation centers near you"
            elif class_name in ['medical', 'e-waste']:
                print("Displaying disposal centers for hazardous materials")
                keyword_recycle = f"recycle or disposal places for {class_name}"
                if user_latitude and user_longitude:
                    # Search for donation centers considering location
                    gmaps_results = find_centers(class_name, user_latitude, user_longitude, keyword_recycle)
                    results_dict["DIY"] = "Be careful while disposing off this hazardous material at these pinned locations near you"
                else:
                    print("Failed to retrieve user location. Using generic search.")
                    gmaps_results = find_centers(class_name, user_latitude, user_longitude, keyword_recycle)  # Fallback to generic search
                    results_dict["DIY"] = "Be careful while disposing off this hazardous material at these pinned locations near you"
            elif class_name in ['glass', 'plastic', 'metal', 'cardboard', 'paper']:
                keyword_recycle = f"recycle or disposal places for {class_name}"
                recycling_info = call_llama_api(class_name)
                print(f"Recycling Info for {class_name}: {recycling_info}")
                results_dict["DIY"] = recycling_info
                #elif user_choice.lower() == 'r':
                if user_latitude and user_longitude:
                    # Search for donation centers considering location
                    gmaps_results = find_centers(class_name, user_latitude, user_longitude, keyword_recycle)
                    results_dict["GMaps"] = gmaps_results
                else:
                    print("Failed to retrieve user location. Using generic search.")
                    gmaps_results = find_centers(class_name, user_latitude, user_longitude, keyword_recycle)  # Fallback to generic search
                    results_dict["GMaps"] = gmaps_results
                #else:
                #    print("Invalid input. Please enter 'u' or 'r")

            elif class_name == 'biowaste':
                keyword_recycle = f"nearby municipal office"
                print("Displaying recommendations for recycling")
                recycling_info = call_llama_api(class_name)
                print(f"Recycling Info for {class_name}: {recycling_info}")
                results_dict["DIY"] = recycling_info
                gmaps_results = find_centers(class_name, user_latitude, user_longitude, keyword_recycle)  # Fallback to generic search

            else:
                print('Invalid category')

            if gmaps_results:
                results_dict["GMaps"] = gmaps_results

            return results_dict
        #exit()

        # Exit on 'q' press
        #if cv2.waitKey(1) & 0xFF == ord('q'):
        #    break

    # Release resources
    #cap.release()
    #cv2.destroyAllWindows()
    #break

    # Main function
def main():
    while True:
        user_choice = input("Would you like to process a static image (type 'image') or detect objects in real-time (type 'realtime')? ")
        if user_choice.lower() == 'image':
            image_path = input("Enter the path to the image: ")
            process_image_new(image_path)
            break
        elif user_choice.lower() == 'realtime':
            process_video()
        else:
            print("Invalid input. Please enter 'image' or 'realtime'.")

#main()
anvil.server.wait_forever()
