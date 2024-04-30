# DATA298A-B

Waste to Wow: Home Waste Management Recommendation System - Recycling and Upcycling

Authors: Vidushi Bhati, Sonali Arcot, Siddharth Solanki, and Krishna Sameera Surapaneni

Affiliation: Department of Applied Data Science, San Jose State University

Course: DATA 298B: MSDA Project 11

Instructor: Dr. Ming Hwa Wang


Approximately 80% of recyclable household waste is incorrectly discarded, leading to landfill congestion and increased carbon emissions. This project addresses this issue by developing a system for classifying household waste and providing instructions on recycling and safe disposal. Unlike previous research focusing on industrial waste, this project emphasizes household items. The data, sourced from various repositories and expanded to include new categories, undergoes meticulous preprocessing. Advanced models like InceptionResNetv2, YOLOv8, MobileNetv2, ResNet-50, and Xception are employed for accurate waste categorization. Integrated with a Language Model (LLM), the system furnishes recommendations for waste management based on item classification, aiming to positively impact sustainable waste management at the household level.


Project Approaches and Methods -

Data Collection: Data is collected from various online repositories like Trashbox, Mendeley, and Kaggle, focusing on household waste items. A novel category, Homegoods, is introduced to encompass items like furniture and clothing.

Sources -

Garbage Classification (12 classes). (2021, January 24). Kaggle. https://www.kaggle.com/datasets/mostafaabla/garbage-classification

Gurumurthy, S., & Manga, S. (2020). Furniture Identification. Kaggle. https://www.kaggle.com/competitions/day-3-kaggle-competition

Kumsetty, N. V., Nekkare, A. B., S, S. K., & M, A. K. (2022). TrashBox: Trash Detection and Classification using Quantum Transfer Learning. 2022 31st Conference of Open Innovations Association (FRUCT). https://doi.org/10.23919/fruct54823.2022.9770922

Nnamoko, N. (2023). Waste Classification Dataset. Mendeley Data. https://doi.org/10.17632/n3gtgm9jxj.3

Data Preparation: Collected data undergoes rigorous preprocessing, including annotation, augmentation, normalization, and feature engineering, to prepare it for training deep learning models.

Modeling: Cutting-edge deep learning models like YOLOv8, Xception, MobileNetV2, ResNet-50 are employed for accurate waste detection.

Deployment: An intuitive user interface (UI) is developed for real-time waste classification and recommendations using Anvil. Integrated with Google Maps API and LLM, users receive personalized disposal recommendations.


File Structure -

project_root/

├── docs/

│   ├── Presentation/

│   ├── Workbooks/

│   └── Report/

└── src/

    ├── Data Processing/
    
    └── Models/
    
        ├── Augmentation/
        
        ├── Classification/
        
        └── Detection/
