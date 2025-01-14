# Color-Classification-ML-Model

Project: Color Classification using Machine Learning Models
🔬 Overview: In this project, I focused on building an image classification model to identify the dominant color in input images. Using a diverse set of machine learning models—namely Convolutional Neural Networks (CNNs), Support Vector Machine (SVM), and Random Forest (RF)—the goal was to train and evaluate each model on a color dataset, ultimately identifying the best model for classifying images by their color.

The project achieved an 84% accuracy with CNN, showing promising results for real-world applications in various industries that require automatic color categorization from images.

Key Features and Methodology
Dataset: The dataset used consists of images belonging to different color categories (e.g., Red, Blue, Green, Yellow). Each image in the dataset is preprocessed and resized to a consistent shape to ensure effective model training.

Data Preprocessing:

Image Resizing: All images were resized to the input shape required by the model.
Normalization: Pixel values were normalized (divided by 255) to fall in the range of [0, 1].
Flattening: Images were flattened for SVM and Random Forest models to convert them into one-dimensional arrays suitable for these algorithms.
Model Approach:

CNN:
I trained a CNN using Keras that included multiple convolutional layers, pooling layers, and fully connected layers for classification. This model showed the best results in the context of image data classification.

SVM:
The Support Vector Machine was trained on a flattened version of the images, suitable for non-CNN models, and the test accuracy and confusion matrix were evaluated.

Random Forest:
The Random Forest model, trained on the flattened images, performed decently but underperformed compared to CNN for this task.

Model Evaluation & Results
After training and testing the models, I analyzed their performance using test accuracy, confusion matrices, and detailed classification reports (precision, recall, F1-score).
CNN outperformed both SVM and Random Forest, achieving an accuracy of 84%, showcasing its effectiveness for color classification from images. The confusion matrix revealed areas where misclassifications happened, and efforts can be made to improve accuracy.
A color prediction function was implemented that predicts the dominant color from input images by resizing, normalizing, and running predictions with the trained model.
Real-Life Applications
The color classification model holds various real-world applications:

Retail & E-commerce:
Automating the identification of colors in product images for inventory management and helping customers filter search results based on their preferred colors.

Digital Design & Media:
Assisting designers and creators with automatic color tagging for digital artwork, photography, or visual media, simplifying the process of sorting images based on color tones.

Fashion Industry:
The model can aid in classifying clothing or accessories based on color for online retail platforms, allowing users to search for specific colors quickly and efficiently.

Automated Quality Control:
Manufacturers can use this color classification model to check products' quality by color consistency in industrial production, like packaging or food processing.

Automated Vehicle Systems:
Color detection is essential in areas like traffic signals, road markings, or color-coded systems used by autonomous vehicles or drones for safe navigation and interaction with environments.

Healthcare:
In fields like dermatology, dentistry, or medical imaging, color classification models can be employed to detect and monitor color changes in scans, images, or treatments.

Next Steps & Future Improvements
To further enhance the model's performance:

Increase the dataset size and include more color variations, lighting conditions, and image backgrounds.
Data augmentation techniques like rotation, flipping, or zoom could be used to improve model generalization and reduce overfitting.
Transfer learning with pre-trained CNN models (like VGG, ResNet) could be explored to further increase accuracy, especially with larger datasets.
Tools and Technologies Used
Python
Keras/TensorFlow (for CNN)
scikit-learn (for SVM & Random Forest)
OpenCV (image handling & preprocessing)
Matplotlib, Seaborn (visualizations)
Conclusion
By leveraging the power of CNN, Color Classification tasks that once seemed daunting are now efficiently tackled with machine learning. This project demonstrates the benefits of deep learning for image classification problems and opens doors for advanced applications in various domains where color plays a crucial role.
