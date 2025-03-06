# Animal Classification (Cat, Dog, Snake) using CNN and VGG16

![Animal Classification](https://media.istockphoto.com/id/106457209/photo/group-of-pets-dog-cat-bird-reptile-rodent-ferret-fish.jpg?s=612x612&w=0&k=20&c=GbbKTrJxNkR9iN7UPbK510fn6WzVnFzB1lpe8xE1eDs=)

### Link to Kaggle Model Training: 
`https://www.kaggle.com/code/geekypriyanka/animal-classification`

## Overview
This project classifies images of **Cats**, **Dogs**, and **Snakes** using a Convolutional Neural Network (CNN) with a pre-trained **VGG16** model. The trained model is deployed on a local **Streamlit** app.

## Key Features:
- **Model**: VGG16 pre-trained and fine-tuned for animal classification.
- **Deployment**: Streamlit app for easy, interactive use.
- **Saved Model**: The model is saved as `model.h5` for reuse.

---

## Requirements
Install the necessary packages with: **pip install -r requirements.txt**


---

## How to Run

1. **Train the Model**: 
   - The model was trained on Kaggle using **VGG16** and a dataset of cats, dogs, and snakes images. The model is fine-tuned and saved as `model.h5`.

2. **Streamlit App**: 
   - After installing the required packages, run the Streamlit app: **streamlit run app.py**

- Access the app at `localhost:8501`.

3. **Test the Model**: 
- Upload images of cats, dogs, or snakes, and the model will predict the class.

---

## Model Details

- **Architecture**: VGG16 (Pre-trained)
- **Model File**: `model.h5`
- **Accuracy**: 92% on the validation set.

---

## Example Usage (Streamlit)
1. **Upload an Image**: Drag and drop an image.
2. **Classify**: The model predicts whether the image is a **cat**, **dog**, or **snake**.

**Screenshots of Prediction**:

![Predicted Result1](https://github.com/fractalpriyanka/Animal_classification/blob/main/predicted_images/predicted%20image%204.png?raw=true)
![Predicted Result2](https://github.com/fractalpriyanka/Animal_classification/blob/main/predicted_images/predicted%20image%206.png?raw=true)

---

## Conclusion
This project demonstrates using the **VGG16** pre-trained model for image classification. The model is fine-tuned and deployed locally using **Streamlit** for easy interaction.

---

## Future Work
- Add more animal classes.
- Experiment with other architectures like **ResNet**.
- Deploy the app on cloud platforms like **Heroku** or **AWS**.

---

## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/fractalpriyanka/Animal_classification/blob/main/LICENSE) file for details.
