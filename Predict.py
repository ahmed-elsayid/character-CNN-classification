from preprocessing import  Chr_encoder
from model import create_model
import torch
from config import  get_config

class_labels = {0:"world",1:"Sports",2:"Business",3:"Sci/Tech"}

def encode_text(text):
    encoder = Chr_encoder()
    encoded_text = encoder.encode(text)
    return encoded_text.unsqueeze(0)

def load_model(model_path,config_name="baseline"):
    config = get_config(config_name)
    model = create_model(config)
    state = torch.load(model_path, map_location=torch.device('cpu'))
    if "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    else:
        model.load_state_dict(state)
    model.eval()

    return model

def predict(text,model_path,config_name="baseline"):
    if not text or not text.strip():
        return "Input text is empty. Please provide valid text for prediction."
    model = load_model(model_path,config_name)
    encoded_text = encode_text(text)
    with torch.no_grad():
        output = model(encoded_text)
        probabilities = torch.nn.functional.softmax(output, dim=1).squeeze().numpy()
        predicted_label = int(probabilities.argmax())
    return {
        "label": class_labels.get(predicted_label, "Unknown"),
        "confidence": float(probabilities[predicted_label]),
        "probabilities": {class_labels[i]: float(prob) for i, prob in enumerate(probabilities)}

    }

if __name__ == "__main__":
    model_path = "models/ablation2_use_adam_optimizer_best.pth"
    input_text = "STOCK MARKET NEWS: Apple Inc. reports record earnings for Q2 2024,driven by strong iPhone sales and growth in services "
    predicted_category = predict(input_text, model_path)
    print(f"Predicted Category: {predicted_category['label']}")
    print(f"Confidence: {predicted_category['confidence']:.4f}")
    print(f"Probabilities: {predicted_category['probabilities']}")