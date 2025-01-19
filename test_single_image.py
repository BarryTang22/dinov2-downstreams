import torch
from PIL import Image
from torchvision import transforms
from models import dinov2_classifier

def predict_image(image_path, checkpoint_path, top_k=5):
    # Load model and checkpoint
    model = dinov2_classifier(num_classes=1000)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.head.load_state_dict(checkpoint['head_state_dict'])
    model.eval()
    
    # Preprocess image
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Load and transform image
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0)
    
    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    input_tensor = input_tensor.to(device)
    
    # Get prediction
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        top_prob, top_class = torch.topk(probabilities, top_k)
    
    # ImageNet class names (just the relevant cat classes for now)
    # You can add more classes if needed
    imagenet_classes = {
        281: "tabby cat",
        282: "tiger cat",
        283: "persian cat",
        284: "siamese cat",
        285: "egyptian cat"
    }
    
    # Print results
    print(f"\nPredictions for {image_path}:")
    for i in range(top_k):
        class_idx = top_class[0][i].item()
        class_name = imagenet_classes.get(class_idx, f"Class {class_idx}")
        print(f"{class_name}: {top_prob[0][i].item()*100:.2f}%")
    
    return top_class[0], top_prob[0]

if __name__ == "__main__":
    image_path = "/home/ct1y23/ct1y23/dinov2-downstreams/coco_cat.jpg"
    checkpoint_path = "/home/ct1y23/ct1y23/dinov2-downstreams/gpu2-256-10epoch/dinov2-head-epoch=03-val_acc=0.79.ckpt"
    predict_image(image_path, checkpoint_path, top_k=5) 