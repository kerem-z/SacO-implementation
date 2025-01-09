import torch
import cv2
import numpy as np
from torchvision import transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from PIL import Image

class CustomGradCam:
    def __init__(self, model, target_layer):
        self.model = model.eval()  # Set the model to evaluation mode
        self.feature = None  # To store the features from the target layer
        self.gradient = None  # To store the gradients from the target layer
        self.handlers = []  # List to keep track of hooks
        self.target_layer = target_layer  # Target layer for Grad-CAM
        self._get_hooks()  # Register hooks to the target layer

    # Hook to get features from the forward pass
    def _get_features_hook(self, module, input, output):
        self.feature = self.reshape_transform(output)  # Store and reshape the output features

    # Hook to get gradients from the backward pass
    def _get_grads_hook(self, module, input_grad, output_grad):
        self.gradient = self.reshape_transform(output_grad)  # Store and reshape the output gradients

        def _store_grad(grad):
            self.gradient = self.reshape_transform(grad)  # Store gradients for later use

        output_grad.register_hook(_store_grad)  # Register hook to store gradients

    # Register forward hooks to the target layer
    def _get_hooks(self):
        self.target_layer.register_forward_hook(self._get_features_hook)
        self.target_layer.register_forward_hook(self._get_grads_hook)

    # Function to reshape the tensor for visualization
    def reshape_transform(self, tensor, height=14, width=14):
        result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
        result = result.transpose(2, 3).transpose(1, 2)  # Rearrange dimensions to (C, H, W)
        return result

    # Function to compute the Grad-CAM salience_map
    def __call__(self, inputs):
        self.model.zero_grad()  # Zero the gradients
        output = self.model(inputs)  # Forward pass

        # Get the index of the highest score in the output
        index = np.argmax(output.cpu().data.numpy())
        target = output[0][index]  # Get the target score
        target.backward()  # Backward pass to compute gradients

        # Get the gradients and features
        gradient = self.gradient[0].cpu().data.numpy()
        weight = np.mean(gradient, axis=(1, 2))  # Average the gradients
        feature = self.feature[0].cpu().data.numpy()

        # Compute the weighted sum of the features
        cam = feature * weight[:, np.newaxis, np.newaxis]
        cam = np.sum(cam, axis=0)  # Sum over the channels
        cam = np.maximum(cam, 0)  # Apply ReLU to remove negative values

        # Normalize the salience_map
        cam -= np.min(cam)
        cam /= np.max(cam)
        cam = cv2.resize(cam, (224, 224))  # Resize to match the input image size
        return cam  # Return the Grad-CAM salience_map

# Ensure the model is on the correct device (CPU or GPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Assuming 'model' is the pre-trained model and 'target_layer' is the layer you want to use for Grad-CAM
model = model.to(device)
target_layer = model.blocks[-1].norm1  # Example for ViT, change if needed

# Preprocessing image
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize image to 224x224
    transforms.ToTensor(),  # Convert image to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize
])

# Load image (example)
image = Image.open("/content/ILSVRC2012_val_00006597.jpg")  # Replace with your image path
input_tensor = preprocess(image).unsqueeze(0).to(device)  # Add batch dimension and move to device

# Initialize Grad-CAM
grad_cam = CustomGradCam(model=model, target_layer=target_layer)

# Generate Grad-CAM salience_map
salience_map = grad_cam(input_tensor)  # Get Grad-CAM salience_map

# Normalize and display the salience_map
image_array = np.array(image) / 255.0  # Normalize the image for overlay
salience_map_resized = cv2.resize(salience_map, (image_array.shape[1], image_array.shape[0]))  # Resize salience_map

# Apply the salience_map to the image
cam_output = np.uint8(255 * salience_map_resized)
salience_map_colored = cv2.applyColorMap(cam_output, cv2.COLORMAP_JET)

# Blend the salience_map with the original image (for better visualization)
overlay = cv2.addWeighted(salience_map_colored, 0.5, (image_array * 255).astype(np.uint8), 0.5, 0)

# Save or display the final result
cv2.imwrite("linnet-gradcam-saliancemap.jpg", overlay)
cv2.waitKey(0)
cv2.destroyAllWindows()
