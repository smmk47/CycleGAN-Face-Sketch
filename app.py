import os
from flask import Flask, request, render_template, redirect, url_for, send_from_directory, flash
from PIL import Image
import torch
from torchvision import transforms
from generator import Generator  # Import the Generator class

# Initialize Flask app
app = Flask(__name__)
app.secret_key = "supersecretkey"

# Configure device (use GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load models
def load_model(model, checkpoint_path):
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()  # Set the model to evaluation mode
    return model

# Initialize Generators
G_A2B = Generator(input_nc=3, output_nc=3).to(device)
G_B2A = Generator(input_nc=3, output_nc=3).to(device)

# Load the checkpoints
G_A2B = load_model(G_A2B, 'checkpoints/G_A2B_epoch_8.pth')
G_B2A = load_model(G_B2A, 'checkpoints/G_B2A_epoch_8.pth')

# Define transformations for the input images
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize image to 256x256
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize
])

# Home route (UI)
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Ensure image is uploaded
        if 'image' not in request.files:
            flash('No image uploaded!', 'error')
            return redirect(request.url)

        img_file = request.files['image']
        if img_file.filename == '':
            flash('No file selected!', 'error')
            return redirect(request.url)

        conversion_type = request.form.get('conversion_type')
        if not conversion_type:
            flash('Please select a conversion type!', 'error')
            return redirect(request.url)

        # Process the uploaded image
        img = Image.open(img_file).convert('RGB')
        img = transform(img).unsqueeze(0).to(device)  # Add batch dimension

        # Run inference using the appropriate model
        if conversion_type == 'photo_to_sketch':
            with torch.no_grad():
                output = G_A2B(img)
        elif conversion_type == 'sketch_to_photo':
            with torch.no_grad():
                output = G_B2A(img)
        else:
            flash('Invalid conversion type!', 'error')
            return redirect(request.url)

        # Convert the output tensor to an image
        output = output.squeeze(0).cpu()
        output_img = transforms.ToPILImage()(output)

        # Save the output image
        output_path = os.path.join('static', 'output.png')
        output_img.save(output_path)

        return redirect(url_for('display_image', filename='output.png'))

    return render_template('index.html')

# Route to display the generated image
@app.route('/static/<path:filename>')
def display_image(filename):
    return send_from_directory('static', filename)

# Favicon route to prevent 404 errors
@app.route('/favicon.ico')
def favicon():
    return '', 204

# Run the Flask app
if __name__ == '__main__':
    os.makedirs('static', exist_ok=True)  # Create static directory if it doesn't exist
    app.run(host='0.0.0.0', port=5000, debug=True)
