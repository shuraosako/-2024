from flask import Flask, request, send_file
from PIL import Image
import io
import numpy as np
import torch
from cyclegan.cyclegan import Generator

app = Flask(__name__)

MODEL_PATH = {
    'statue_to_human': 'models/model_generator_AB_246.pth',
    'human_to_statue': 'models/model_generator_BA_246.pth'
}

statue_to_human = Generator()
human_to_statue = Generator()

statue_to_human.load_state_dict(torch.load(MODEL_PATH['statue_to_human'], map_location=torch.device('cpu')))
human_to_statue.load_state_dict(torch.load(MODEL_PATH['human_to_statue'], map_location=torch.device('cpu')))

statue_to_human.eval()
human_to_statue.eval()

def preprocess_image(img):
    img = torch.from_numpy(img.transpose((2, 0, 1))).float()
    img = img.unsqueeze(0) / 255.0
    img = (img - 0.5) / 0.5
    return img

def postprocess_image(tensor):
    img = (tensor + 1) / 2
    img = img.squeeze().detach().numpy()
    img = img.transpose(1, 2, 0) * 255.0
    return img

@app.route('/transform', methods=['POST'])
def transform_image():
    if 'image' not in request.files:
        return 'No image file provided', 400
    
    direction = request.form.get('direction', 'statue_to_human')
    file = request.files['image']
    img = Image.open(file.stream)
    
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    img_array = np.array(img)
    img_tensor = preprocess_image(img_array)
    
    with torch.no_grad():
        if direction == 'statue_to_human':
            transformed_tensor = statue_to_human(img_tensor)
        else:
            transformed_tensor = human_to_statue(img_tensor)
    
    transformed_array = postprocess_image(transformed_tensor)
    transformed_img = Image.fromarray(transformed_array.astype('uint8'))
    
    img_byte_arr = io.BytesIO()
    transformed_img.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    
    return send_file(img_byte_arr, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)