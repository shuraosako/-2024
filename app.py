from flask import Flask, request, send_file
from PIL import Image
import io
import numpy as np
# CycleGANモデルをインポートする
# from cyclegan_model import CycleGANModel

app = Flask(__name__)

# CycleGANモデルのインスタンスを作成
# model = CycleGANModel()

@app.route('/transform', methods=['POST'])
def transform_image():
    if 'image' not in request.files:
        return 'No image file provided', 400
    
    file = request.files['image']
    img = Image.open(file.stream)
    
    # 画像をNumPy配列に変換
    img_array = np.array(img)
    
    # CycleGANモデルを使用して画像を変換
    # transformed_img_array = model.transform(img_array)
    # 仮の処理：元の画像をそのまま返す
    transformed_img_array = img_array
    
    # NumPy配列をPIL Imageに変換
    transformed_img = Image.fromarray(transformed_img_array.astype('uint8'))
    
    # 変換された画像をバイトストリームに変換
    img_byte_arr = io.BytesIO()
    transformed_img.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    
    return send_file(img_byte_arr, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)