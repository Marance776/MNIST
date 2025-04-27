import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from flask import Flask, request, jsonify, Blueprint, render_template, send_from_directory
from flask_cors import CORS
from PIL import Image
import io
import threading

# 主應用 - 用於 AI 模型預測
app = Flask(__name__)
CORS(app)

# 靜態文件應用 - 用於提供 index.html
static_app = Flask(__name__, static_folder='static')
CORS(static_app)

# 全局變數，用於存儲訓練後的模型
model = None


def train_and_save_model():
    """訓練 MNIST 模型並保存"""
    print("開始加載數據集...")
    # 加載 MNIST 數據集
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # 數據預處理
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

    print("數據集加載完成。訓練數據: {}, 測試數據: {}".format(len(x_train), len(x_test)))

    # 建立 CNN 模型
    print("開始構建模型...")
    model = keras.Sequential([
        keras.Input(shape=(28, 28, 1)),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(10, activation="softmax"),
    ])

    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer="adam", metrics=["accuracy"])

    # 訓練模型
    print("開始訓練模型...")
    model.fit(x_train, y_train, batch_size=128, epochs=5,
              validation_split=0.1, verbose=1)

    # 評估模型
    score = model.evaluate(x_test, y_test, verbose=0)
    print(f"測試集準確率: {score[1]:.4f}")

    # 保存模型
    model_dir = 'model'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_path = os.path.join(model_dir, 'mnist_model.h5')
    model.save(model_path)
    print(f"模型已保存至 {model_path}")

    return model


def preprocess_image(image_data):
    """將上傳的圖像數據預處理成模型可接受的格式"""
    # 將圖像轉換為 PIL 圖像
    img = Image.open(io.BytesIO(image_data)).convert('L')
    # 調整圖像大小為 28x28
    img = img.resize((28, 28))
    # 將圖像轉換為 numpy 數組
    img_array = np.array(img)
    # 反轉顏色（MNIST 數據集中數字是白底黑字）
    img_array = 255 - img_array
    # 標準化到 [0, 1]
    img_array = img_array.astype("float32") / 255.0
    # 添加通道維度
    img_array = np.expand_dims(img_array, axis=-1)
    # 添加批次維度
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


@app.route('/predict', methods=['POST'])
def predict():
    """接收一張圖片並預測數字"""
    global model

    # 確保模型已加載
    if model is None:
        # 檢查是否有已保存的模型
        if os.path.exists('model/mnist_model.h5'):
            print("加載現有模型...")
            model = keras.models.load_model('model/mnist_model.h5')
        else:
            # 如果沒有，則訓練一個新模型
            print("沒有找到現有模型，開始訓練新模型...")
            model = train_and_save_model()

    # 從請求中獲取圖像
    if 'image' not in request.files:
        return jsonify({"error": "沒有提供圖像"}), 400

    image_file = request.files['image']
    image_data = image_file.read()

    # 預處理圖像
    processed_image = preprocess_image(image_data)

    # 預測
    prediction = model.predict(processed_image)
    predicted_class = np.argmax(prediction, axis=1)[0]

    # 返回預測結果
    return jsonify({"predicted_digit": int(predicted_class)})


@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy"})


# 靜態應用的路由
@static_app.route('/', methods=['GET'])
def index():
    return send_from_directory('static', 'index-zh.html')

@static_app.route('/<path:path>', methods=['GET'])
def static_files(path):
    return send_from_directory('static', path)


def run_ai_app():
    """運行 AI 模型預測應用"""
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)


def run_static_app():
    """運行靜態文件應用"""
    static_app.run(host='0.0.0.0', port=80)


if __name__ == "__main__":
    # 檢查是否有已保存的模型，如果沒有，則訓練並保存
    if not os.path.exists('model/mnist_model.h5'):
        print("首次運行，訓練並保存模型...")
        model = train_and_save_model()
    else:
        print("加載現有模型...")
        model = keras.models.load_model('model/mnist_model.h5')

    # 創建並啟動靜態文件應用線程
    static_thread = threading.Thread(target=run_static_app)
    static_thread.daemon = True  # 確保主程序結束時線程也會結束
    static_thread.start()
    print("靜態文件服務已在 80 端口啟動")

    # 啟動 AI 模型預測應用
    print(f"AI 模型預測服務啟動在 {os.environ.get('PORT', 5000)} 端口")
    run_ai_app()
