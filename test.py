import requests


def predict_digit(image_path):
    url = "http://localhost:5000/predict"
    with open(image_path, "rb") as f:
        files = {"image": f}
        response = requests.post(url, files=files)

    if response.status_code == 200:
        result = response.json()
        return result["predicted_digit"]
    else:
        print(f"錯誤: {response.text}")
        return None


# 使用示例
digit = predict_digit("ph.png")
print(f"預測的數字是: {digit}")
