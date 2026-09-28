import requests
import os

def test_api():
    # Health check
    print("Testing /api/health ...")
    r = requests.get('http://localhost:8000/api/health')
    print("Health response:", r.json())
    
    # Predict
    image_path = os.path.join(os.path.dirname(__file__), 'photos', '000_jpg.rf.37110a5377a77cf7b46daac60c0158cd.jpg')
    print(f"Testing /api/predict with {image_path} ...")
    
    with open(image_path, 'rb') as f:
        files = {'file': f}
        r = requests.post('http://localhost:8000/api/predict', files=files)
        
    print("Predict response:", r.json())

if __name__ == '__main__':
    test_api()
