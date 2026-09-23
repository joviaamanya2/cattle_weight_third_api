# Cattle Weight Prediction API

This service loads `cattle_weight_B3_B4_MobileNetV3_BASELINE.keras` and exposes the contract already used by the Flutter app.

## Run locally

From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r api\requirements.txt
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

The model path defaults to the repository root. To use another model file:

```powershell
$env:MODEL_PATH = 'C:\path\to\model.keras'
```

## Endpoints

### `GET /health`

Returns service and model information.

### `POST /predict`

Send a `multipart/form-data` request with the image in a field named `file`:

```powershell
curl.exe -X POST http://127.0.0.1:8000/predict -F "file=@path\to\cow.jpg"
```

The JSON response includes `predicted_weight_kg` and the compatibility fields expected by the Flutter client.

The saved model is a weight regressor only. It does not contain a cattle detector or uncertainty metadata, so `cattle_detected` is a compatibility flag and confidence/error fields are returned as `0.0`. Use a separate detector and calibrated validation metrics before treating those fields as measurements.

## Flutter configuration

Point the existing client to the running API when launching the app:

```powershell
flutter run --dart-define=API_BASE_URL=http://10.0.2.2:8000
```

Use `http://127.0.0.1:8000` for Flutter web/desktop. For a physical Android device, use the development machine's LAN IP instead of `10.0.2.2`.