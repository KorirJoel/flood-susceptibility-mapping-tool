from flask import Flask, request, jsonify, send_from_directory
from flask import Flask, render_template, request, send_file
from flask_cors import CORS
import os
import numpy as np
import pandas as pd
import rasterio
from rasterio import Affine
from rasterio.enums import Resampling
import joblib
from sklearn.ensemble import RandomForestClassifier
import tempfile
from jenkspy import JenksNaturalBreaks
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for server
import matplotlib.pyplot as plt
import jenkspy
import shutil
import numpy as np
import matplotlib.pyplot as plt
import rasterio
from matplotlib.colors import ListedColormap

app = Flask(__name__, template_folder='templates', static_folder='static')
CORS(app)

# Expected order of flood factor rasters
flood_factor_order = [
    'twi', 'tri', 'tpi', 'spi', 'soil_type', 'slope', 'profile_cu',
    'ppt', 'plan_curva', 'ndvi', 'lulc', 'elevation', 'dtoS', 'aspect'
]

# Load your trained model
model = joblib.load('rf_flood_model.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/flood')
def flood():
    return render_template('flood.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        uploaded_files = request.files.getlist("files[]")
        if not uploaded_files:
            return jsonify({"error": "No files uploaded"}), 400

        with tempfile.TemporaryDirectory() as temp_dir:
            saved_paths = []
            for file in uploaded_files:
                save_path = os.path.join(temp_dir, file.filename)
                file.save(save_path)
                saved_paths.append(save_path)

            # Create dict of filename without extension → full path (case-sensitive)
            file_dict = {
                os.path.splitext(os.path.basename(f))[0]: f
                for f in saved_paths
            }

            rasters = []
            transform = None
            dtoS_profile = None

            for factor in flood_factor_order:
                if factor not in file_dict:
                    return jsonify({'error': f'Missing raster for: {factor}'}), 400

                with rasterio.open(file_dict[factor]) as src:
                    arr = src.read(1, resampling=Resampling.bilinear).astype(np.float32)
                    arr[arr == src.nodata] = np.nan
                    rasters.append(arr)

                    # Store profile from reference raster
                    if factor == 'dtoS':
                        dtoS_profile = src.profile
                        transform = src.transform

            # Stack into 3D array (height, width, bands)
            raster_stack = np.stack(rasters, axis=-1)

            # Create mask of valid pixels (no NaNs in any band)
            valid_mask = ~np.any(np.isnan(raster_stack), axis=-1)

            # Extract valid pixels (N x features)
            valid_pixels = raster_stack[valid_mask]
            df_predict = pd.DataFrame(valid_pixels, columns=flood_factor_order)

            # Batch prediction (for memory efficiency)
            def batch_predict(X, batch_size=100000):
                return np.concatenate([
                    model.predict_proba(X[i:i+batch_size])[:, 1]
                    for i in range(0, len(X), batch_size)
                ])

            probabilities = batch_predict(df_predict)

            # Reconstruct probability map (NaN for invalid)
            prob_map = np.full(valid_mask.shape, np.nan, dtype=np.float32)
            prob_map[valid_mask] = probabilities

            # Fast classification into 5 classes using percentiles
            breaks = np.percentile(probabilities, [20, 40, 60, 80])
            classes = np.digitize(probabilities, breaks)

            # Rebuild full-classified raster
            classified = np.full(valid_mask.shape, 0, dtype=np.uint8)
            classified[valid_mask] = classes + 1  # Classes 1–5

            # Save as GeoTIFF for frontend rendering
            # Save as GeoTIFF
            # Save classified raster to a persistent folder (e.g., 'outputs')
            
            from matplotlib.colors import ListedColormap, BoundaryNorm

            # Set output directory for PNG
            output_dir = os.path.join(os.getcwd(), 'static')
            os.makedirs(output_dir, exist_ok=True)

            # Define PNG path
            png_path = os.path.join(output_dir, 'fsm_overlay.png')

            # Assume `classified` is a 2D NumPy array of class values: 0 (NoData), 1–5 valid classes
            # Also assume `transform` and `dtoS_profile['crs']` are already available

            # Get raster extent
            height, width = classified.shape
            west, north = transform * (0, 0)
            east, south = transform * (width, height)

            # Define bounds for Leaflet
            bounds = [
                [south, west],  # SW
                [north, east]   # NE
            ]

            # Mask NoData values (assume 0 = NoData)
            masked_data = np.ma.masked_equal(classified, 0)

            # Define custom colormap for 5 classes (1–5)
            # You can change colors to fit your theme
            class_colors = [
                (0.0, 0.0, 0.5, 1.0),   # Class 1 - dark blue
                (0.0, 0.5, 0.0, 1.0),   # Class 2 - green
                (1.0, 1.0, 0.0, 1.0),   # Class 3 - yellow
                (1.0, 0.5, 0.0, 1.0),   # Class 4 - orange
                (1.0, 0.0, 0.0, 1.0),   # Class 5 - red
            ]
            cmap = ListedColormap(class_colors)
            norm = BoundaryNorm([0.5, 1.5, 2.5, 3.5, 4.5, 5.5], cmap.N)

            # Plot and save PNG with transparency for NoData
            plt.figure(figsize=(10, 8))
            plt.axis('off')
            plt.imshow(masked_data, cmap=cmap, norm=norm)
            plt.savefig(png_path, transparent=True, bbox_inches='tight', pad_inches=0)
            plt.close()

            # Return to frontend
            return jsonify({
                'flood_map_url': f'/static/fsm_overlay.png',
                'bounds': bounds
            })


            
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True)