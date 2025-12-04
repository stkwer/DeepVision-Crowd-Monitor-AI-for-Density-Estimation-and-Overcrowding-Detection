# DeepVision Crowd Monitor  
AI for Density Estimation and Overcrowding Detection

##  Project Description

DeepVision Crowd Monitor is a Python-based AI solution for estimating crowd density and detecting overcrowding in images or video feeds.  
It uses a deep-learning crowd-counting / density-estimation model to output density maps, crowd counts, and helps flag potential overcrowded zones — useful for crowd monitoring in public spaces, events, or surveillance systems.

##  Repository Contents (key files)

- `app_streamlit.py` — Streamlit-based UI frontend for live inference / monitoring.  
- `crowd_app.py` — Main application logic for processing inputs and crowd/density estimation.  
- `train_csrnet.py` — Script to train the underlying model.  
- `model_csrnet.py` — The model definition (likely implementing CSRNet-based architecture).  
- `evaluate_and_plot.py`, `visualize_density_maps.py` — Scripts to evaluate model performance / generate density visualizations.  
- `inference_utils.py`, `process_crowd.py` — Utility scripts for inference and processing.  
- `requirements.txt` — List of Python dependencies.  
- `.gitignore`, `LICENSE` (MIT) — typical project metadata files.  

##  Requirements & Setup

1. Ensure you have Python installed (version compatible with the dependencies).  
2. Install dependencies using:  
   ```bash
   pip install -r requirements.txt
   ```  
3. (Optional) If you want to train the model — check `train_csrnet.py` for dataset and configuration requirements.  

##  Usage

### Running inference / monitoring (with UI)
```bash
python app_streamlit.py
```
This should launch a local Streamlit app that allows you to input images / video feeds and visualize crowd density / overcrowding detection.

### For training / evaluation
- To train the model:  
  ```bash
  python train_csrnet.py
  ```  
- To visualize density maps / evaluate results:  
  ```bash
  python visualize_density_maps.py
  ```  
  or  
  ```bash
  python evaluate_and_plot.py
  ```

##  Example Workflow

1. Launch the Streamlit app (`app_streamlit.py`) for live crowd monitoring.  
2. Feed an image / video stream or upload a file.  
3. The system outputs a density map, estimated crowd count, and highlights zones with high crowd density / possible overcrowding.  
4. (Optional) If you modify/train the model, you can evaluate performance using the evaluation / visualization scripts.

##  License

This project is licensed under the MIT License — see the `LICENSE` file for details.
