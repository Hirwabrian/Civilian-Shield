import streamlit as st
import os
import sys
import time
import pandas as pd
import numpy as np
import pickle

# LIBRARY CHECKS
try:
    import librosa
    from tensorflow.keras.utils import to_categorical
except ImportError:
    st.error("ERROR: Missing libraries. Please install: `pip install librosa tensorflow`")
    st.stop()

# PATH SETUP
current_file_path = os.path.abspath(__file__)
app_folder = os.path.dirname(current_file_path)
parent_dir = os.path.dirname(app_folder)

if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# IMPORTS FROM SRC
try:
    from src.prediction import predict_audio
    from src.model import train_and_save_model
    # We keep get_data just in case, though we use local extraction for retraining now
    from src.preprocessing import get_data 
except ImportError:
    st.error("SYSTEM ERROR: CORE SRC MODULES NOT FOUND.")
    st.stop()

# 1. PAGE CONFIGURATION
st.set_page_config(
    page_title="CIVILIAN_SHIELD // UPLINK", 
    page_icon="✊", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. CSS STYLING (The Resistance Theme)
def local_css():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Oswald:wght@500;700&family=Share+Tech+Mono&display=swap');
        
        /* GLOBAL RESET & BACKGROUND */
        .stApp {
            background-color: #050505;
            background-image: 
                linear-gradient(rgba(0, 0, 0, 0.7), rgba(0, 0, 0, 0.7)),
                radial-gradient(#222 1px, transparent 1px);
            background-size: cover, 20px 20px;
            color: #e0e0e0;
        }
        
        /* TYPOGRAPHY */
        h1, h2, h3 { 
            font-family: 'Oswald', sans-serif !important; 
            color: #FF3333 !important; /* Alarm Red */
            text-transform: uppercase; 
            letter-spacing: 2px;
            text-shadow: 2px 2px 0px #000;
        }
        
        p, div, label, li, span, button, input { 
            font-family: 'Share Tech Mono', monospace !important; 
        }
        
        /* SIDEBAR STATS */
        .sidebar-stat {
            border-left: 2px solid #FF3333;
            padding-left: 10px;
            margin-bottom: 10px;
            color: #888;
        }

        /* BUTTONS (Brutalist Style) */
        div.stButton > button {
            background-color: transparent; 
            border: 2px solid #FF3333; 
            color: #FF3333 !important;
            border-radius: 0px; 
            text-transform: uppercase; 
            font-weight: bold;
            padding: 10px 20px;
            transition: all 0.3s ease;
            box-shadow: 4px 4px 0px #111;
        }
        div.stButton > button:hover {
            background-color: #FF3333; 
            color: #000 !important;
            box-shadow: 0 0 10px #FF3333;
            transform: translate(-2px, -2px);
        }
        
        /* TABS */
        .stTabs [data-baseweb="tab-list"] {
            gap: 5px;
            border-bottom: 1px solid #333;
        }
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            background-color: #0a0a0a;
            border: 1px solid #333;
            border-bottom: none;
            border-radius: 0px;
            color: #666;
        }
        .stTabs [data-baseweb="tab"][aria-selected="true"] {
            background-color: #FF3333;
            color: black;
            font-weight: bold;
        }

        /* METRICS */
        div[data-testid="stMetricValue"] {
            font-size: 3rem !important;
            color: #FFF !important;
            font-family: 'Oswald', sans-serif !important;
        }
        
        /* RED ALERT FLASH ANIMATION */
        .red-alert { animation: flash-red 0.5s ease-in-out; }
        @keyframes flash-red { 
            0% { background-color: rgba(255, 0, 0, 0.5); } 
            100% { background-color: transparent; } 
        }

        /* SCANLINE */
        .scanline {
            width: 100%;
            height: 100px;
            z-index: 99999;
            background: linear-gradient(0deg, rgba(0,0,0,0) 0%, rgba(255, 50, 50, 0.05) 50%, rgba(0,0,0,0) 100%);
            opacity: 0.1;
            position: fixed;
            bottom: 100%;
            left: 0;
            animation: scanline 8s linear infinite;
            pointer-events: none;
        }
        @keyframes scanline {
            0% { bottom: 100%; }
            100% { bottom: -100%; }
        }
    </style>
    <div class="scanline"></div>
    """, unsafe_allow_html=True)

local_css()

# CONFIGURATION PATHS
MODEL_PATH = os.path.join(parent_dir, 'models/audio_classifier.h5')
ENCODER_PATH = os.path.join(parent_dir, 'models/label_encoder.pkl')
UPLOAD_DIR = os.path.join(parent_dir, 'data/uploads')
os.makedirs(UPLOAD_DIR, exist_ok=True)

# HELPER: FEATURE EXTRACTION (LOCAL)
# Defined here to ensure Main App works independently of Preprocessing changes
def extract_features(file_path):
    try:
        # Load audio with librosa (Kaiser Fast for speed)
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
        # Standard MFCC extraction (40 features)
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        # Average across time to get shape (40,)
        mfccs_scaled = np.mean(mfccs.T, axis=0)
        return mfccs_scaled
    except Exception as e:
        return None

# SIDEBAR: SYSTEM STATUS
with st.sidebar:
    st.markdown("## // SYSTEM_LOG")
    st.code(f"""
    [LOC]: KIGALI_NODE
    [USR]: ANONYMOUS
    [NET]: ENCRYPTED
    [TIME]: {time.strftime('%H:%M:%S')}
    """)
    st.markdown("---")
    
    if os.path.exists(MODEL_PATH):
        st.markdown("<div class='sidebar-stat' style='border-color: #33FF33; color: #33FF33;'>:: NEURAL_NET_ACTIVE ::</div>", unsafe_allow_html=True)
        st.caption(f"ID: {os.path.basename(MODEL_PATH)}")
    else:
        st.markdown("<div class='sidebar-stat' style='border-color: #FF3333; color: #FF3333;'>:: NEURAL_NET_OFFLINE ::</div>", unsafe_allow_html=True)
        
    st.markdown("### // DIRECTIVES")
    st.info("""
    1. RECORD ENVIRONMENT
    2. UPLOAD TO SHIELD
    3. EVADE OR ASSEMBLE
    """)
    st.markdown("---")
    st.caption("CIVILIAN_SHIELD v2.5 // STABLE")

# HEADER
col1, col2 = st.columns([3, 1])
with col1:
    st.markdown("""
    ```text
       ___ _____ _   _ ___ _    ___   _   _  _   
      / __|_   _| | | |_ _| |  |_ _| /_\ | \| |  
     | (__  | | | |_| || || |__ | | / _ \| .` |  
      \___|___|_|\___/|___|____|___/_/ \_\_|\_|  
    ```
    """)
    st.title("ACOUSTIC COUNTER-SURVEILLANCE")
with col2:
    st.markdown("""
    <div style="text-align: right; color: #FF3333; border: 1px solid #FF3333; padding: 10px; margin-top: 20px;">
    <strong>STATUS: LISTENING</strong><br>
    <span style="font-size: 10px;">FREQ: 20Hz - 20kHz</span>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# MAIN TABS
tab1, tab2, tab3 = st.tabs(["// THREAT_SCAN", "// INTEL_GRID", "// HIVE_LEARN"])

# TAB 1: THREAT SCAN (Prediction)
with tab1:
    st.subheader(">> INITIATE AUDIO ANALYSIS")
    st.markdown("Upload environmental audio. System will classify: **SIREN**, **GUNSHOT**, or **CIVILIAN NOISE**.")
    
    col_a, col_b = st.columns([2, 1])
    with col_a:
        uploaded_file = st.file_uploader("DROP ENCRYPTED PACKET (.wav / .mp3)", type=["wav", "mp3"])

    if uploaded_file:
        # Save temp file
        temp_path = "temp_audio.wav"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        with col_b:
            st.write(":: PACKET RECEIVED ::")
            st.audio(uploaded_file)

        if st.button("EXECUTE ANALYSIS ALGORITHM"):
            # Progress bar for effect
            progress_bar = st.progress(0)
            status_text = st.empty()
            steps = ["Loading Weights...", "Extracting MFCC...", "Running Inference..."]
            for i, step in enumerate(steps):
                status_text.text(f">> {step}")
                time.sleep(0.1)
                progress_bar.progress((i + 1) * 33)
            
            try:
                # PREDICTION
                label, confidence, probs = predict_audio(temp_path, MODEL_PATH, ENCODER_PATH)
                st.markdown("---")
                
                # RESULT DISPLAY
                res_col1, res_col2 = st.columns([2, 2])
                threats = ['siren', 'gun_shot']
                
                with res_col1:
                    st.markdown("### // CLASSIFICATION")
                    if label in threats:
                        # INJECT RED FLASH SCRIPT
                        st.markdown(f"<script>document.querySelector('.stApp').classList.add('red-alert');</script>", unsafe_allow_html=True)
                        st.markdown(f"<h1 style='color:red; border: 3px solid red; padding: 10px; display: inline-block;'>{label.upper()}</h1>", unsafe_allow_html=True)
                        st.error("!!! IMMINENT THREAT DETECTED !!!")
                    else:
                        st.markdown(f"<h1 style='color:#33ff33;'>{label.upper()}</h1>", unsafe_allow_html=True)
                        st.success("STATUS: SAFE / CIVILIAN ACTIVITY")

                with res_col2:
                    st.markdown("### // PROBABILITY_MATRIX")
                    st.metric("CONFIDENCE", f"{confidence*100:.1f}%")

            except Exception as e:
                st.error(f"SYSTEM FAILURE: {e}")
            
            # Cleanup
            if os.path.exists(temp_path):
                os.remove(temp_path)

# TAB 2: INTEL GRID (Visualization)
with tab2:
    st.subheader(">> TRAINING DATA MANIFEST")
    try:
        csv_path = os.path.join(parent_dir, 'data/train/metadata/UrbanSound8K.csv')
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            st.dataframe(df.head(5), use_container_width=True)
            
            st.markdown("---")
            c1, c2 = st.columns(2)
            
            with c1:
                st.markdown("**:: CLASS DISTRIBUTION ::**")
                st.bar_chart(df['class'].value_counts(), color="#FF3333")
            
            with c2:
                st.markdown("**:: AUDIO LENGTH SCATTER ::**")
                st.scatter_chart(df[['end', 'start']], color="#33FF33")
        else:
            st.warning(":: MANIFEST NOT FOUND ::")
    except Exception as e:
        st.error(f"DATA CORRUPTION: {e}")

# TAB 3: HIVE LEARN (Real Retraining)
with tab3:
    st.write("#### `// DECENTRALIZED_LEARNING`")
    st.info("TEACH THE MACHINE. UPLOAD BATCH DATA + ASSIGN SHARED LABEL.")
    
    col_input, col_label = st.columns([2, 1])
    
    with col_input:
        new_files = st.file_uploader("UPLOAD_DATA_BATCH (.wav)", type=['wav', 'mp3'], accept_multiple_files=True)
        
    with col_label:
        # Load label options from Encoder if it exists, otherwise default list
        encoder = None
        if os.path.exists(ENCODER_PATH):
            with open(ENCODER_PATH, 'rb') as f:
                encoder = pickle.load(f)
            class_options = list(encoder.classes_)
        else:
            class_options = ["siren", "gun_shot", "dog_bark", "drilling", "engine_idling", "children_playing", "street_music", "air_conditioner", "jackhammer", "car_horn"]
            
        selected_label = st.selectbox("ASSIGN_BATCH_SIGNATURE", class_options)
    
    if st.button(">> EXECUTE_WEIGHT_UPDATE <<"):
        if new_files and encoder:
            terminal = st.empty()
            bar = st.progress(0)
            
            X_batch = []
            y_batch = []
            
            terminal.code(f">> INGESTING {len(new_files)} PACKETS...", language="bash")
            
            # LOOP THROUGH NEW FILES
            for i, file in enumerate(new_files):
                # Save to temp location
                temp_retrain_path = os.path.join(UPLOAD_DIR, f"temp_{i}.wav")
                with open(temp_retrain_path, "wb") as f:
                    f.write(file.getbuffer())
                
                # Extract Features
                features = extract_features(temp_retrain_path)
                
                if features is not None:
                    X_batch.append(features)
                    
                    # ONE-HOT ENCODING
                    try:
                        # 1. Turn Label into Index (e.g. "Siren" -> 8)
                        y_numeric = encoder.transform([selected_label])[0]
                        # 2. Turn Index into Vector (e.g. 8 -> [0,0,0,0,0,0,0,0,1,0])
                        y_encoded = to_categorical(y_numeric, num_classes=len(class_options))
                        y_batch.append(y_encoded)
                    except Exception as e:
                        st.error(f"Encoding Error: {e}")

                # Cleanup
                if os.path.exists(temp_retrain_path):
                    os.remove(temp_retrain_path)
                
                # Update UI
                bar.progress(int((i / len(new_files)) * 50))

            # TRAIN ON BATCH
            if len(X_batch) > 0:
                terminal.code(f">> BACKPROPAGATING ERROR ON {len(X_batch)} SAMPLES...", language="bash")
                
                try:
                    # Convert lists to Numpy Arrays
                    X_train_new = np.array(X_batch)
                    y_train_new = np.array(y_batch)
                    
                    # Call Model Training (Fine-Tuning)
                    train_and_save_model(X_train_new, y_train_new, save_path=MODEL_PATH, epochs=1)
                    
                    bar.progress(100)
                    terminal.code(">> MODEL_SAVED. INTELLIGENCE_EXPANDED.", language="bash")
                    st.success(f"SUCCESS: System learned '{selected_label.upper()}' from {len(new_files)} new files.")
                    st.balloons()
                except Exception as e:
                    terminal.code(f">> FATAL ERROR: {e}", language="bash")
            else:
                st.error("ERROR: No valid audio features could be extracted.")
        elif not encoder:
            st.error("ERROR: Label Encoder not found. Cannot retrain.")
        else:
            st.warning(">> AWAITING_DATA_INPUT")