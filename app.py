import streamlit as st 
import torch 
import torch.nn as nn 
from torchvision import models, transforms 
from PIL import Image 
from pathlib import Path 
 
 
# ============================================================ 
# SETTINGS 
# ============================================================ 
 
MODEL_PATH = Path("best_model.pth") 
 
CLASSES = [ 
    "black", 
    "blue", 
    "red", 
    "white", 
    "yellow" 
] 
 
BIN_INFO = { 
    "black": { 
        "name": "Other Waste", 
        "color": "⚫ Black" 
    }, 
    "blue": { 
        "name": "Glass Waste", 
        "color": "🔵 Blue" 
    }, 
    "red": { 
        "name": "Sharps", 
        "color": "🔴 Red" 
    }, 
    "white": { 
        "name": "Chemical Waste", 
        "color": "⚪ White" 
    }, 
    "yellow": { 
        "name": "Biological Waste", 
        "color": "🟡 Yellow" 
    } 
} 
 
 
# ============================================================ 
# PAGE CONFIGURATION 
# ============================================================ 
 
st.set_page_config( 
    page_title="BioSort360", 
    page_icon="♻️", 
    layout="centered" 
) 
 
 
# ============================================================ 
# CUSTOM CSS 
# ============================================================ 
 
st.markdown(""" 
<style> 
 
.main { 
    background-color: #eef5f9; 
} 
 
.title { 
    text-align: center; 
    font-size: 42px; 
    font-weight: bold; 
    color: #1f4d3a; 
} 
 
.subtitle { 
    text-align: center; 
    font-size: 20px; 
    color: #555555; 
    margin-bottom: 30px; 
} 
 
.result-box { 
    padding: 25px; 
    border-radius: 15px; 
    background-color: #ffffff; 
    text-align: center; 
    box-shadow: 0px 4px 15px rgba(0,0,0,0.10); 
    margin-top: 20px; 
} 
 
.prediction { 
    font-size: 32px; 
    font-weight: bold; 
} 
 
.category { 
    font-size: 22px; 
    margin-top: 10px; 
} 
 
.confidence { 
    font-size: 20px; 
    margin-top: 15px; 
} 
 
.footer { 
    text-align: center; 
    color: #777777; 
    margin-top: 40px; 
} 
 
</style> 
""", unsafe_allow_html=True) 
 
 
# ============================================================ 
# HEADER 
# ============================================================ 
 
st.markdown( 
    '<div class="title">♻️ BioSort360</div>', 
    unsafe_allow_html=True 
) 
 
st.markdown( 
    '<div class="subtitle">' 
    'Medical Waste Classification System' 
    '</div>', 
    unsafe_allow_html=True 
) 
 
 
st.write( 
    "Upload an image of medical waste " 
    "will predict the appropriate waste category." 
) 
 
 
# ============================================================ 
# LOAD MODEL 
# ============================================================ 
 
@st.cache_resource 
def load_model(): 
 
    device = torch.device( 
        "cuda" if torch.cuda.is_available() 
        else "cpu" 
    ) 
 
    model = models.mobilenet_v3_small( 
        weights=None 
    ) 
 
    input_features = model.classifier[-1].in_features 
 
    model.classifier[-1] = nn.Linear( 
        input_features, 
        len(CLASSES) 
    ) 
 
    model.load_state_dict( 
        torch.load( 
            MODEL_PATH, 
            map_location=device 
        ) 
    ) 
 
    model = model.to(device) 
 
    model.eval() 
 
    return model, device 
 
 
model, device = load_model() 
 
 
# ============================================================ 
# IMAGE TRANSFORMATION 
# ============================================================ 
 
transform = transforms.Compose([ 
 
    transforms.Resize( 
        (224, 224) 
    ), 
 
    transforms.ToTensor(), 
 
    transforms.Normalize( 
        mean=[0.485, 0.456, 0.406], 
 
        std=[0.229, 0.224, 0.225] 
    ) 
]) 
 
 
# ============================================================ 
# IMAGE UPLOAD 
# ============================================================ 
 
uploaded_file = st.file_uploader( 
 
    "📷 Upload Waste Image", 
 
    type=[ 
        "jpg", 
        "jpeg", 
        "png" 
    ] 
) 
 
 
# ============================================================ 
# PREDICTION 
# ============================================================ 
 
if uploaded_file is not None: 
 
    image = Image.open( 
        uploaded_file 
    ).convert("RGB") 
 
 
    st.image( 
        image, 
        caption="Uploaded Waste Image", 
        width=400 
    ) 
 
 
    if st.button( 
        "🔍 Classify Waste", 
        use_container_width=True 
    ): 
 
        with st.spinner( 
            "AI is analyzing the image..." 
        ): 
 
            input_tensor = transform( 
                image 
            ).unsqueeze(0).to(device) 
 
 
            with torch.no_grad(): 
 
                output = model( 
                    input_tensor 
                ) 
 
                probabilities = torch.softmax( 
                    output, 
                    dim=1 
                ) 
 
 
                confidence, prediction = torch.max( 
                    probabilities, 
                    dim=1 
                ) 
 
 
            predicted_class = CLASSES[ 
                prediction.item() 
            ] 
 
            confidence_value = ( 
                confidence.item() * 100 
            ) 
 
 
        info = BIN_INFO[ 
            predicted_class 
        ] 
 
 
        # ==================================================== 
        # RESULT 
        # ==================================================== 
 
        st.markdown( 
            '<div class="result-box">', 
            unsafe_allow_html=True 
        ) 
 
 
        st.markdown( 
            f'<div class="prediction">' 
            f'{info["color"]}' 
            f'</div>', 
            unsafe_allow_html=True 
        ) 
 
 
        st.markdown( 
            f'<div class="category">' 
            f'<b>{info["name"]}</b>' 
            f'</div>', 
            unsafe_allow_html=True 
        ) 
 
 
        st.markdown( 
            f'<div class="confidence">' 
            f'AI Confidence: ' 
            f'<b>{confidence_value:.2f}%</b>' 
            f'</div>', 
            unsafe_allow_html=True 
        ) 
 
 
        st.markdown( 
            '</div>', 
            unsafe_allow_html=True 
        ) 
 
 
# ============================================================ 
# FOOTER 
# ============================================================ 
 
st.markdown( 
    '<div class="footer">' 
    'BioSort360 | Medical Waste Segregation' 
    '</div>', 
    unsafe_allow_html=True 
)