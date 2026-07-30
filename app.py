import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import json
import os
import base64

st.set_page_config(
    page_title="Sistem Pakar Deteksi Penyakit Daun Anggur",
    layout="wide"
)

def muat_gambar_base64(jalur_file):
    if os.path.exists(jalur_file):
        with open(jalur_file, "rb") as berkas_img:
            return base64.b64encode(berkas_img.read()).decode()
    return ""

file_banner = "banner_kebun.png.png" if os.path.exists("banner_kebun.png.png") else "banner_kebun.png"
file_logo = "logo_daun.png.png" if os.path.exists("logo_daun.png.png") else "logo_daun.png"

banner_b64 = muat_gambar_base64(file_banner)
logo_b64 = muat_gambar_base64(file_logo)

st.markdown(f"""
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    
    <style>
    .stApp {{ background-color: #fcfdfc; }}
    [data-testid="stHeader"] {{ background-color: rgba(0,0,0,0); }}
    
    .banner-premium-container {{
        background-image: linear-gradient(to right, rgba(255,255,255,1) 35%, rgba(255,255,255,0.8) 45%, rgba(255,255,255,0) 70%), url("data:image/png;base64,{banner_b64}");
        background-size: cover;
        background-position: right center;
        padding: 40px 50px;
        border-radius: 0px 0px 12px 12px;
        margin-top: -60px;
        margin-bottom: 35px;
        display: flex;
        align-items: center;
        min-height: 200px;
    }}
    
    .banner-content {{
        display: flex;
        align-items: center;
        max-width: 700px;
    }}
    
    .logo-img-style {{
        width: 75px;
        height: 75px;
        margin-right: 20px;
        object-fit: contain;
    }}
    
    p, span, label, .stMarkdown p {{
        font-size: 15.5px !important; 
        line-height: 1.6 !important;
        color: #444444;
    }}
    
    .info-box-premium {{
        background-color: #ffffff;
        padding: 22px;
        border-radius: 12px;
        border: 1px solid #e8f0e8;
        border-left: 5px solid #2e7d32;
    }}
    
    .judul-kolon-pakar {{
        color: #1b5e20 !important;
        font-size: 17px !important;
        font-weight: bold !important;
        margin-top: 5px !important;
        margin-bottom: 15px !important;
        display: flex;
        align-items: center;
        gap: 10px;
    }}
    .judul-kolon-pakar i {{
        color: #2e7d32 !important;
        font-size: 18px !important;
    }}
    
    .stMainBlockContainer {{ padding-top: 60px !important; padding-bottom: 40px !important; }}
    hr {{ margin-top: 30px !important; margin-bottom: 30px !important; border: 0.5px solid #edf4ed; }}
    
    .card-solusi-pakar {{
        background-color: #ffffff;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #edf3ed;
        height: 100%;
        box-shadow: 0 2px 6px rgba(0,0,0,0.01);
    }}
    .card-solusi-pakar h5 {{
        font-weight: bold !important;
        font-size: 15.5px !important;
        margin-top: 0px !important;
        margin-bottom: 12px !important;
        display: flex;
        align-items: center;
        gap: 10px;
    }}
    .card-solusi-pakar h5 i {{
        font-size: 16px !important;
    }}
    .card-solusi-pakar p {{
        font-size: 14.5px !important;
        color: #555555 !important;
        line-height: 1.6 !important;
    }}
    
    .stButton>button {{
        background-color: #2e7d32 !important;
        border: none !important;
        width: 100%;
        padding: 12px 24px !important;
        border-radius: 8px !important;
        transition: all 0.3s;
    }}
    .stButton>button, .stButton>button p, .stButton>button span, .stButton>button div {{
        color: white !important;
        font-size: 16px !important;
        font-weight: bold !important;
    }}
    .stButton>button:hover {{
        background-color: #1b5e20 !important;
        box-shadow: 0 4px 14px rgba(46,125,50,0.3);
    }}
    
    .stProgress > div > div > div > div {{ background-color: #2e7d32; }}
    </style>
""", unsafe_allow_html=True)

with open('rekomendasi.json', 'r') as f:
    DATABASE_REKOMENDASI = json.load(f)

class_names = ['Black Rot', 'ESCA', 'Healthy', 'Leaf Blight']

@st.cache_resource
def load_model_skripsi():
    model = models.resnet34(weights=None)
    target_modules = []
    for name, module in model.named_modules():
        if isinstance(module, nn.ReLU):
            target_modules.append((name, module))
    for name, module in target_modules:
        if '.' in name:
            parent_name, child_name = name.rsplit('.', 1)
            parent_module = dict(model.named_modules())[parent_name]
            setattr(parent_module, child_name, nn.Mish(inplace=True))
        else:
            setattr(model, name, nn.Mish(inplace=True))
            
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(class_names))
    
    model_path = 'model_resnet34_anggur_mish.pth'
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

try:
    model_terlatih = load_model_skripsi()
except Exception as e:
    st.error(f"Gagal me-load file model .pth: {e}")

logo_tag = f'<img src="data:image/png;base64,{logo_b64}" class="logo-img-style">' if logo_b64 else ""

st.markdown(f"""
    <div class="banner-premium-container">
        <div class="banner-content">
            {logo_tag}
            <div>
                <span style="color:#2e7d32; font-weight:bold; font-size:14px; letter-spacing:1.5px; display:block; margin-bottom:-2px;"><i class="fa-solid fa-leaf" style="color:#2e7d32; margin-right:5px;"></i> SISTEM PAKAR</span>
                <h1 style='margin:0; color:#1b5e20; font-size:34px; font-weight:bold; letter-spacing:-0.5px;'>Deteksi Penyakit Daun Anggur</h1>
                <p style='margin:6px 0 0 0; color:#444444; font-size:15px; font-weight:500; line-height:1.5 !important;'>
                    Sistem berbasis Artificial Intelligence untuk membantu identifikasi penyakit daun anggur secara cepat, akurat, dan terpercaya.
                </p>
            </div>
        </div>
    </div>
""", unsafe_allow_html=True)

kol_upload, kol_preview, kol_info = st.columns([1.1, 1.2, 1])

with kol_upload:
    st.markdown('<div class="judul-kolon-pakar"><i class="fa-solid fa-cloud-arrow-up"></i> Upload Gambar Daun Anggur</div>', unsafe_allow_html=True)
    st.write("Unggah citra daun anggur (format: JPG, PNG, JPEG)")
    file_gambar = st.file_uploader("Upload Box", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    st.caption("Maks. ukuran file: 5MB")

with kol_preview:
    st.markdown('<div class="judul-kolon-pakar"><i class="fa-solid fa-eye"></i> Preview Gambar</div>', unsafe_allow_html=True)
    if file_gambar is not None:
        image = Image.open(file_gambar).convert('RGB')
        st.image(image, width=320)
    else:
        st.markdown('<div style="background-color:#f9fbf9; border:1px dashed #ccc; border-radius:10px; height:155px; display:flex; justify-content:center; align-items:center; color:#777; font-size:15px;"><i class="fa-solid fa-images" style="margin-right:8px; color:#2e7d32;"></i> Belum ada gambar yang diunggah</div>', unsafe_allow_html=True)

with kol_info:
    st.markdown("""
        <div class="info-box-premium" style="padding: 16px 20px !important;">
            <h5 style="margin-top:0; color:#1b5e20; font-size:16px; font-weight:bold; display:flex; align-items:center; gap:8px;"><i class="fa-solid fa-circle-info" style="color:#2e7d32;"></i> Informasi</h5>
            <p style="margin:0; line-height:1.6; font-size:14.5px !important; color:#555555;">
                Pastikan gambar daun anggur diletakkan tepat di tengah kamera, mendapatkan pencahayaan yang cukup, and objek terfokus agar model mampu membaca bercak visual secara optimal.
            </p>
        </div>
    """, unsafe_allow_html=True)
    st.markdown("<div style='margin-top:14px;'></div>", unsafe_allow_html=True)
    tombol_prediksi = st.button("🔍 Prediksi Sekarang")

if file_gambar is not None and tombol_prediksi:
    st.markdown("<hr>", unsafe_allow_html=True)
    
    with st.spinner("Model AI sedang menganalisis karakteristik visual daun..."):
        transformasi = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        tensor_gambar = transformasi(image).unsqueeze(0)
        
        with torch.no_grad():
            outputs = model_terlatih(tensor_gambar)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0] * 100
            nilai_conf, index_prediksi = torch.max(probabilities, 0)
            
        key_penyakit = class_names[index_prediksi.item()]
        skor_persen = nilai_conf.item()
        list_probabilitas = probabilities.cpu().numpy()
        data_pakar = DATABASE_REKOMENDASI[key_penyakit]

    with st.container():
        st.markdown("<h3 style='color:#1b5e20; font-size:20px; font-weight:bold; margin-top:0px; margin-bottom:18px;'><i class='fa-solid fa-chart-bar' style='color:#2e7d32; margin-right:5px;'></i> Hasil Prediksi & Analisis Pakar</h3>", unsafe_allow_html=True)
        
        kol_res_img, kol_res_utama, kol_res_prob = st.columns([1, 1.1, 1.3])
        
        with kol_res_img:
            st.markdown("<b style='font-size:15px; display:block; margin-bottom:8px;'><i class='fa-solid fa-image' style='color:#2e7d32; margin-right:5px;'></i> Gambar Input</b>", unsafe_allow_html=True)
            st.image(image, width=240)
            
        with kol_res_utama:
            st.markdown("<b style='font-size:15px; display:block; margin-bottom:8px;'><i class='fa-solid fa-shield-halved' style='color:#2e7d32; margin-right:5px;'></i> Prediksi Utama</b>", unsafe_allow_html=True)
            st.markdown(f"""
                <div style="text-align:center; padding:20px; border:1px solid #e8f2e8; border-radius:10px; background-color:#ffffff; height:165px;">
                    <h2 style="color:#2e7d32; margin:0; font-size:23px; font-weight:bold;"><i class="fa-solid fa-circle-check" style="color:#2e7d32; margin-right:5px;"></i> {data_pakar['nama_penyakit']}</h2>
                    <p style="color:#666; font-style:italic; margin-top:4px; margin-bottom:8px; font-size:14px !important;">Penyebab: {data_pakar['penyebab']}</p>
                    <p style="margin-bottom:0px; font-size:13.5px; color:#555;">Tingkat Keyakinan</p>
                    <h1 style="color:#2e7d32; margin:0; font-size:36px; font-weight:bold;">{skor_persen:.2f}%</h1>
                </div>
            """, unsafe_allow_html=True)
            
        with kol_res_prob:
            st.markdown("<b style='font-size:15px; display:block; margin-bottom:8px;'><i class='fa-solid fa-clock' style='color:#2e7d32; margin-right:5px;'></i> Probabilitas Setiap Kelas</b>", unsafe_allow_html=True)
            for nama_kelas, prob in zip(class_names, list_probabilitas):
                nama_tampilan = DATABASE_REKOMENDASI[nama_kelas]['nama_penyakit']
                st.markdown(f"<div style='color: #1e1e1e !important; font-size:14.5px !important; font-weight:500; margin-bottom:1px;'><b>{nama_tampilan}</b> : {prob:.2f}%</div>", unsafe_allow_html=True)
                st.progress(float(prob / 100))

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<h4 style='color:#1b5e20; font-size:17.5px; font-weight:bold; margin-bottom:14px;'><i class='fa-solid fa-file-invoice' style='color:#2e7d32; margin-right:5px;'></i> Panduan Penanganan & Solusi Pakar Berdasarkan Rekomendasi:</h4>", unsafe_allow_html=True)
        kol_s1, kol_s2, kol_s3 = st.columns(3)
        
        with kol_s1:
            st.markdown(f"""
                <div class="card-solusi-pakar" style="border-top: 4px solid #d32f2f;">
                    <h5 style="color:#d32f2f !important;"><i class="fa-solid fa-triangle-exclamation" style="color:#d32f2f; margin-right:5px;"></i> Tindakan Darurat Pengendalian:</h5>
                    <p>{data_pakar['tindakan']}</p>
                </div>
            """, unsafe_allow_html=True)
            
        with kol_s2:
            st.markdown(f"""
                <div class="card-solusi-pakar" style="border-top: 4px solid #2e7d32;">
                    <h5 style="color:#2e7d32 !important;"><i class="fa-solid fa-shield-virus" style="color:#2e7d32; margin-right:5px;"></i> Langkah Pencegahan (Preventif):</h5>
                    <p>{data_pakar['pencegahan']}</p>
                </div>
            """, unsafe_allow_html=True)
            
        with kol_s3:
            st.markdown(f"""
                <div class="card-solusi-pakar" style="border-top: 4px solid #1976d2;">
                    <h5 style="color:#1976d2 !important;"><i class="fa-solid fa-flask-vial" style="color:#1976d2; margin-right:5px;"></i> Rekomendasi Agrokimia / Pengobatan:</h5>
                    <p>{data_pakar['obtain'] if 'obtain' in data_pakar else data_pakar['obat']}</p>
                </div>
            """, unsafe_allow_html=True)

elif file_gambar is not None and not tombol_prediksi:
    st.info("Silakan klik tombol '🔍 Prediksi Sekarang' di bawah kolom Informasi untuk memicu analisis sistem pakar.")
