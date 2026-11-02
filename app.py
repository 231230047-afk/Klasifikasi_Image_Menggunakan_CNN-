
import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
import matplotlib.pyplot as plt

from pathlib import Path
from PIL import Image, ImageOps

# ==========================================
# 1. KONFIGURASI APLIKASI
# ==========================================

st.set_page_config(
    page_title="Oceanic Life Classification",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "output"

MODEL_PATH = OUT_DIR / "final_oceanic_model.keras"
CLASS_PATH = OUT_DIR / "class_names.json"

IMG_SIZE = (160, 160)

# ==========================================
# 2. STYLE DASHBOARD
# ==========================================

st.markdown("""
<style>
    .stApp {
        background-color: #F4F8FB;
    }

    [data-testid="stSidebar"] {
        background-color: #E7F1F8;
    }

    .main-title {
        font-size: 34px;
        font-weight: 800;
        color: #075985;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 15px;
        color: #64748B;
        margin-bottom: 25px;
    }

    .info-card {
        background: white;
        padding: 20px;
        border-radius: 14px;
        border: 1px solid #DCE8F0;
        box-shadow: 0 3px 10px rgba(0,0,0,0.04);
    }

    .info-label {
        font-size: 14px;
        color: #64748B;
    }

    .info-value {
        font-size: 27px;
        font-weight: bold;
        color: #0F766E;
    }

    div.stButton > button {
        background-color: #087E8B;
        color: white;
        border-radius: 9px;
        border: none;
        padding: 10px 20px;
        font-weight: bold;
    }

    div.stButton > button:hover {
        background-color: #075985;
        color: white;
    }

    h1, h2, h3 {
        color: #075985;
    }
</style>
""", unsafe_allow_html=True)


# ==========================================
# 3. MEMUAT MODEL DAN CLASS NAMES
# ==========================================

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model tidak ditemukan: {MODEL_PATH}"
        )

    return tf.keras.models.load_model(MODEL_PATH)


@st.cache_data
def load_classes():
    if not CLASS_PATH.exists():
        raise FileNotFoundError(
            f"File class_names.json tidak ditemukan: {CLASS_PATH}"
        )

    with open(CLASS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


try:
    model = load_model()
    CLASS_NAMES = load_classes()

except Exception as e:
    st.error(f"Gagal memuat model atau kelas: {e}")
    st.info(
        "Pastikan notebook sudah selesai dijalankan dan "
        "folder output berisi model serta class_names.json."
    )
    st.stop()


# ==========================================
# 4. MEMUAT DATA HASIL PENELITIAN
# ==========================================

def load_json(filename):
    path = OUT_DIR / filename

    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    return None


def load_csv(filename):
    path = OUT_DIR / filename

    if path.exists():
        return pd.read_csv(path)

    return None


metrics = load_json("metrics.json")
history = load_json("history.json")
config = load_json("config.json")

labels_df = load_csv("labels.csv")
metrics_df = load_csv("metrics_table.csv")
counts_df = load_csv("class_counts.csv")


# ==========================================
# 5. FUNGSI PREDIKSI GAMBAR
# ==========================================

def preprocess_image(image):
    image = ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    image = image.resize(IMG_SIZE)

    image_array = np.asarray(
        image,
        dtype=np.float32
    )

    # Sesuai preprocessing notebook:
    # gambar dibagi 255 sebelum masuk model.
    image_array = image_array / 255.0

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    return image_array


def predict_image(image):
    image_array = preprocess_image(image)

    probabilities = model.predict(
        image_array,
        verbose=0
    )[0]

    predicted_idx = int(np.argmax(probabilities))

    predicted_class = CLASS_NAMES[predicted_idx]

    confidence = float(probabilities[predicted_idx])

    sorted_indices = np.argsort(probabilities)[::-1]

    results = []

    for idx in sorted_indices:
        results.append({
            "Kelas": CLASS_NAMES[idx],
            "Probabilitas": float(probabilities[idx])
        })

    return predicted_class, confidence, results


# ==========================================
# 6. SIDEBAR
# ==========================================

st.sidebar.title("🌊 Oceanic Life")
st.sidebar.caption("CNN Image Classification Dashboard")

menu = st.sidebar.radio(
    "MENU NAVIGASI",
    [
        "Dashboard",
        "Prediksi Gambar",
        "Distribusi Dataset",
        "Evaluasi Model",
        "Tentang Aplikasi"
    ]
)

st.sidebar.divider()

st.sidebar.markdown("### Informasi Model")
st.sidebar.write("**Metode:** CNN")
st.sidebar.write("**Arsitektur:** MobileNetV2")
st.sidebar.write("**Ukuran gambar:** 160 × 160")
st.sidebar.write("**Jumlah kelas:**", len(CLASS_NAMES))

st.sidebar.divider()
st.sidebar.caption("Oceanic Life Classification Project")


# ==========================================
# 7. HALAMAN DASHBOARD
# ==========================================

if menu == "Dashboard":

    st.markdown(
        '<div class="main-title">🌊 Oceanic Life Classification</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">Sistem klasifikasi citra organisme laut menggunakan '
        'Convolutional Neural Network (CNN) dengan arsitektur MobileNetV2.</div>',
        unsafe_allow_html=True
    )

    total_images = (
        len(labels_df)
        if labels_df is not None
        else 0
    )

    accuracy = (
        metrics.get("accuracy", 0)
        if metrics
        else 0
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Total Gambar",
            f"{total_images:,}"
        )

    with col2:
        st.metric(
            "Jumlah Kelas",
            len(CLASS_NAMES)
        )

    with col3:
        st.metric(
            "Akurasi Model",
            f"{accuracy:.2%}"
        )

    with col4:
        st.metric(
            "Ukuran Input",
            "160 × 160"
        )

    st.divider()

    st.subheader("Mengenal Organisme Laut")

    st.write(
        "Aplikasi ini digunakan untuk mengidentifikasi jenis organisme "
        "laut berdasarkan gambar yang diunggah. Model CNN akan memproses "
        "gambar dan memberikan hasil prediksi berdasarkan kelas yang "
        "telah dipelajari selama proses training."
    )

    # Informasi kelas
    st.subheader("Kategori Organisme Laut")

    descriptions = {
        "other": "Organisme atau objek laut lainnya.",
        "fish": "Berbagai jenis ikan yang hidup di perairan.",
        "seal": "Mamalia laut seperti anjing laut.",
        "coral": "Terumbu karang yang hidup di ekosistem laut.",
        "crab": "Kepiting dan jenis hewan laut bercapit.",
        "shark": "Ikan hiu yang hidup di berbagai perairan.",
        "turtle": "Penyu atau kura-kura laut.",
        "octopus": "Gurita yang memiliki delapan lengan.",
        "starfish": "Bintang laut yang hidup di dasar laut.",
        "jellyfish": "Ubur-ubur yang memiliki tubuh transparan."
    }

    cols = st.columns(2)

    for i, class_name in enumerate(CLASS_NAMES):
        with cols[i % 2]:
            st.markdown(
                f"""
                <div class="info-card" style="margin-bottom:12px;">
                    <b style="color:#087E8B;font-size:17px;">
                        {class_name.replace("_", " ").title()}
                    </b>
                    <p style="color:#64748B;margin-bottom:0;">
                        {descriptions.get(class_name, "Kategori organisme laut.")}
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )

    st.divider()

    st.subheader("Grafik Training Model")

    curve_path = OUT_DIR / "4_training_curves.png"

    if curve_path.exists():
        st.image(
            str(curve_path),
            use_container_width=True
        )
    else:
        st.info("Grafik training belum tersedia.")


# ==========================================
# 8. HALAMAN PREDIKSI GAMBAR
# ==========================================

elif menu == "Prediksi Gambar":

    st.title("🔍 Prediksi Organisme Laut")

    st.write(
        "Unggah gambar organisme laut untuk mengetahui "
        "hasil klasifikasi menggunakan model CNN."
    )

    uploaded_file = st.file_uploader(
        "Pilih gambar organisme laut",
        type=["jpg", "jpeg", "png", "webp", "bmp"]
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file)
        image = ImageOps.exif_transpose(image).convert("RGB")

        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("Gambar yang Diunggah")
            st.image(
                image,
                caption="Gambar input",
                use_container_width=True
            )

        with col2:
            st.subheader("Hasil Klasifikasi")

            if st.button(
                "🔎 Mulai Prediksi",
                use_container_width=True
            ):

                with st.spinner("Model sedang menganalisis gambar..."):

                    predicted_class, confidence, results = predict_image(image)

                st.success("Prediksi berhasil!")

                st.markdown("### Hasil Prediksi")

                st.markdown(
                    f"""
                    <div class="info-card">
                        <div class="info-label">Organisme terdeteksi</div>
                        <div class="info-value">
                            {predicted_class.replace("_", " ").title()}
                        </div>
                        <br>
                        <div class="info-label">Tingkat keyakinan model</div>
                        <div class="info-value">{confidence:.2%}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.progress(confidence)

                st.caption(
                    "Probabilitas menunjukkan tingkat keyakinan relatif "
                    "model, bukan jaminan bahwa prediksi benar."
                )

                st.markdown("### Top 5 Prediksi")

                top_results = results[:5]

                for item in top_results:
                    st.write(
                        f"**{item['Kelas'].title()}** "
                        f"— {item['Probabilitas']:.2%}"
                    )

                    st.progress(item["Probabilitas"])

                result_df = pd.DataFrame(top_results)

                st.download_button(
                    "⬇️ Download Hasil Prediksi",
                    data=result_df.to_csv(index=False).encode("utf-8"),
                    file_name="hasil_prediksi_organisme_laut.csv",
                    mime="text/csv"
                )


# ==========================================
# 9. HALAMAN DISTRIBUSI DATASET
# ==========================================

elif menu == "Distribusi Dataset":

    st.title("📊 Distribusi Dataset")

    if labels_df is not None and "label" in labels_df.columns:

        counts = (
            labels_df["label"]
            .value_counts()
            .reindex(CLASS_NAMES, fill_value=0)
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Total Gambar", f"{counts.sum():,}")

        with col2:
            st.metric("Kelas Terbanyak", counts.idxmax())

        with col3:
            st.metric("Kelas Tersedikit", counts.idxmin())

        st.subheader("Jumlah Gambar per Kelas")

        chart_df = pd.DataFrame({
            "Kelas": counts.index,
            "Jumlah Gambar": counts.values
        })

        st.bar_chart(
            chart_df.set_index("Kelas"),
            color="#087E8B"
        )

        st.subheader("Detail Jumlah Data")

        st.dataframe(
            chart_df,
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "⬇️ Download Data Distribusi",
            data=chart_df.to_csv(index=False).encode("utf-8"),
            file_name="distribusi_dataset.csv",
            mime="text/csv"
        )

    else:
        st.warning(
            "File output/labels.csv belum ditemukan. "
            "Jalankan notebook sampai selesai terlebih dahulu."
        )

    st.divider()

    st.subheader("Distribusi Dataset dari Notebook")

    distribution_path = OUT_DIR / "1_class_distribution.png"

    if distribution_path.exists():
        st.image(
            str(distribution_path),
            use_container_width=True
        )


# ==========================================
# 10. HALAMAN EVALUASI MODEL
# ==========================================

elif menu == "Evaluasi Model":

    st.title("📈 Evaluasi Model CNN")

    if metrics:

        accuracy = metrics.get("accuracy", 0)
        macro = metrics.get("macro", {})
        weighted = metrics.get("weighted", {})

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Accuracy", f"{accuracy:.2%}")

        with col2:
            st.metric(
                "Precision Macro",
                f"{macro.get('precision', 0):.2%}"
            )

        with col3:
            st.metric(
                "Recall Macro",
                f"{macro.get('recall', 0):.2%}"
            )

        with col4:
            st.metric(
                "F1-Score Macro",
                f"{macro.get('f1', 0):.2%}"
            )

        st.divider()

        st.subheader("Perbandingan Macro dan Weighted")

        comparison = pd.DataFrame({
            "Jenis": ["Precision", "Recall", "F1-Score"],
            "Macro": [
                macro.get("precision", 0),
                macro.get("recall", 0),
                macro.get("f1", 0)
            ],
            "Weighted": [
                weighted.get("precision", 0),
                weighted.get("recall", 0),
                weighted.get("f1", 0)
            ]
        })

        st.bar_chart(
            comparison.set_index("Jenis"),
            color=["#087E8B", "#F4A261"]
        )

    else:
        st.warning("File metrics.json belum ditemukan.")

    st.divider()

    st.subheader("Classification Report")

    if metrics_df is not None:
        st.dataframe(
            metrics_df,
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("File metrics_table.csv belum tersedia.")

    st.divider()

    st.subheader("Confusion Matrix")

    cm_path = OUT_DIR / "5_confusion_matrix.png"

    if cm_path.exists():
        st.image(
            str(cm_path),
            use_container_width=True
        )
    else:
        cm_file = OUT_DIR / "confusion_matrix.npy"

        if cm_file.exists():
            cm = np.load(cm_file)

            fig, ax = plt.subplots(figsize=(9, 7))

            im = ax.imshow(cm, cmap="Blues")
            fig.colorbar(im, ax=ax)

            ax.set_xticks(range(len(CLASS_NAMES)))
            ax.set_xticklabels(
                CLASS_NAMES,
                rotation=45,
                ha="right"
            )

            ax.set_yticks(range(len(CLASS_NAMES)))
            ax.set_yticklabels(CLASS_NAMES)

            ax.set_xlabel("Prediksi")
            ax.set_ylabel("Kelas Sebenarnya")
            ax.set_title("Confusion Matrix")

            st.pyplot(fig)
            plt.close(fig)

        else:
            st.info("Confusion matrix belum tersedia.")

    st.divider()

    st.subheader("Grafik Accuracy dan Loss")

    if history:

        if "accuracy" in history and "val_accuracy" in history:

            epoch_df = pd.DataFrame({
                "Epoch": range(1, len(history["accuracy"]) + 1),
                "Training Accuracy": history["accuracy"],
                "Validation Accuracy": history["val_accuracy"]
            })

            st.line_chart(
                epoch_df.set_index("Epoch")
            )

        if "loss" in history and "val_loss" in history:

            loss_df = pd.DataFrame({
                "Epoch": range(1, len(history["loss"]) + 1),
                "Training Loss": history["loss"],
                "Validation Loss": history["val_loss"]
            })

            st.line_chart(
                loss_df.set_index("Epoch")
            )

    else:
        curve_path = OUT_DIR / "4_training_curves.png"

        if curve_path.exists():
            st.image(str(curve_path), use_container_width=True)
        else:
            st.info("Data training history belum tersedia.")


# ==========================================
# 11. HALAMAN TENTANG APLIKASI
# ==========================================

elif menu == "Tentang Aplikasi":

    st.title("ℹ️ Tentang Aplikasi")

    st.markdown("""
    ### Oceanic Life Classification

    Aplikasi ini merupakan sistem klasifikasi citra organisme laut
    menggunakan metode **Convolutional Neural Network (CNN)**
    dengan arsitektur **MobileNetV2**.

    ### Tujuan Aplikasi

    Membantu mengenali kategori organisme laut berdasarkan gambar
    yang dimasukkan oleh pengguna.

    ### Metode yang Digunakan

    1. Pengumpulan dataset gambar organisme laut.
    2. Pemberian label menggunakan CLIP zero-shot dan koreksi label.
    3. Preprocessing dan resize gambar menjadi 160 × 160 piksel.
    4. Pembagian data training, validation, dan testing.
    5. Pelatihan model CNN menggunakan MobileNetV2.
    6. Evaluasi menggunakan accuracy, precision, recall, F1-score,
       dan confusion matrix.
    7. Implementasi model dalam aplikasi Streamlit.

    ### Kategori Klasifikasi
    """)

    for i, name in enumerate(CLASS_NAMES, start=1):
        st.write(f"{i}. {name.replace('_', ' ').title()}")

    st.divider()

    st.markdown("""
    **Teknologi:**
    - Python
    - TensorFlow / Keras
    - MobileNetV2
    - Streamlit
    - Pandas
    - NumPy
    - Matplotlib
    """)


# ==========================================
# 12. FOOTER
# ==========================================

st.divider()

st.markdown(
    """
    <div style="text-align:center;color:#64748B;padding:10px;">
        Oceanic Life Classification Dashboard<br>
        CNN - MobileNetV2 | Image Classification
    </div>
    """,
    unsafe_allow_html=True
)
