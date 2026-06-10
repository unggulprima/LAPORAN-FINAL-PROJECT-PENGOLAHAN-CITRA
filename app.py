import cv2
import numpy as np
import streamlit as st

# ====================================================================
# 1. FUNGSI PENGOLAHAN CITRA DIGITAL (CORE LOGIC)
# ====================================================================

def preprocess_image(image):
    """Mengubah citra ke grayscale dan menerapkan blur untuk mereduksi noise."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    return blurred

def get_document_contour(edged_image):
    """Mencari kontur berbentuk 4 sudut yang merepresentasikan kertas/buku."""
    contours, _ = cv2.findContours(edged_image.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # Urutkan kontur dari yang paling luas ke terkecil
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
    
    # Hitung luas total gambar sebagai batas ambang ukuran (threshold)
    total_area = edged_image.shape[0] * edged_image.shape[1]
    
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        
        # Validasi: Harus memiliki 4 sudut DAN ukurannya minimal 15% dari luas total gambar
        if len(approx) == 4 and cv2.contourArea(c) > (0.15 * total_area):
            return approx
    return None

def order_points(pts):
    """Mengurutkan 4 titik sudut secara konsisten: top-left, top-right, bottom-right, bottom-left."""
    pts = pts.reshape((4, 2))
    rect = np.zeros((4, 2), dtype="float32")
    
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    
    return rect

def perspective_transform(image, pts):
    """Meluruskan gambar dokumen miring menggunakan Warp Perspective (Bird's Eye View)."""
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    
    # Menghitung dimensi lebar baru dari hasil kalkulasi jarak Euclidean titik sudut
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    max_width = max(int(widthA), int(widthB))
    
    # Menghitung dimensi tinggi baru
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    max_height = max(int(heightA), int(heightB))
    
    # Matriks tujuan transformasi
    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]
    ], dtype="float32")
    
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (max_width, max_height))
    return warped

# ====================================================================
# 2. ANTARMUKA WEB INTERAKTIF (STREAMLIT UI)
# ====================================================================

st.set_page_config(page_title="SmartScan PCD", layout="wide")
st.title("📸 Aplikasi Scanner Dokumen Sederhana")
st.write("Final Project Pengolahan Citra Digital - Kelas 1243H")
st.write("**Oleh:** Unggul Prima Dhani (312210477) & Irfan Tarwin Suryadi (312210311)")

st.markdown("---")

# Input Citra Gambar melalui File Uploader
uploaded_file = st.file_uploader("Pilih atau Drop Foto Dokumen Anda (JPG/PNG)", type=["jpg", "jpeg", "png"]) 

if uploaded_file is not None:
    # Membaca data gambar upload ke dalam format OpenCV (NumPy array)
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_original = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_display = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB)
    
    # Menginisialisasi layout 3 kolom di Streamlit 
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("1. Gambar Asli")
        st.image(img_display, use_container_width=True) 
        
    # --- TAHAPAN PROSES PENGOLAHAN CITRA ---
    # 1. Preprocessing (Grayscale dan Gaussian Blur)
    preprocessed = preprocess_image(img_original)
    
    # 2. Edge Detection menggunakan Algoritma Canny
    edged = cv2.Canny(preprocessed, 75, 200)
    
    with col2:
        st.subheader("2. Deteksi Tepi (Canny)")
        st.image(edged, use_container_width=True, channels="GRAY") 
        
    # 3. Segmentasi Kontur untuk mencari 4 sudut fisik kertas
    doc_contour = get_document_contour(edged)
    
    # Evaluasi deteksi kontur
    if doc_contour is not None:
        # Jika berhasil mendeteksi 4 sudut objek utama, jalankan Warp Perspective 
        warped = perspective_transform(img_original, doc_contour)
        st.success("🎉 Dokumen berhasil dideteksi dan diluruskan otomatis!")
    else:
        # FALLBACK: Jika tidak kontras/terputus, gunakan seluruh gambar asli (mencegah hasil terpotong kecil)
        warped = img_original.copy()
        st.warning("⚠️ Sudut dokumen kurang kontras dari background. Menampilkan pemrosesan gambar penuh (Full Image).")
        
    # 4. Tahap Akhir: Thresholding efek Scanner digital
    warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    scan_bw = cv2.adaptiveThreshold(warped_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 10)
    
    with col3:
        st.subheader("3. Hasil Scan Akhir") 
        # Pilihan Filter Hasil Akhir secara Real-Time
        filter_option = st.radio("Pilih Efek Hasil:", ("Berwarna (Rapi)", "Hitam Putih (Scanner)"))
        
        if filter_option == "Berwarna (Rapi)":
            final_result = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
            st.image(final_result, use_container_width=True) 
        else:
            st.image(scan_bw, use_container_width=True, channels="GRAY") 