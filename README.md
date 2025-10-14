# Simulasi N-Benda (N-Body) dengan CUDA dan OpenGL

## Deskripsi Umum

Proyek ini merupakan implementasi dari simulasi N-benda (N-body) yang memvisualisasikan interaksi gravitasi antara ribuan partikel dalam ruang tiga dimensi. Perhitungan fisika yang intensif, khususnya kalkulasi gaya gravitasi antar partikel, diakselerasi secara masif menggunakan arsitektur komputasi paralel NVIDIA CUDA. Aspek visualisasi dan rendering grafis direalisasikan secara *real-time* dengan memanfaatkan pustaka grafis OpenGL.

Aplikasi ini menampilkan sebuah lubang hitam (black hole) masif di pusat sistem, yang menjadi sumber gaya gravitasi dominan, serta partikel-partikel yang mengorbit dan berinteraksi satu sama lain.

## Fitur Utama

- **Akselerasi GPU dengan CUDA**: Kernel CUDA kustom (`updateParticles`) digunakan untuk menghitung gaya gravitasi dan memperbarui posisi serta kecepatan setiap partikel secara paralel di GPU, memungkinkan simulasi dengan jumlah partikel yang besar (16.384 dalam konfigurasi saat ini).
- **Render Instancing**: OpenGL dimanfaatkan untuk merender semua partikel secara efisien menggunakan teknik *instanced rendering* (`glDrawElementsInstanced`), yang secara signifikan mengurangi jumlah panggilan `draw call` dan meningkatkan performa.
- **Siklus Hidup Partikel**: Partikel yang tertarik terlalu dekat ke lubang hitam (dianggap "terserap") akan didaur ulang dan diinisialisasi kembali di tepi sistem, menjaga jumlah partikel tetap konstan dan simulasi berjalan secara kontinu.
- **Kamera 3D Interaktif**: Implementasi kamera terbang (*fly-through camera*) yang memungkinkan pengguna untuk menavigasi dan mengamati simulasi dari berbagai sudut pandang.
- **Pencahayaan Dinamis**: Model pencahayaan Blinn-Phong sederhana diimplementasikan dalam *shader* GLSL untuk memberikan persepsi kedalaman dan volume pada partikel dan lubang hitam. Warna partikel juga berubah secara dinamis berdasarkan jaraknya dari pusat sistem.

## Struktur Proyek

```
.
├── Makefile              # Skrip untuk kompilasi proyek
├── README.md             # Dokumentasi proyek
├── include/              # File header (.h) untuk deklarasi kelas dan fungsi
│   ├── camera.h
│   ├── graphics.h
│   ├── particle.h
│   ├── simulation.h
│   ├── sphere.h
│   └── utils.h
├── shaders/              # Kode shader GLSL (.vert, .frag)
│   ├── instance.frag
│   ├── instance.vert
│   ├── simple.frag
│   └── simple.vert
└── src/                  # Kode sumber implementasi (.cpp, .cu)
    ├── camera.cpp
    ├── graphics.cpp
    ├── main.cpp
    ├── simulation.cu
    ├── sphere.cpp
    └── utils.cu
```

## Prasyarat

Untuk dapat mengkompilasi dan menjalankan proyek ini, sistem Anda harus memiliki:
- Kompilator C++ yang mendukung C++11 (misalnya, `g++`)
- NVIDIA CUDA Toolkit (untuk kompilator `nvcc`)
- Pustaka OpenGL
- Pustaka `GLEW` (The OpenGL Extension Wrangler Library)
- Pustaka `GLFW` (A multi-platform library for OpenGL)
- Pustaka `GLM` (OpenGL Mathematics)

### Instalasi Pustaka
- **Arch Linux**:
  ```bash
  sudo pacman -S base-devel cuda
  sudo pacman -S glew glfw-x11 glm
  ```

## Kompilasi dan Eksekusi

1.  **Kompilasi Proyek**:
    Buka terminal di direktori root proyek dan jalankan perintah `make`. `Makefile` akan secara otomatis mengkompilasi semua file sumber CUDA dan C++ dan menautkannya menjadi satu file biner.
    ```bash
    make
    ```

2.  **Menjalankan Simulasi**:
    Setelah kompilasi berhasil, jalankan file biner yang dihasilkan:
    ```bash
    ./n_body_simulation
    ```

3.  **Membersihkan Proyek**:
    Untuk menghapus semua file objek (`.o`) dan file biner yang telah dikompilasi, jalankan:
    ```bash
    make clean
    ```

## Detail Teknis

### Modul Simulasi (`simulation.cu`)

Inti dari simulasi berada pada kernel CUDA `updateParticles`. Untuk setiap partikel, kernel ini melakukan:
1.  Menghitung vektor gaya gravitasi yang diberikan oleh lubang hitam pusat.
2.  Melakukan iterasi melalui semua partikel lain dalam sistem untuk mengakumulasi total gaya gravitasi dari interaksi partikel-partikel.
3.  Untuk mencegah singularitas numerik (pembagian dengan nol) ketika partikel berada pada jarak yang sangat dekat, sebuah **faktor pelunakan (softening factor)** ditambahkan pada perhitungan jarak.
4.  Berdasarkan total gaya yang dihitung, akselerasi partikel ditentukan (F=ma).
5.  Posisi dan kecepatan partikel diperbarui menggunakan integrasi Euler sederhana.

### Modul Grafis (`graphics.cpp`)

Modul ini bertanggung jawab untuk semua operasi rendering menggunakan OpenGL.
- **Inisialisasi**: Membuat dan mengkompilasi program *shader* (satu untuk partikel via *instancing*, satu lagi untuk lubang hitam). Geometri bola (untuk partikel dan lubang hitam) dibuat dan diunggah ke VBO (Vertex Buffer Object) di GPU.
- **Rendering**: Pada setiap frame, posisi partikel yang telah diperbarui oleh modul simulasi disalin ke *buffer* VBO khusus untuk matriks transformasi instan. Warna setiap partikel juga dihitung di CPU berdasarkan jaraknya dari pusat dan diunggah ke *buffer* instan lainnya. `glDrawElementsInstanced` kemudian dipanggil untuk merender semua partikel dalam satu panggilan.

## Kontrol

- **Gerakan Kamera**:
  - `W`: Maju
  - `S`: Mundur
  - `A`: Kiri
  - `D`: Kanan
  - `Spasi`: Naik
  - `Shift Kiri`: Turun
- **Orientasi Kamera**: Gerakkan *mouse* untuk melihat sekeliling.
- **Keluar**: Tekan tombol `ESC` untuk menutup aplikasi.
