# 🎵 Music Recommendation System

A content-based Music Recommendation System built with **Python** and **Streamlit**. This web application suggests similar songs based on user selection, audio features, and track metadata using similarity algorithms (such as Cosine Similarity).

---

## 📌 Features

- 🎧 **Interactive UI**: Clean, intuitive interface powered by Streamlit.
- 🔍 **Song Selection**: Dropdown search to choose any track from the dataset.
- ⚡ **Real-time Recommendations**: Get instantaneous recommendations based on similarity metrics.
- 🖼️ **Album Artwork & Metadata**: Fetch album covers, artist names, and preview links.

---

## 🛠️ Tech Stack & Dependencies

- **Programming Language:** Python 3.8+
- **Frontend / UI Framework:** [Streamlit](https://streamlit.io/)
- **Data Handling:** Pandas, NumPy
- **Machine Learning / Similarity:** Scikit-Learn (Cosine Similarity, Nearest Neighbors)
- **Serialization:** Pickle / Joblib (for loading pre-calculated similarity matrices and dataframe artifacts)

---

## 📁 Repository Structure

```text
music-recommendation/
│
├── app.py                  # Main Streamlit web application script
├── df.pkl                  # Serialized DataFrame containing music metadata
├── similarity.pkl          # Serialized similarity matrix computed from audio features
├── requirements.txt        # Required Python packages
└── README.md               # Project documentation
```

---

## 🚀 Getting Started

Follow these instructions to get a copy of the project up and running on your local machine.

### Prerequisites

Ensure you have Python 3.8+ installed on your system. You can verify your installation by running:
```bash
python --version
```

### Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/Vishakh18/music-recommendation.git
   cd music-recommendation
   ```

2. **Create a Virtual Environment (Optional but Recommended)**
   ```bash
   # On macOS/Linux
   python -m venv venv
   source venv/bin/activate

   # On Windows
   python -m venv venv
   venv\Scripts\activate
   ```

3. **Install Dependencies**
   Install the required packages using pip:
   ```bash
   pip install -r requirements.txt
   ```

---

## 💻 How to Run

Launch the Streamlit web application by executing:

```bash
streamlit run app.py
```

Once executed, your default browser will open to `http://localhost:8501`.

---

## 🧠 How It Works

1. **Feature Extraction & Preprocessing**: Audio features (such as *danceability, energy, acousticness, tempo, valence*) or track metadata are processed and standardized.
2. **Similarity Computation**: High-dimensional vectors representing each track are compared using **Cosine Similarity**:
   $$	ext{Cosine Similarity}(A, B) = \frac{A \cdot B}{\|A\| \|B\|}$$
3. **Recommendation Pipeline**:
   - The user selects a song from the UI dropdown.
   - The app looks up the corresponding vector in the pre-computed `similarity.pkl` matrix.
   - The top $N$ tracks with the highest similarity scores are sorted and displayed to the user.

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/Vishakh18/music-recommendation/issues).

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.
