import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
from geopy.geocoders import Nominatim
import os

st.set_page_config(
    page_title="NYC Restaurant Location Recommender",
    page_icon="🍽️",
    layout="wide"
)

st.title("🍽️ NYC Restaurant Location Suitability Recommender")
st.markdown("Select a cuisine type and enter any location in NYC to see how suitable it is for opening a new restaurant.")

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

CUISINE_OPTIONS = {
    "🍔 American": ("American", 0.005),
    "🥡 Chinese": ("Chinese", 0.005),
    "🍕 Pizza": ("Pizza", 0.005),
    "☕ Coffee/Tea": ("Coffee/Tea", 0.005),
    "🌯 Mexican": ("Mexican", 0.005),
    "🥐 Bakery Products/Desserts": ("Bakery Products/Desserts", 0.005),
    "🍗 Chicken": ("Chicken", 0.005),
    "🌮 Latin American": ("Latin American", 0.005),
    "🍩 Donuts": ("Donuts", 0.005),
    "🍝 Italian": ("Italian", 0.005),
    "🍣 Japanese": ("Japanese", 0.005),
    "🍛 Caribbean": ("Caribbean", 0.005),
    "🍔 Hamburgers": ("Hamburgers", 0.005),
    "🥪 Sandwiches": ("Sandwiches", 0.005),
    "🥤 Juice, Smoothies, Fruit Salads": ("Juice, Smoothies, Fruit Salads", 0.005),
    "🥘 Spanish": ("Spanish", 0.008),
    "🍜 Asian/Asian Fusion": ("Asian/Asian Fusion", 0.008),
    "🍲 Thai": ("Thai", 0.008),
    "🍛 Indian": ("Indian", 0.008),
    "🍲 Korean": ("Korean", 0.01),
}

BOROUGH_COORDS = {
    "Manhattan": (40.7831, -73.9712),
    "Brooklyn": (40.6782, -73.9442),
    "Queens": (40.7282, -73.7949),
    "Bronx": (40.8448, -73.8648),
    "Staten Island": (40.5795, -74.1502),
}

st.sidebar.header("⚙️ Configuration")

# ---------- 1. 菜系选择 ----------
st.sidebar.subheader("1️⃣ Cuisine Type")
selected_label = st.sidebar.selectbox("Select Cuisine", list(CUISINE_OPTIONS.keys()))
selected_cuisine, GRID_SIZE = CUISINE_OPTIONS[selected_label]

@st.cache_data
def load_scores(cuisine):
    filename = f"final_scores_{cuisine.replace('/', '_').replace(' ', '_')}.csv"
    path = os.path.join(DATA_DIR, filename)
    return pd.read_csv(path)

@st.cache_data
def load_restaurants(cuisine):
    path = os.path.join(DATA_DIR, "allcuisine.csv")
    df = pd.read_csv(path)
    df = df.sort_values('INSPECTION DATE', ascending=False).drop_duplicates(subset=['CAMIS'], keep='first')
    df = df[(df['Latitude'] != 0) & (df['Longitude'] != 0)]
    df = df.dropna(subset=['CUISINE DESCRIPTION'])
    df_target = df[df['CUISINE DESCRIPTION'] == cuisine]
    return df_target[['Latitude', 'Longitude', 'DBA', 'SCORE', 'GRADE']]

try:
    grid_info = load_scores(selected_cuisine)
    restaurant_points = load_restaurants(selected_cuisine)
except FileNotFoundError as e:
    st.error(f"❌ Score file not found: {e}")
    st.stop()

# ---------- 2. 行政区快捷选择 ----------
st.sidebar.subheader("2️⃣ Borough Quick Select")
borough = st.sidebar.selectbox("Select Borough", ["Custom"] + list(BOROUGH_COORDS.keys()))

if borough != "Custom":
    default_lat, default_lon = BOROUGH_COORDS[borough]
else:
    default_lat, default_lon = 40.7580, -73.9855

# ---------- 3. 地址输入 ----------
st.sidebar.subheader("3️⃣ Address Input (optional)")
address = st.sidebar.text_input("Enter address (e.g., 350 5th Ave, NYC)", value="")

# ---------- 4. 坐标输入 ----------
st.sidebar.subheader("4️⃣ Coordinates")
st.sidebar.caption("NYC coordinate range:")
st.sidebar.caption("Latitude: 40.4 – 41.0")
st.sidebar.caption("Longitude: -74.3 – -73.7")

lat = st.sidebar.number_input("Latitude", value=default_lat, format="%.4f", step=0.001,
                               min_value=40.0, max_value=41.5)
lon = st.sidebar.number_input("Longitude", value=default_lon, format="%.4f", step=0.001,
                               min_value=-74.5, max_value=-73.5)

if address:
    try:
        geolocator = Nominatim(user_agent="nyc_restaurant_recommender")
        location = geolocator.geocode(address + ", New York City, NY", timeout=10)
        if location:
            lat, lon = location.latitude, location.longitude
            st.sidebar.success(f"✅ Found: {location.address[:40]}...")
            st.sidebar.info(f"Lat: {lat:.4f}, Lon: {lon:.4f}")
        else:
            st.sidebar.warning("Address not found. Using coordinates above.")
    except Exception as e:
        st.sidebar.warning(f"Geocoding error: {str(e)[:50]}")

# 打分逻辑
grid_x = int(lon / GRID_SIZE)
grid_y = int(lat / GRID_SIZE)
grid_id = f"{grid_x}_{grid_y}"

result = grid_info[grid_info['grid_id'] == grid_id]

st.subheader(f"📍 Results for {selected_label}")

if len(result) == 0:
    st.warning("⚠️ No restaurant data found for this grid. Try a different location within NYC.")
    score = None
else:
    row = result.iloc[0]
    score = row['suitability_score']
    n_restaurants = int(row['n_restaurants'])
    avg_score = row['avg_score']
    cuisine_diversity = int(row['cuisine_diversity'])
    target_count = int(row['n_target'])
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🎯 Suitability Score", f"{score:.1f}/100")
    col2.metric("🍽️ Restaurants in Grid", f"{n_restaurants}")
    col3.metric(f"{selected_label} in Grid", f"{target_count}")
    col4.metric("📊 Cuisine Diversity", f"{cuisine_diversity}")
    
    if score >= 70:
        st.success(f"✅ **Highly suitable** for opening a {selected_cuisine} restaurant.")
    elif score >= 50:
        st.info(f"🟡 **Moderately suitable** for a {selected_cuisine} restaurant.")
    else:
        st.warning(f"⚠️ **Low suitability** for a {selected_cuisine} restaurant.")
    
    with st.expander("📊 Detailed Analysis"):
        col_a, col_b = st.columns(2)
        with col_a:
            st.write("**Grid Information:**")
            st.write(f"- Grid ID: `{grid_id}`")
            st.write(f"- Grid size: {GRID_SIZE}° (~{GRID_SIZE*111:.0f} km)")
            st.write(f"- Center: ({row['center_lat']:.4f}, {row['center_lon']:.4f})")
        with col_b:
            st.write("**Nearby Restaurants:**")
            st.write(f"- Total: {n_restaurants}")
            st.write(f"- {selected_cuisine}: {target_count}")
            st.write(f"- Avg inspection score: {avg_score:.1f}")

# 地图
st.subheader(f"🗺️ {selected_label} Restaurants in NYC")
st.caption(f"Each marker represents an existing {selected_cuisine} restaurant. Gold markers with numbers show the Top 10 recommended grids.")

m = folium.Map(location=[40.7128, -74.0060], zoom_start=11, tiles='OpenStreetMap')

# ---------- 1. 显示所有网格评分 ----------
for _, r in grid_info.iterrows():
    s = r['suitability_score']
    if s >= 70:
        color = '#2ecc71'
    elif s >= 50:
        color = '#f39c12'
    else:
        color = '#e74c3c'
    
    folium.CircleMarker(
        location=[r['center_lat'], r['center_lon']],
        radius=3,
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.3,
        weight=0,
    ).add_to(m)

# ---------- 2. 显示已有餐厅 ----------
for _, r in restaurant_points.iterrows():
    grade = r.get('GRADE', 'N/A')
    if grade == 'A':
        color = 'green'
    elif grade == 'B':
        color = 'orange'
    elif grade == 'C':
        color = 'red'
    else:
        color = 'gray'
    
    folium.CircleMarker(
        location=[r['Latitude'], r['Longitude']],
        radius=3,
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.6,
        popup=folium.Popup(
            f"<b>{r['DBA']}</b><br>"
            f"Grade: {grade}<br>"
            f"Inspection Score: {r.get('SCORE', 'N/A')} <i>(lower is better)</i>",
            max_width=250
        )
    ).add_to(m)

# ---------- 3. Top 10 推荐网格 ----------
top10 = grid_info.nlargest(10, 'suitability_score').reset_index(drop=True)

for i, row in top10.iterrows():
    rank = i + 1
    folium.Marker(
        location=[row['center_lat'], row['center_lon']],
        popup=folium.Popup(
            f"<b>🏆 Rank #{rank}</b><br>"
            f"Score: {row['suitability_score']:.1f}/100<br>"
            f"Restaurants: {int(row['n_restaurants'])}<br>"
            f"{selected_cuisine}: {int(row['n_target'])}<br>"
            f"Cuisine Diversity: {int(row['cuisine_diversity'])}",
            max_width=220
        ),
        tooltip=f"Rank #{rank} | Score: {row['suitability_score']:.1f}",
        icon=folium.DivIcon(
            html=f'''
                <div style="
                    background-color: gold;
                    border: 3px solid darkorange;
                    border-radius: 50%;
                    width: 32px;
                    height: 32px;
                    text-align: center;
                    line-height: 26px;
                    font-weight: bold;
                    font-size: 14px;
                    color: #333;
                    box-shadow: 0 2px 6px rgba(0,0,0,0.4);
                ">{rank}</div>
            ''',
            icon_size=(32, 32),
            icon_anchor=(16, 16)
        )
    ).add_to(m)

# ---------- 4. 用户位置 ----------
if score is not None:
    folium.Marker(
        [lat, lon],
        popup=f"Your Location<br>Score: {score:.1f}",
        tooltip="Your Location",
        icon=folium.Icon(color='blue', icon='star', prefix='fa')
    ).add_to(m)

# ---------- 5. 渲染地图 ----------
st_folium(m, width=1200, height=550, returned_objects=[])

# ==================== Top 10 表格 ====================
st.subheader(f"🏆 Top 10 Recommended Grids for {selected_label}")

top10_display = top10[['grid_id', 'center_lat', 'center_lon',
                        'suitability_score', 'n_restaurants', 'n_target']].copy()
top10_display.columns = ['Grid ID', 'Latitude', 'Longitude', 'Score', 'Restaurants', f'{selected_cuisine} Count']
top10_display.index = top10_display.index + 1

st.dataframe(
    top10_display.style.format({
        'Latitude': '{:.4f}',
        'Longitude': '{:.4f}',
        'Score': '{:.2f}'
    }).background_gradient(subset=['Score'], cmap='RdYlGn'),
    width='stretch'
)

# 图例
st.markdown("---")
col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("### 📖 Legend")
    st.markdown(f"""
    **Map layers:**
    - 🟢🟡🔴 Semi-transparent dots: grid suitability score
    - 🟢🟠🔴 Small dots: existing {selected_cuisine} restaurants (by grade)
    - 🏆 Gold numbered markers: Top 10 recommended grids
    - 🔵 Blue star: your selected location
    
    **Score interpretation:**
    - 🟢 70-100: Highly suitable
    - 🟡 50-69: Moderately suitable
    - 🔴 0-49: Low suitability
    """)

with col2:
    st.markdown("### 💡 How It Works")
    st.markdown(f"""
    This recommender uses a **Graph Convolutional Network (GCN)** trained on NYC restaurant inspection data.
    It analyzes spatial patterns of existing **{selected_cuisine}** restaurants and their neighbors to predict
    which locations are most suitable for opening a new restaurant.
    
    **Model details:**
    - Grid size: {GRID_SIZE}° (~{GRID_SIZE*111:.0f} km)
    - Features: restaurant density, average inspection score, cuisine diversity, target cuisine count
    - 5-seed multi-run training for stable results
    """)
