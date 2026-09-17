import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor

# 1. عنوان الواجهة
st.set_page_config(page_title="المساعد الذكي للمذيبات الخضراء", page_icon="🧪")
st.title("🧪 المساعد الذكي لتوقع كفاءة المذيبات المستدامة")
st.write(
    "هذا التطبيق يستخدم الذكاء الاصطناعي للتنبؤ بكفاءة التفاعل التحفيزي بناءً على خصائص المذيب الأخضر."
)

st.divider()

# 2. إدخال بيانات المذيب من المستخدم
st.subheader("📥 أدخل خصائص المذيب المستدام:")
col1, col2, col3 = st.columns(3)

with col1:
    viscosity = st.number_input("اللزوجة (Viscosity - mPa·s)", value=100.0)
with col2:
    polarity = st.number_input("القطبية (Polarity Index)", value=0.8)
with col3:
    thermal = st.number_input(
        "الاستقرار الحراري (Thermal Stability - °C)", value=150.0
    )

# 3. نموذج ذكاء اصطناعي تجريبي
X = pd.DataFrame(
    {
        "Viscosity": [120, 250, 90, 400, 50],
        "Polarity": [0.85, 0.40, 0.90, 0.20, 0.95],
        "Thermal": [180, 120, 200, 100, 220],
    }
)
y = [88, 52, 94, 30, 98]
model = RandomForestRegressor().fit(X, y)

# 4. زر التنبؤ والتجهيز
if st.button("🚀 تحليل وحساب الكفاءة المتوقعة", type="primary"):
    prediction = model.predict([[viscosity, polarity, thermal]])[0]
    st.success(
        f"💡 **نسبة كفاءة التفاعل التحفيزي المتوقعة:** `{prediction:.2f}%`"
    )
