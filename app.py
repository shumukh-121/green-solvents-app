import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title="المساعد الذكي للمذيبات الخضراء", page_icon="🧪")

st.title("🧪 المستشار الذكي لاختيار المذيبات المستدامة")
st.write(
    "نظام ذكي للتنبؤ بكفاءة المذيبات الخضراء وتحليل التوافق الكيميائي واقتراح البدائل الأفضل."
)
st.divider()

# 1. قاعدة بيانات مبسطة تجريبية (سيتم توسيعها بالبيانات الحقيقية لاحقاً)
solvents_db = [
    {
        "name": "Choline Chloride : Urea (DES 1)",
        "viscosity": 120.0,
        "polarity": 0.85,
        "thermal": 180.0,
        "safety": "آمن بيئياً، ولكن ينبغي تجنب استخدامه مع الأحماض القوية جداً عند درجات حرارة أعلى من 150°C.",
    },
    {
        "name": "Choline Chloride : Glycerol (DES 2)",
        "viscosity": 95.0,
        "polarity": 0.90,
        "thermal": 200.0,
        "safety": "استقرار حراري ممتاز وتوافق عالٍ مع التفاعلات التحفيزية العضوية.",
    },
    {
        "name": "Ethyl Lactate",
        "viscosity": 2.5,
        "polarity": 0.55,
        "thermal": 150.0,
        "safety": "مذيب استري خفيف، ينبغي الحذر من التفكك في الوسط القاعدي القوي.",
    },
    {
        "name": "[BMIM][PF6] Ionic Liquid",
        "viscosity": 310.0,
        "polarity": 0.40,
        "thermal": 120.0,
        "safety": "زوجة عالية قد تعيق انتقال المادة، ويتفكك في ظروف الرطوبة العالية.",
    },
]

df_db = pd.DataFrame(solvents_db)

# تدريب نموذج تعلم الآلة
X = df_db[["viscosity", "polarity", "thermal"]]
y = [88.0, 94.0, 65.0, 48.0]  # كفاءة التفاعل %
model = RandomForestRegressor(n_estimators=10, random_state=42).fit(X, y)

# 2. إدخال بيانات المذيب المراد اختباره
st.subheader("📥 أدخلي الخصائص الفيزيوكيميائية للمذيب:")
col1, col2, col3 = st.columns(3)

with col1:
    visc_input = st.number_input(
        "اللزوجة (Viscosity - mPa·s)", value=100.0, step=5.0
    )
with col2:
    pol_input = st.number_input(
        "القطبية (Polarity Index)", value=0.85, step=0.05
    )
with col3:
    therm_input = st.number_input(
        "الاستقرار الحراري (Thermal Stability °C)", value=180.0, step=10.0
    )

# 3. زر التحليل
if st.button("🚀 تحليل الكفاءة والتوافق الكيميائي", type="primary"):
    # التنبؤ بالكفاءة
    pred_yield = model.predict([[visc_input, pol_input, therm_input]])[0]

    # التعرف على أقرب مذيب في قاعدة البيانات بناءً على الخصائص
    distances = np.sqrt(
        (df_db["viscosity"] - visc_input) ** 2
        + (df_db["polarity"] - pol_input) ** 2 * 100
        + (df_db["thermal"] - therm_input) ** 2
    )
    closest_idx = distances.idxmin()
    matched_solvent = df_db.iloc[closest_idx]

    st.markdown("---")
    st.subheader("📊 نتائج التحليل:")

    st.info(f"🔍 **المذيب الأقرب لهذه الخصائص:** `{matched_solvent['name']}`")
    st.metric(
        label="نسبة كفاءة التفاعل المتوقعة (Yield %)",
        value=f"{pred_yield:.1f}%",
    )

    # تحذير التوافق والسلامة الكيميائية
    st.warning(f"⚠️ **ملاحظات السلامة والتوافق:** {matched_solvent['safety']}")

    # اقتراح بديل إذا كانت الكفاءة أقل من 80%
    if pred_yield < 80.0:
        st.error(
            "⚠️ **الكفاءة المتوقعة أقل من 80%! يوصى باستخدام بديل أفضل.**"
        )
        best_idx = np.argmax(y)
        best_solvent = df_db.iloc[best_idx]
        st.success(
            f"💡 **المذيب البديل الموصى به:** `{best_solvent['name']}`\n\n"
            f"- **الكفاءة المتوقعة:** `{y[best_idx]}%`\n"
            f"- **اللزوجة:** {best_solvent['viscosity']} | **القطبية:** {best_solvent['polarity']} | **الاستقرار:** {best_solvent['thermal']}°C"
        )
    else:
        st.success(
            "✅ **هذا المذيب يحقق كفاءة عالية وتفاعل ممتاز في المنظومة التحفيزية.**"
        )
