import cv2
import numpy as np

def resize_h(img, target=300):
    if img is None or img.size == 0:
        return img
    h = img.shape[0]
    s = max(1, target / max(1, h))
    return cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)

def add_border(img, p=0.12):
    h, w = img.shape[:2]
    px, py = int(w * p), int(h * p)
    return cv2.copyMakeBorder(img, py, py, px, px, cv2.BORDER_REPLICATE)

def gamma_correct(img, gamma):
    table = np.array([(i / 255.0) ** gamma * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(img, table)

def clahe_gray(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.createCLAHE(2.0, (8, 8)).apply(g)

def gray3(img):
    return cv2.cvtColor(clahe_gray(img), cv2.COLOR_GRAY2BGR)

def low_light(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    m = g.mean()
    if m < 90:
        x = gamma_correct(img, 0.65)
    elif m > 180:
        x = gamma_correct(img, 1.35)
    else:
        x = img.copy()
    y = clahe_gray(x)
    return cv2.cvtColor(y, cv2.COLOR_GRAY2BGR)

def sharpen(img):
    g = clahe_gray(img)
    b = cv2.GaussianBlur(g, (0, 0), 1)
    s = cv2.addWeighted(g, 1.9, b, -0.9, 0)
    return cv2.cvtColor(s, cv2.COLOR_GRAY2BGR)

def deblur(img):
    x = cv2.bilateralFilter(img, 5, 45, 45)
    g = clahe_gray(x)
    b = cv2.GaussianBlur(g, (0, 0), 1.2)
    s = cv2.addWeighted(g, 2.1, b, -1.1, 0)
    return cv2.cvtColor(s, cv2.COLOR_GRAY2BGR)

def otsu(img):
    g = clahe_gray(img)
    t = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    return cv2.cvtColor(t, cv2.COLOR_GRAY2BGR)

def adaptive(img):
    g = clahe_gray(img)
    t = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 7)
    return cv2.cvtColor(t, cv2.COLOR_GRAY2BGR)

def sr2(img):
    x = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    return sharpen(x)

def rotate_img(img, angle):
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1)
    return cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

def deskew(img):
    if img is None or img.size == 0:
        return img
    x = resize_h(img, 300)
    g = clahe_gray(x)
    t = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    t = cv2.morphologyEx(t, cv2.MORPH_CLOSE, np.ones((5, 25), np.uint8))
    cnts, _ = cv2.findContours(t, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return x
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < 0.03 * x.shape[0] * x.shape[1]:
        return x
    rect = cv2.minAreaRect(c)
    angle = rect[-1]
    angle = angle + 90 if angle < -45 else angle
    if abs(angle) > 18:
        return x
    return rotate_img(x, angle)

def make_variants(raw_crop, pad_crop):
    imgs = []
    for name, img in [("raw", raw_crop), ("pad", pad_crop)]:
        if img is None or img.size == 0:
            continue
        x = add_border(resize_h(img, 300), 0.10)
        l = low_light(x)
        d = deskew(l)
        for n, v in [
            (name, x),
            (name + "_gray", gray3(x)),
            (name + "_light", l),
            (name + "_sharp", sharpen(x)),
            (name + "_deblur", deblur(x)),
            (name + "_otsu", otsu(x)),
            (name + "_adaptive", adaptive(x)),
            (name + "_sr2", sr2(x)),
            (name + "_sr2_light", sr2(l)),
            (name + "_deskew", d),
            (name + "_deskew_sharp", sharpen(d)),
            (name + "_deskew_sr2", sr2(d)),
        ]:
            imgs.append((n, v))
    return imgs

def split_lines(img):
    if img is None or img.size == 0:
        return []
    x = resize_h(img, 380)
    h = x.shape[0]
    gap = int(h * 0.05)
    return [x[:h//2+gap], x[h//2-gap:]]
def make_variants_video(raw_crop, pad_crop):
    imgs = []
    base = pad_crop if pad_crop is not None and pad_crop.size else raw_crop

    if base is None or base.size == 0:
        return []

    x = add_border(resize_h(base, 260), 0.08)
    l = low_light(x)
    s = sharpen(l)
    d = deblur(l)

    imgs += [
        ("video_light", l),
        ("video_sharp", s),
        ("video_deblur", d),
    ]

    if base.shape[0] < 80:
        imgs.append(("video_sr2", sr2(l)))

    return imgs
#=========
app_code = r'''
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(
    page_title="Heart Failure Prediction Dashboard",
    page_icon="❤️",
    layout="wide"
)

st.markdown("""
<style>
.main {background-color: #f8fafc;}
.block-container {padding-top: 1.5rem;}
.big-title {
    font-size: 42px;
    font-weight: 800;
    color: #b91c1c;
    margin-bottom: 0px;
}
.sub-title {
    font-size: 18px;
    color: #475569;
    margin-bottom: 20px;
}
.card {
    padding: 22px;
    border-radius: 18px;
    background: white;
    box-shadow: 0px 4px 18px rgba(0,0,0,0.08);
    border: 1px solid #e2e8f0;
}
.metric-label {
    font-size: 15px;
    color: #64748b;
}
.metric-value {
    font-size: 30px;
    font-weight: 800;
    color: #0f172a;
}
.highlight {
    padding: 15px;
    border-radius: 14px;
    background-color: #fee2e2;
    border-left: 6px solid #dc2626;
    color: #7f1d1d;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

DATA_PATH = Path("data/heart_clean.csv")
REPORT_DIR = Path("reports")
MODEL_DIR = Path("models")

num_features = ["Age", "RestingBP", "Cholesterol", "FastingBS", "MaxHR", "Oldpeak"]
cat_features = ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]
target_col = "HeartDisease"

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

@st.cache_data
def load_reports():
    reports = {}
    for name in [
        "model_results",
        "final_model_compare_v2",
        "chi_square_results",
        "anova_results",
        "best_params_v2"
    ]:
        path = REPORT_DIR / f"{name}.csv"
        if path.exists():
            reports[name] = pd.read_csv(path)
    return reports

@st.cache_resource
def load_models():
    files = {
        "Logistic Regression V1": "logistic_regression_v1.pkl",
        "Decision Tree V1": "decision_tree_v1.pkl",
        "Logistic Regression Tuned V2": "logistic_regression_tuned_v2.pkl",
        "Random Forest Tuned V2": "random_forest_tuned_v2.pkl",
        "SVM Tuned V2": "svm_tuned_v2.pkl",
        "KNN V2": "knn_v2.pkl",
        "Decision Tree V2": "decision_tree_v2.pkl",
        "Gradient Boosting Tuned V2": "gradient_boosting_tuned_v2.pkl",
        "Linear Regression V1": "linear_regression_v1.pkl"
    }
    models = {}
    for name, file in files.items():
        path = MODEL_DIR / file
        if path.exists():
            models[name] = joblib.load(path)
    scalers = {}
    for name in ["scaler_v1.pkl", "scaler_v2.pkl"]:
        path = MODEL_DIR / name
        if path.exists():
            scalers[name] = joblib.load(path)
    columns = {}
    for name in ["model_columns_v1.pkl", "model_columns_v2.pkl"]:
        path = MODEL_DIR / name
        if path.exists():
            columns[name] = joblib.load(path)
    return models, scalers, columns

def make_model_data(df):
    df_model = pd.get_dummies(df, columns=cat_features, drop_first=True)
    bool_cols = df_model.select_dtypes(include=["bool"]).columns
    df_model[bool_cols] = df_model[bool_cols].astype(int)
    X = df_model.drop(target_col, axis=1)
    y = df_model[target_col]
    return X, y

def scale_data(X, scaler):
    X_scaled = X.copy()
    X_scaled[num_features] = scaler.transform(X[num_features])
    return X_scaled

def need_scaled(model_name):
    return model_name in [
        "Logistic Regression V1",
        "Logistic Regression Tuned V2",
        "SVM Tuned V2",
        "KNN V2"
    ]

def get_scaler_name(model_name):
    if "V1" in model_name:
        return "scaler_v1.pkl"
    return "scaler_v2.pkl"

def label_target(x):
    return "Heart Disease" if x == 1 else "Normal"

df = load_data()
reports = load_reports()
models, scalers, model_columns = load_models()

st.markdown('<div class="big-title">❤️ Heart Failure Prediction Dashboard</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Dashboard phân tích dữ liệu, kiểm định thống kê và dự đoán nguy cơ bệnh tim bằng Machine Learning.</div>',
    unsafe_allow_html=True
)

st.sidebar.title("🎛️ Bộ lọc dữ liệu")

sex_filter = st.sidebar.multiselect("Giới tính", sorted(df["Sex"].unique()), default=sorted(df["Sex"].unique()))
cp_filter = st.sidebar.multiselect("Loại đau ngực", sorted(df["ChestPainType"].unique()), default=sorted(df["ChestPainType"].unique()))
slope_filter = st.sidebar.multiselect("ST Slope", sorted(df["ST_Slope"].unique()), default=sorted(df["ST_Slope"].unique()))
target_filter = st.sidebar.multiselect("HeartDisease", sorted(df["HeartDisease"].unique()), default=sorted(df["HeartDisease"].unique()))

df_view = df[
    df["Sex"].isin(sex_filter)
    & df["ChestPainType"].isin(cp_filter)
    & df["ST_Slope"].isin(slope_filter)
    & df["HeartDisease"].isin(target_filter)
].copy()

total_samples = len(df_view)
disease_count = int((df_view["HeartDisease"] == 1).sum())
normal_count = int((df_view["HeartDisease"] == 0).sum())
disease_rate = disease_count / total_samples * 100 if total_samples > 0 else 0

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(f'<div class="card"><div class="metric-label">Tổng số mẫu</div><div class="metric-value">{total_samples}</div></div>', unsafe_allow_html=True)

with c2:
    st.markdown(f'<div class="card"><div class="metric-label">Normal</div><div class="metric-value">{normal_count}</div></div>', unsafe_allow_html=True)

with c3:
    st.markdown(f'<div class="card"><div class="metric-label">Heart Disease</div><div class="metric-value">{disease_count}</div></div>', unsafe_allow_html=True)

with c4:
    st.markdown(f'<div class="card"><div class="metric-label">Tỷ lệ bệnh tim</div><div class="metric-value">{disease_rate:.2f}%</div></div>', unsafe_allow_html=True)

st.write("")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📌 Tổng quan",
    "📊 EDA",
    "🧪 Thống kê",
    "🤖 Mô hình",
    "📈 Linear Regression",
    "🔮 Dự đoán"
])

with tab1:
    st.subheader("📌 Tổng quan dữ liệu")
    col1, col2 = st.columns([2, 1])

    with col1:
        st.dataframe(df_view.head(20), use_container_width=True)

    with col2:
        st.markdown("### Ý nghĩa biến mục tiêu")
        st.info("HeartDisease = 0: Normal\n\nHeartDisease = 1: Heart Disease")
        st.markdown("### Kích thước dữ liệu")
        st.write("Dữ liệu gốc:", df.shape)
        st.write("Dữ liệu sau lọc:", df_view.shape)

    st.subheader("📋 Thống kê mô tả")
    st.dataframe(df_view.describe().T, use_container_width=True)

    st.subheader("🧹 Chất lượng dữ liệu")
    q1, q2 = st.columns(2)
    with q1:
        st.metric("Missing values", int(df_view.isnull().sum().sum()))
    with q2:
        st.metric("Duplicate rows", int(df_view.duplicated().sum()))

with tab2:
    st.subheader("📊 Phân tích trực quan dữ liệu")

    if len(df_view) == 0:
        st.warning("Không có dữ liệu sau khi lọc.")
    else:
        col1, col2 = st.columns(2)

        with col1:
            target_counts = df_view["HeartDisease"].value_counts().reset_index()
            target_counts.columns = ["HeartDisease", "Count"]
            target_counts["Label"] = target_counts["HeartDisease"].map({0: "Normal", 1: "Heart Disease"})
            fig = px.pie(
                target_counts,
                values="Count",
                names="Label",
                hole=0.45,
                title="Phân phối HeartDisease"
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.histogram(
                df_view,
                x="Age",
                color="HeartDisease",
                barmode="overlay",
                title="Phân phối Age theo HeartDisease"
            )
            st.plotly_chart(fig, use_container_width=True)

        col3, col4 = st.columns(2)

        with col3:
            fig = px.histogram(
                df_view,
                x="Sex",
                color="HeartDisease",
                barmode="group",
                title="HeartDisease theo giới tính"
            )
            st.plotly_chart(fig, use_container_width=True)

        with col4:
            fig = px.histogram(
                df_view,
                x="ChestPainType",
                color="HeartDisease",
                barmode="group",
                title="HeartDisease theo loại đau ngực"
            )
            st.plotly_chart(fig, use_container_width=True)

        col5, col6 = st.columns(2)

        with col5:
            fig = px.box(
                df_view,
                x="HeartDisease",
                y="MaxHR",
                color="HeartDisease",
                title="Boxplot MaxHR theo HeartDisease"
            )
            st.plotly_chart(fig, use_container_width=True)

        with col6:
            fig = px.box(
                df_view,
                x="HeartDisease",
                y="Oldpeak",
                color="HeartDisease",
                title="Boxplot Oldpeak theo HeartDisease"
            )
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("🔥 Ma trận tương quan biến số")
        corr = df_view[num_features + [target_col]].corr()
        fig = px.imshow(
            corr,
            text_auto=".2f",
            aspect="auto",
            title="Correlation Heatmap"
        )
        st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.subheader("🧪 Kiểm định thống kê")

    st.markdown('<div class="highlight">Chi-square dùng để kiểm định mối liên hệ giữa biến phân loại và HeartDisease.</div>', unsafe_allow_html=True)
    st.write("")

    if "chi_square_results" in reports:
        chi_df = reports["chi_square_results"]
        st.dataframe(chi_df, use_container_width=True)
        fig = px.bar(
            chi_df,
            x="Feature",
            y="Chi-square",
            color="Significant",
            title="Chi-square theo biến phân loại"
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="highlight">ANOVA dùng để kiểm tra sự khác biệt trung bình của biến số giữa nhóm Normal và Heart Disease.</div>', unsafe_allow_html=True)
    st.write("")

    if "anova_results" in reports:
        anova_df = reports["anova_results"]
        st.dataframe(anova_df, use_container_width=True)
        fig = px.bar(
            anova_df,
            x="Feature",
            y="F-statistic",
            color="Significant",
            title="ANOVA F-statistic theo biến số"
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("📦 Boxplot các biến ANOVA")
    box_feature = st.selectbox("Chọn biến số", ["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak"])
    fig = px.box(
        df,
        x="HeartDisease",
        y=box_feature,
        color="HeartDisease",
        title=f"{box_feature} theo HeartDisease"
    )
    st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.subheader("🤖 So sánh mô hình Machine Learning")

    if "final_model_compare_v2" in reports:
        compare_df = reports["final_model_compare_v2"].sort_values("F1-score", ascending=False)
    elif "model_results" in reports:
        compare_df = reports["model_results"].sort_values("F1-score", ascending=False)
    else:
        compare_df = pd.DataFrame()

    if len(compare_df) > 0:
        st.dataframe(compare_df, use_container_width=True)

        best_row = compare_df.iloc[0]
        st.success(
            f"Mô hình tốt nhất: {best_row['Model']} | "
            f"Accuracy = {best_row['Accuracy']:.4f} | "
            f"F1-score = {best_row['F1-score']:.4f}"
        )

        plot_df = compare_df.melt(
            id_vars="Model",
            value_vars=["Accuracy", "Precision", "Recall", "F1-score"],
            var_name="Metric",
            value_name="Score"
        )

        fig = px.bar(
            plot_df,
            x="Metric",
            y="Score",
            color="Model",
            barmode="group",
            title="So sánh Accuracy, Precision, Recall, F1-score"
        )
        fig.update_yaxes(range=[0, 1])
        st.plotly_chart(fig, use_container_width=True)

    if "best_params_v2" in reports:
        st.subheader("⚙️ Best Params V2")
        st.dataframe(reports["best_params_v2"], use_container_width=True)

    st.subheader("🧩 Confusion Matrix")
    model_names = [m for m in models.keys() if m != "Linear Regression V1"]
    selected_model = st.selectbox("Chọn mô hình để xem Confusion Matrix", model_names, index=model_names.index("Gradient Boosting Tuned V2") if "Gradient Boosting Tuned V2" in model_names else 0)

    X, y = make_model_data(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = models[selected_model]
    X_eval = X_test.copy()

    if need_scaled(selected_model):
        scaler_key = get_scaler_name(selected_model)
        X_eval = scale_data(X_eval, scalers[scaler_key])

    y_pred = model.predict(X_eval)
    cm = confusion_matrix(y_test, y_pred)

    fig = px.imshow(
        cm,
        text_auto=True,
        labels=dict(x="Predicted", y="Actual"),
        x=["Normal", "Heart Disease"],
        y=["Normal", "Heart Disease"],
        title=f"Confusion Matrix - {selected_model}"
    )
    st.plotly_chart(fig, use_container_width=True)

with tab5:
    st.subheader("📈 Hồi quy tuyến tính")

    st.info("Linear Regression được dùng để dự đoán biến liên tục MaxHR từ Age, RestingBP, Cholesterol và Oldpeak.")

    if "Linear Regression V1" in models:
        linear_model = models["Linear Regression V1"]

        X_lr = df[["Age", "RestingBP", "Cholesterol", "Oldpeak"]]
        y_lr = df["MaxHR"]

        X_lr_train, X_lr_test, y_lr_train, y_lr_test = train_test_split(
            X_lr,
            y_lr,
            test_size=0.2,
            random_state=42
        )

        y_lr_pred = linear_model.predict(X_lr_test)

        l1, l2, l3 = st.columns(3)
        with l1:
            st.metric("MAE", f"{mean_absolute_error(y_lr_test, y_lr_pred):.4f}")
        with l2:
            st.metric("MSE", f"{mean_squared_error(y_lr_test, y_lr_pred):.4f}")
        with l3:
            st.metric("R2-score", f"{r2_score(y_lr_test, y_lr_pred):.4f}")

        fig = px.scatter(
            x=y_lr_test,
            y=y_lr_pred,
            labels={"x": "MaxHR thực tế", "y": "MaxHR dự đoán"},
            title="Linear Regression: Actual vs Predicted"
        )

        min_val = min(y_lr_test.min(), y_lr_pred.min())
        max_val = max(y_lr_test.max(), y_lr_pred.max())

        fig.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode="lines",
                name="Đường lý tưởng y = x",
                line=dict(color="red", dash="dash")
            )
        )

        st.plotly_chart(fig, use_container_width=True)

with tab6:
    st.subheader("🔮 Dự đoán nguy cơ bệnh tim")

    st.warning("Kết quả chỉ phục vụ mục đích học tập và tham khảo, không thay thế chẩn đoán y khoa.")

    pred_model_names = [m for m in models.keys() if m != "Linear Regression V1"]
    default_idx = pred_model_names.index("Gradient Boosting Tuned V2") if "Gradient Boosting Tuned V2" in pred_model_names else 0

    pred_model_name = st.selectbox("Chọn mô hình dự đoán", pred_model_names, index=default_idx)

    with st.form("prediction_form"):
        c1, c2, c3 = st.columns(3)

        with c1:
            age = st.number_input("Age", min_value=18, max_value=100, value=50)
            sex = st.selectbox("Sex", ["M", "F"])
            chest = st.selectbox("ChestPainType", ["ASY", "ATA", "NAP", "TA"])
            resting_bp = st.number_input("RestingBP", min_value=60, max_value=220, value=130)

        with c2:
            cholesterol = st.number_input("Cholesterol", min_value=80, max_value=650, value=237)
            fasting_bs = st.selectbox("FastingBS", [0, 1])
            resting_ecg = st.selectbox("RestingECG", ["Normal", "ST", "LVH"])
            max_hr = st.number_input("MaxHR", min_value=60, max_value=220, value=140)

        with c3:
            exercise_angina = st.selectbox("ExerciseAngina", ["N", "Y"])
            oldpeak = st.number_input("Oldpeak", min_value=-3.0, max_value=7.0, value=1.0, step=0.1)
            st_slope = st.selectbox("ST_Slope", ["Up", "Flat", "Down"])

        submitted = st.form_submit_button("Dự đoán")

    if submitted:
        input_df = pd.DataFrame([{
            "Age": age,
            "Sex": sex,
            "ChestPainType": chest,
            "RestingBP": resting_bp,
            "Cholesterol": cholesterol,
            "FastingBS": fasting_bs,
            "RestingECG": resting_ecg,
            "MaxHR": max_hr,
            "ExerciseAngina": exercise_angina,
            "Oldpeak": oldpeak,
            "ST_Slope": st_slope
        }])

        input_encoded = pd.get_dummies(input_df, columns=cat_features, drop_first=True)

        col_key = "model_columns_v1.pkl" if "V1" in pred_model_name else "model_columns_v2.pkl"
        target_columns = model_columns[col_key]

        input_encoded = input_encoded.reindex(columns=target_columns, fill_value=0)

        if need_scaled(pred_model_name):
            scaler_key = get_scaler_name(pred_model_name)
            input_encoded = scale_data(input_encoded, scalers[scaler_key])

        pred_model = models[pred_model_name]
        pred = int(pred_model.predict(input_encoded)[0])

        if hasattr(pred_model, "predict_proba"):
            proba = float(pred_model.predict_proba(input_encoded)[0][1])
        else:
            proba = np.nan

        if pred == 1:
            st.error("Kết quả dự đoán: Có nguy cơ bệnh tim")
        else:
            st.success("Kết quả dự đoán: Bình thường")

        if not np.isnan(proba):
            st.metric("Xác suất Heart Disease", f"{proba * 100:.2f}%")
            st.progress(min(max(proba, 0), 1))

        st.dataframe(input_df, use_container_width=True)
'''

with open("app.py", "w", encoding="utf-8") as f:
    f.write(app_code)

print("Đã ghi app.py Streamlit bản siêu xịn")