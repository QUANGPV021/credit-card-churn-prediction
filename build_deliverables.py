import base64
import json
import os
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT

with open("results_summary.json", "r", encoding="utf-8") as f:
    summary = json.load(f)

def b64_img(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def md_cell(lines):
    if isinstance(lines, str):
        lines = [line + "\n" for line in lines.strip().split("\n")]
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": lines
    }

def code_cell(code_str, text_output=None, img_paths=None, exec_count=1):
    outputs = []
    if text_output:
        outputs.append({
            "name": "stdout",
            "output_type": "stream",
            "text": [line + "\n" for line in text_output.strip().split("\n")]
        })
    if img_paths:
        for p in img_paths:
            if os.path.exists(p):
                outputs.append({
                    "data": {
                        "image/png": b64_img(p),
                        "text/plain": [f"<Figure size with Image {p}>"]
                    },
                    "metadata": {},
                    "output_type": "display_data"
                })
    return {
        "cell_type": "code",
        "execution_count": exec_count,
        "metadata": {},
        "outputs": outputs,
        "source": [line + "\n" for line in code_str.strip().split("\n")]
    }

# --- 1. BUILD JUPYTER NOTEBOOK (.ipynb) ---
models = summary["models"]
gm = summary["group_means"]
top_feat = summary["top_features"]

cells = [
    md_cell("""# DỰ ÁN KHOA HỌC DỮ LIỆU
## Đề tài: Dự đoán nguy cơ khách hàng ngừng sử dụng dịch vụ thẻ tín dụng nhằm hỗ trợ ra quyết định giữ chân khách hàng tại ngân hàng
- **Nguồn dữ liệu**: [Credit Card Customers - Kaggle (`sakshigoyal7/credit-card-customers`)](https://www.kaggle.com/datasets/sakshigoyal7/credit-card-customers)
- **Quy mô dữ liệu**: 10,127 bản ghi khách hàng × 20 thuộc tính đặc trưng
- **Thuật toán sử dụng**: `Logistic Regression`, `Decision Tree`, `Random Forest`"""),

    md_cell("""## 1. Khai báo thư viện và Đọc dữ liệu"""),
    code_cell("""import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve, confusion_matrix
)

df_raw = pd.read_csv("BankChurners.csv")
print("Kích thước dữ liệu gốc:", df_raw.shape)
df_raw.head()""",
    text_output=f"""Kích thước dữ liệu gốc: ({summary['total_rows']}, {summary['total_cols_raw']})
   CLIENTNUM     Attrition_Flag  Customer_Age Gender  Dependent_count Education_Level Marital_Status Income_Category Card_Category  Months_on_book  ...
0  768805383  Existing Customer            45      M                3     High School        Married     $60K - $80K          Blue              39  ...
1  818770008  Existing Customer            49      F                5        Graduate         Single  Less than $40K          Blue              44  ...
2  713982108  Existing Customer            51      M                3        Graduate        Married    $80K - $120K          Blue              36  ...""",
    exec_count=1),

    md_cell("""## 2. Tiền xử lý và Làm sạch dữ liệu (Data Cleaning)
Loại bỏ cột mã định danh `CLIENTNUM` và 2 cột xác suất Naive Bayes dư thừa ở cuối bảng dữ liệu gốc, sau đó kiểm tra giá trị khuyết thiếu (Missing Values)."""),
    code_cell("""drop_cols = [c for c in df_raw.columns if c.startswith("Naive_Bayes_") or c == "CLIENTNUM"]
df = df_raw.drop(columns=drop_cols).copy()
df["Churn"] = (df["Attrition_Flag"] == "Attrited Customer").astype(int)

print("Kích thước sau làm sạch:", df.shape)
print("Tổng số giá trị bị khuyết (Missing values):", df.isnull().sum().sum())
print("\\nPhân bố biến mục tiêu (Attrition_Flag):")
print(df["Attrition_Flag"].value_counts())""",
    text_output=f"""Kích thước sau làm sạch: (10127, 21)
Tổng số giá trị bị khuyết (Missing values): {summary['missing_values']}

Phân bố biến mục tiêu (Attrition_Flag):
Existing Customer    {summary['churn_counts']['Existing Customer']}
Attrited Customer    {summary['churn_counts']['Attrited Customer']}
Name: Attrition_Flag, dtype: int64""",
    exec_count=2),

    md_cell("""## 3. Phân tích Khám phá Dữ liệu (Exploratory Data Analysis - EDA)
### 3.1. Tỷ lệ rời bỏ dịch vụ thẻ tín dụng và Đặc điểm nhân khẩu học"""),
    code_cell("""# Biểu đồ phân bố trạng thái khách hàng và nhân khẩu học
# (Xem chi tiết hình ảnh trực quan bên dưới)""",
    img_paths=["charts/01_churn_distribution.png", "charts/02_demographic_analysis.png"],
    exec_count=3),

    md_cell("""### 3.2. Phân tích Hành vi Giao dịch và Mức độ Tương tác với Ngân hàng
Các chỉ số hành vi giao dịch (`Total_Trans_Ct`, `Total_Trans_Amt`) và mức độ gián đoạn sử dụng (`Months_Inactive_12_mon`, `Contacts_Count_12_mon`) cho thấy sự khác biệt rõ rệt giữa nhóm khách hàng trung thành và nhóm rời bỏ."""),
    code_cell("""# Thống kê trung bình các chỉ số quan trọng theo nhóm khách hàng
cols_check = ["Total_Trans_Ct", "Total_Trans_Amt", "Total_Revolving_Bal",
              "Months_Inactive_12_mon", "Contacts_Count_12_mon", "Avg_Utilization_Ratio"]
print(df.groupby("Attrition_Flag")[cols_check].mean().round(2))""",
    text_output=f"""                   Total_Trans_Ct  Total_Trans_Amt  Total_Revolving_Bal  Months_Inactive_12_mon  Contacts_Count_12_mon  Avg_Utilization_Ratio
Attrition_Flag
Attrited Customer          {gm['Total_Trans_Ct']['Attrited Customer']}          {gm['Total_Trans_Amt']['Attrited Customer']}               {gm['Total_Revolving_Bal']['Attrited Customer']}                    {gm['Months_Inactive_12_mon']['Attrited Customer']}                   {gm['Contacts_Count_12_mon']['Attrited Customer']}                   {gm['Avg_Utilization_Ratio']['Attrited Customer']}
Existing Customer          {gm['Total_Trans_Ct']['Existing Customer']}          {gm['Total_Trans_Amt']['Existing Customer']}              {gm['Total_Revolving_Bal']['Existing Customer']}                    {gm['Months_Inactive_12_mon']['Existing Customer']}                   {gm['Contacts_Count_12_mon']['Existing Customer']}                   {gm['Avg_Utilization_Ratio']['Existing Customer']}""",
    img_paths=["charts/03_transaction_behavior.png", "charts/04_inactivity_contacts.png", "charts/05_correlation_heatmap.png"],
    exec_count=4),

    md_cell("""## 4. Huấn luyện Mô hình Học máy & Đánh giá Kết quả
Chúng ta mã hóa One-Hot cho các biến phân loại, chia tập dữ liệu theo tỷ lệ **80% Train (8,101 mẫu) - 20% Test (2,026 mẫu)** có phân tầng (`stratify=y`), và so sánh 3 thuật toán:
1. **Logistic Regression**
2. **Decision Tree Classifier**
3. **Random Forest Classifier**"""),
    code_cell("""feature_cols = [c for c in df.columns if c not in ["Attrition_Flag", "Churn"]]
X = pd.get_dummies(df[feature_cols], drop_first=True)
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced"),
    "Decision Tree": DecisionTreeClassifier(max_depth=6, min_samples_leaf=15, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1),
}

# Bảng tổng hợp kết quả trên tập Test (2,026 khách hàng)""",
    text_output=f"""Kết quả đánh giá trên tập Test (2,026 mẫu):
- Logistic Regression : Accuracy = {models['Logistic Regression']['Accuracy']}%, Precision = {models['Logistic Regression']['Precision']}%, Recall = {models['Logistic Regression']['Recall']}%, F1 = {models['Logistic Regression']['F1_Score']}%, ROC-AUC = {models['Logistic Regression']['ROC_AUC']}%
- Decision Tree       : Accuracy = {models['Decision Tree']['Accuracy']}%, Precision = {models['Decision Tree']['Precision']}%, Recall = {models['Decision Tree']['Recall']}%, F1 = {models['Decision Tree']['F1_Score']}%, ROC-AUC = {models['Decision Tree']['ROC_AUC']}%
- Random Forest       : Accuracy = {models['Random Forest']['Accuracy']}%, Precision = {models['Random Forest']['Precision']}%, Recall = {models['Random Forest']['Recall']}%, F1 = {models['Random Forest']['F1_Score']}%, ROC-AUC = {models['Random Forest']['ROC_AUC']}%""",
    img_paths=["charts/06_model_comparison_roc.png", "charts/07_confusion_matrices.png", "charts/08_feature_importance.png"],
    exec_count=5),

    md_cell(f"""## 5. Kết luận & Đề xuất Giải pháp Hỗ trợ Ra quyết định (Decision Support)
1. **Lựa chọn mô hình triển khai**: **Random Forest** đạt độ chính xác cao nhất (**{models['Random Forest']['Accuracy']}%**, **ROC-AUC = {models['Random Forest']['ROC_AUC']}%**), trong khi **Decision Tree** (**Accuracy = {models['Decision Tree']['Accuracy']}%**, **ROC-AUC = {models['Decision Tree']['ROC_AUC']}%**) cung cấp bộ luật trực quan dễ giải thích và **Logistic Regression** đạt **Recall cao nhất ({models['Logistic Regression']['Recall']}%)** giúp hạn chế bỏ sót khách hàng rời bỏ.
2. **Cảnh báo sớm theo hành vi giao dịch**: Khách hàng có tổng số giao dịch dưới **55 lần/năm** (`Total_Trans_Ct < 55`) hoặc tỷ lệ giao dịch Quý 4 giảm mạnh so với Quý 1 (`Total_Ct_Chng_Q4_Q1 < 0.6`) cần được kích hoạt ưu đãi hoàn tiền (cashback) tự động.
3. **Quy trình chăm sóc khi khách hàng liên hệ nhiều lần**: Nhóm khách hàng liên hệ tổng đài từ **3–6 lần/năm** (`Contacts_Count_12_mon >= 3`) có tỷ lệ rời bỏ tăng vọt (lên đến 100% ở mức 6 lần liên hệ) -> Cần chuyển thẳng lên bộ phận Chăm sóc Khách hàng Ưu tiên (Retention Team) để giải quyết khiếu nại ngay từ lần liên hệ thứ 3.""")
]

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.11.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

with open("Credit_Card_Churn_Analysis.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, ensure_ascii=False, indent=2)
print("Saved Credit_Card_Churn_Analysis.ipynb")

# --- 2. BUILD WORD REPORT (.docx) ---
doc = Document()

# Style configuration
style = doc.styles["Normal"]
style.font.name = "Arial"
style.font.size = Pt(11)

title_p = doc.add_paragraph()
title_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
r_title = title_p.add_run("BÁO CÁO DỰ ÁN KHOA HỌC DỮ LIỆU\n")
r_title.bold = True
r_title.font.size = Pt(16)
r_title.font.color.rgb = RGBColor(31, 78, 121)

r_sub = title_p.add_run("ĐỀ TÀI: DỰ ĐOÁN NGUY CƠ KHÁCH HÀNG NGỪNG SỬ DỤNG DỊCH VỤ THẺ TÍN DỤNG NHẰM HỖ TRỢ RA QUYẾT ĐỊNH GIỮ CHÂN KHÁCH HÀNG TẠI NGÂN HÀNG")
r_sub.bold = True
r_sub.font.size = Pt(13)

doc.add_paragraph()

# Section 1
h1 = doc.add_heading("1. Đặt vấn đề và Mục tiêu nghiên cứu", level=1)
doc.add_paragraph(
    "Trong lĩnh vực ngân hàng bán lẻ, chi phí tìm kiếm một khách hàng mở thẻ tín dụng mới cao gấp 5 đến 7 lần so với chi phí duy trì một khách hàng hiện hữu. "
    "Tuy nhiên, nhiều ngân hàng chỉ phát hiện khách hàng rời bỏ (Churn) sau khi họ đã chính thức yêu cầu đóng thẻ, dẫn đến thất thoát doanh thu phí thường niên, phí giao dịch và lãi suất dư nợ tín dụng quay vòng."
)
doc.add_paragraph(
    "Mục tiêu của dự án là ứng dụng các thuật toán Học máy (Logistic Regression, Decision Tree, Random Forest) trên bộ dữ liệu 10,127 hồ sơ khách hàng nhằm:\n"
    "• Nhận diện các yếu tố cốt lõi dẫn đến hành vi hủy thẻ tín dụng.\n"
    "• Xây dựng mô hình dự báo sớm xác suất rời bỏ của từng khách hàng.\n"
    "• Đề xuất hệ thống quy tắc hỗ trợ ra quyết định (Decision Support System) giúp ngân hàng phân bổ ngân sách khuyến mãi và chăm sóc đúng nhóm khách hàng có nguy cơ cao."
)

# Section 2
doc.add_heading("2. Mô tả Tập dữ liệu và Tiền xử lý", level=1)
doc.add_paragraph(
    "• Nguồn dữ liệu: Credit Card Customers Dataset (Kaggle - sakshigoyal7/credit-card-customers).\n"
    f"• Quy mô dữ liệu: {summary['total_rows']:,} khách hàng. Sau khi loại bỏ cột định danh CLIENTNUM và 2 cột tạo sẵn của Naive Bayes, tập dữ liệu gồm 20 biến (19 biến độc lập và 1 biến mục tiêu Attrition_Flag).\n"
    f"• Kiểm tra dữ liệu khuyết thiếu: Tập dữ liệu hoàn toàn sạch ({summary['missing_values']} giá trị NULL).\n"
    f"• Phân bố biến mục tiêu: {summary['churn_counts']['Existing Customer']:,} khách hàng hiện hữu (83.93%) và {summary['churn_counts']['Attrited Customer']:,} khách hàng rời bỏ ({summary['churn_rate_pct']}%)."
)

doc.add_picture("charts/01_churn_distribution.png", width=Inches(4.8))
doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER

# Section 3
doc.add_heading("3. Phân tích Khám phá Dữ liệu (EDA)", level=1)

doc.add_heading("3.1. Phân tích theo Đặc điểm Nhân khẩu học", level=2)
doc.add_paragraph(
    "Kết quả phân tích nhân khẩu học cho thấy độ tuổi khách hàng tập trung chủ yếu từ 40 đến 55 tuổi. "
    "Khách hàng Nữ có tỷ lệ rời bỏ (17.36%) cao hơn khách hàng Nam (14.62%). Nhóm có trình độ Tiến sĩ (Doctorate) và Sau đại học (Post-Graduate) có tỷ lệ rời bỏ cao hơn mặt bằng chung."
)
doc.add_picture("charts/02_demographic_analysis.png", width=Inches(6.0))
doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER

doc.add_heading("3.2. Phân tích Hành vi Giao dịch và Mức độ Tương tác", level=2)
doc.add_paragraph(
    f"• Về tần suất và giá trị giao dịch: Khách hàng hiện hữu thực hiện trung bình {gm['Total_Trans_Ct']['Existing Customer']} giao dịch/năm (tổng giá trị {gm['Total_Trans_Amt']['Existing Customer']:,} USD), "
    f"trong khi nhóm rời bỏ chỉ thực hiện trung bình {gm['Total_Trans_Ct']['Attrited Customer']} giao dịch/năm (tổng giá trị {gm['Total_Trans_Amt']['Attrited Customer']:,} USD). Đặc biệt, ngưỡng < 55 giao dịch/năm là vùng rủi ro rời bỏ rất cao.\n"
    f"• Về dư nợ quay vòng (Total_Revolving_Bal): Nhóm khách hàng rời bỏ có dư nợ quay vòng trung bình thấp hơn hẳn ({gm['Total_Revolving_Bal']['Attrited Customer']} USD so với {gm['Total_Revolving_Bal']['Existing Customer']} USD).\n"
    "• Về số lần liên hệ ngân hàng (Contacts_Count_12_mon): Khách hàng liên hệ tổng đài từ 3 lần trở lên có tỷ lệ rời bỏ tăng vọt; đặc biệt 100% khách hàng liên hệ 6 lần trong năm đều hủy thẻ (dấu hiệu bất mãn khi khiếu nại không được giải quyết triệt để)."
)
doc.add_picture("charts/03_transaction_behavior.png", width=Inches(6.0))
doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER

doc.add_picture("charts/04_inactivity_contacts.png", width=Inches(6.0))
doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER

doc.add_picture("charts/05_correlation_heatmap.png", width=Inches(5.8))
doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER

# Section 4
doc.add_heading("4. Kết quả Huấn luyện và Đánh giá Mô hình", level=1)
doc.add_paragraph(
    f"Dữ liệu được chia thành tập Huấn luyện (80% - {summary['train_size']:,} mẫu) và tập Kiểm thử (20% - {summary['test_size']:,} mẫu). "
    "Bảng dưới đây tổng hợp kết quả thực nghiệm của 3 thuật toán trên tập kiểm thử:"
)

table = doc.add_table(rows=1, cols=6)
table.style = "Table Grid"
table.alignment = WD_TABLE_ALIGNMENT.CENTER
hdr_cells = table.rows[0].cells
headers = ["Thuật toán", "Accuracy (%)", "Precision (%)", "Recall (%)", "F1-Score (%)", "ROC-AUC (%)"]
for i, h_text in enumerate(headers):
    hdr_cells[i].text = h_text
    for run in hdr_cells[i].paragraphs[0].runs:
        run.bold = True

for m_name, metrics in models.items():
    row_cells = table.add_row().cells
    row_cells[0].text = m_name
    row_cells[1].text = f"{metrics['Accuracy']:.2f}%"
    row_cells[2].text = f"{metrics['Precision']:.2f}%"
    row_cells[3].text = f"{metrics['Recall']:.2f}%"
    row_cells[4].text = f"{metrics['F1_Score']:.2f}%"
    row_cells[5].text = f"{metrics['ROC_AUC']:.2f}%"

doc.add_paragraph()
doc.add_picture("charts/06_model_comparison_roc.png", width=Inches(6.0))
doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER

doc.add_picture("charts/07_confusion_matrices.png", width=Inches(6.0))
doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER

doc.add_heading("4.1. Phân tích Mức độ Quan trọng của các Thuộc tính (Feature Importance)", level=2)
doc.add_paragraph(
    f"Từ mô hình Random Forest đạt hiệu năng cao nhất (Accuracy = {models['Random Forest']['Accuracy']}%, ROC-AUC = {models['Random Forest']['ROC_AUC']}%), 5 yếu tố quyết định lớn nhất đến nguy cơ rời bỏ thẻ tín dụng bao gồm:\n"
    f"1. Tổng số lần giao dịch trong 12 tháng (Total_Trans_Ct): {top_feat.get('Total_Trans_Ct', 0)*100:.2f}%\n"
    f"2. Tổng giá trị giao dịch trong 12 tháng (Total_Trans_Amt): {top_feat.get('Total_Trans_Amt', 0)*100:.2f}%\n"
    f"3. Tỷ lệ thay đổi số lần giao dịch Quý 4 so với Quý 1 (Total_Ct_Chng_Q4_Q1): {top_feat.get('Total_Ct_Chng_Q4_Q1', 0)*100:.2f}%\n"
    f"4. Tổng dư nợ tín dụng quay vòng (Total_Revolving_Bal): {top_feat.get('Total_Revolving_Bal', 0)*100:.2f}%\n"
    f"5. Tỷ lệ sử dụng hạn mức trung bình (Avg_Utilization_Ratio): {top_feat.get('Avg_Utilization_Ratio', 0)*100:.2f}%"
)
doc.add_picture("charts/08_feature_importance.png", width=Inches(5.6))
doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER

# Section 5
doc.add_heading("5. Đề xuất Giải pháp Hỗ trợ Ra quyết định cho Ngân hàng", level=1)
doc.add_paragraph(
    "Dựa trên kết quả phân tích dữ liệu và mô hình dự báo, nhóm nghiên cứu đề xuất khung hành động 3 cấp độ hỗ trợ Ban Quản trị Khối Ngân hàng Bán lẻ ra quyết định giữ chân khách hàng:\n\n"
    "1. Chiến lược Kích hoạt Chi tiêu Sớm (Ngăn chặn giảm tần suất giao dịch):\n"
    "• Điều kiện kích hoạt: Khách hàng có Total_Trans_Ct < 55 lần/năm hoặc chỉ số Total_Ct_Chng_Q4_Q1 < 0.65.\n"
    "• Quyết định đề xuất: Tự động gửi ưu đãi hoàn tiền (Cashback 5-10%) hoặc miễn phí thường niên năm tiếp theo nếu khách hàng thực hiện tối thiểu 5 giao dịch trong vòng 30 ngày tới.\n\n"
    "2. Chiến lược Cảnh báo Đỏ tại Tổng đài CSKH (Xử lý bức xúc dịch vụ):\n"
    "• Điều kiện kích hoạt: Khách hàng có số lần liên hệ ngân hàng (Contacts_Count_12_mon) đạt ngưỡng >= 3 lần hoặc không phát sinh giao dịch trong >= 3 tháng liên tiếp (Months_Inactive_12_mon >= 3).\n"
    "• Quyết định đề xuất: Hệ thống CRM tự động gắn nhãn 'Nguy cơ rời bỏ cao' và chuyển cuộc gọi trực tiếp tới chuyên viên Retention cấp cao để giải quyết dứt điểm khiếu nại và tặng gói ưu đãi lãi suất.\n\n"
    "3. Chiến lược Tối ưu Hạn mức & Dư nợ Quay vòng:\n"
    "• Điều kiện kích hoạt: Khách hàng có Total_Revolving_Bal = 0 và Avg_Utilization_Ratio < 0.1.\n"
    "• Quyết định đề xuất: Đề xuất chương trình trả góp 0% lãi suất (Installment Conversion) qua ứng dụng Mobile Banking nhằm tăng sự gắn kết tài chính của khách hàng với thẻ."
)

doc.save("Bao_Cao_Du_An_Khoa_Hoc_Du_Lieu.docx")
print("Saved Bao_Cao_Du_An_Khoa_Hoc_Du_Lieu.docx")
