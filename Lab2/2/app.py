from flask import Flask, render_template, request
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.naive_bayes import GaussianNB

# Khởi tạo Flask app
app = Flask(__name__)

# Bước 1: Đọc dữ liệu và xử lý mô hình
data = pd.read_csv('drug.csv')

# Label Encoding cho các cột phân loại
label_encoder_sex = LabelEncoder()
label_encoder_bp = LabelEncoder()
label_encoder_cholesterol = LabelEncoder()

label_encoder_sex.fit(['M', 'F'])
label_encoder_bp.fit(['HIGH', 'NORMAL', 'LOW'])
label_encoder_cholesterol.fit(['HIGH', 'NORMAL'])

# Chuyển đổi biến phân loại thành dạng số
data['Sex'] = label_encoder_sex.transform(data['Sex'])
data['BP'] = label_encoder_bp.transform(data['BP'])
data['Cholesterol'] = label_encoder_cholesterol.transform(data['Cholesterol'])

# Chia tập dữ liệu
X = data[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']]
y = data['Drug']

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Huấn luyện mô hình Naive Bayes
gnb = GaussianNB()
gnb.fit(X, y)

# Hàm để dự đoán thuốc dựa trên dữ liệu người dùng
def predict_drug(age, sex_input, bp_input, cholesterol_input):
    sex = label_encoder_sex.transform([sex_input])[0]
    bp = label_encoder_bp.transform([bp_input])[0]
    cholesterol = label_encoder_cholesterol.transform([cholesterol_input])[0]
    
    # Tìm Na_to_K từ dữ liệu
    possible_entries = data[(data['Age'] == age) & (data['Sex'] == sex) & (data['BP'] == bp) & (data['Cholesterol'] == cholesterol)]
    
    if possible_entries.empty:
        return None, "No matching data found."
    
    na_to_k = possible_entries['Na_to_K'].values[0]
    
    # Chuẩn hóa đầu vào người dùng
    input_data = pd.DataFrame([[age, sex, bp, cholesterol, na_to_k]], columns=['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K'])
    input_data = scaler.transform(input_data)
    
    # Dự đoán
    predicted_drug = gnb.predict(input_data)
    
    return predicted_drug[0], None

# Route cho trang chủ
@app.route('/')
def index():
    return render_template('index.html')

# Route xử lý khi người dùng gửi form
@app.route('/predict', methods=['POST'])
def predict():
    age = int(request.form['age'])
    sex = request.form['sex']
    bp = request.form['bp']
    cholesterol = request.form['cholesterol']
    
    predicted_drug, error_message = predict_drug(age, sex, bp, cholesterol)
    
    if error_message:
        return render_template('index.html', error=error_message)
    
    return render_template('index.html', age=age, sex=sex, bp=bp, cholesterol=cholesterol, predicted_drug=predicted_drug)

if __name__ == '__main__':
    app.run(debug=True)
