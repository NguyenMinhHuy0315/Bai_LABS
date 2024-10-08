import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report

# Bước 1: Đọc dữ liệu
data = pd.read_csv('Drug.csv')

# Bước 2: Tiền xử lý dữ liệu
label_encoder_sex = LabelEncoder()
label_encoder_bp = LabelEncoder()
label_encoder_cholesterol = LabelEncoder()

# Fit LabelEncoder với tất cả các giá trị có thể
label_encoder_sex.fit(['M', 'F'])
label_encoder_bp.fit(['HIGH', 'NORMAL', 'LOW'])
label_encoder_cholesterol.fit(['HIGH', 'NORMAL'])

# Chuyển đổi biến phân loại thành dạng số
data['Sex'] = label_encoder_sex.transform(data['Sex'])
data['BP'] = label_encoder_bp.transform(data['BP'])
data['Cholesterol'] = label_encoder_cholesterol.transform(data['Cholesterol'])

# Chia tập dữ liệu thành biến độc lập và biến mục tiêu
X = data[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']]
y = data['Drug']

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Bước 3: Huấn luyện mô hình Naive Bayes với phân phối Gaussian
gnb = GaussianNB()
gnb.fit(X_train, y_train)

# Dự đoán trên tập kiểm tra
y_pred = gnb.predict(X_test)

# Bước 4: Đánh giá mô hình
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Hàm để nhập dữ liệu từ người dùng
def get_user_input():
    age = input("Nhập Age (ví dụ: 25, 45): ")
    while not age.isdigit():  # Kiểm tra xem người dùng có nhập số hay không
        print("Vui lòng nhập Age dưới dạng số.")
        age = input("Nhập Age (ví dụ: 25, 45): ")
    age = int(age)

    sex_input = input("Nhập Sex (M/F): ").strip().upper()
    while sex_input not in ['M', 'F']:  # Kiểm tra xem người dùng có nhập M hoặc F hay không
        print("Vui lòng nhập Sex là M hoặc F.")
        sex_input = input("Nhập Sex (M/F): ").strip().upper()
    sex = label_encoder_sex.transform([sex_input])[0]

    bp_input = input("Nhập BP (HIGH/NORMAL/LOW): ").strip().upper()
    while bp_input not in ['HIGH', 'NORMAL', 'LOW']:  # Kiểm tra xem người dùng có nhập HIGH, NORMAL, hoặc LOW hay không
        print("Vui lòng nhập BP là HIGH, NORMAL hoặc LOW.")
        bp_input = input("Nhập BP (HIGH/NORMAL/LOW): ").strip().upper()
    bp = label_encoder_bp.transform([bp_input])[0]

    cholesterol_input = input("Nhập Cholesterol (HIGH/NORMAL): ").strip().upper()
    while cholesterol_input not in ['HIGH', 'NORMAL']:  # Kiểm tra xem người dùng có nhập HIGH hoặc NORMAL hay không
        print("Vui lòng nhập Cholesterol là HIGH hoặc NORMAL.")
        cholesterol_input = input("Nhập Cholesterol (HIGH/NORMAL): ").strip().upper()
    cholesterol = label_encoder_cholesterol.transform([cholesterol_input])[0]

    # Tìm Na_to_K từ dữ liệu gốc dựa trên Age, Sex, BP và Cholesterol
    possible_entries = data[(data['Age'] == age) & (data['Sex'] == sex) & (data['BP'] == bp) & (data['Cholesterol'] == cholesterol)]

    if possible_entries.empty:
        print("Không tìm thấy dữ liệu phù hợp cho Age, Sex, BP và Cholesterol đã nhập. Vui lòng kiểm tra lại.")
        return None

    # Lấy Na_to_K từ dữ liệu
    na_to_k = possible_entries['Na_to_K'].values[0]

    return age, sex_input, bp_input, cholesterol_input, sex, bp, cholesterol, na_to_k

# Nhận dữ liệu từ người dùng
user_data = get_user_input()

if user_data:
    age, sex_input, bp_input, cholesterol_input, sex, bp, cholesterol, na_to_k = user_data

    # Chuyển đổi dữ liệu người dùng thành DataFrame với các tên cột phù hợp
    input_data = pd.DataFrame([[age, sex, bp, cholesterol, na_to_k]], columns=['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K'])
    input_data = scaler.transform(input_data)

    # Dự đoán dựa trên dữ liệu người dùng
    predicted_drug = gnb.predict(input_data)

    # In ra kết quả
    print(f"Age: {age}, Sex: {sex_input}, BP: {bp_input}, Cholesterol: {cholesterol_input}, Na_to_K: {na_to_k}, Predicted Drug: {predicted_drug[0]}")

