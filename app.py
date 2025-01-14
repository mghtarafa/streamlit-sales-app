import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt
from io import StringIO

# بيانات تسجيل الدخول (يمكنك تغييرها)
USERNAME = "Tarafamgh"
PASSWORD = "qmd2099"

def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if (
            st.session_state["username"] == USERNAME
            and st.session_state["password"] == PASSWORD
        ):
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # don't store password
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First run, show inputs for username and password.
        st.text_input(label="اسم المستخدم", key="username")
        st.text_input(
            label="كلمة المرور", key="password", type="password", on_change=password_entered
        )
        return False
    elif not st.session_state["password_correct"]:
        # Password not correct, show input + error.
        st.text_input(label="اسم المستخدم", key="username")
        st.text_input(
            label="كلمة المرور", key="password", type="password", on_change=password_entered
        )
        st.error("اسم المستخدم أو كلمة المرور غير صحيحة")
        return False
    else:
        # Password correct.
        return True


if check_password():

    # عنوان التطبيق
    st.title("تطبيق التنبؤ بالمبيعات")

    # وصف التطبيق
    st.write("هذا التطبيق يتيح لك تحليل بيانات المبيعات والتنبؤ بالمبيعات المستقبلية باستخدام نموذج ARIMA.")

    # زر لتحميل البيانات
    uploaded_file = st.file_uploader("اختر ملف CSV أو Excel", type=["csv", "xlsx"])

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith('.xlsx'):
                df = pd.read_excel(uploaded_file)
            
            st.write("تم تحميل البيانات بنجاح!")

            # معالجة البيانات
            df['Order_Date'] = pd.to_datetime(df['Order_Date'])
            df['Discount_Applied'] = df['Discount_Applied'].astype(bool)
            daily_sales = df.groupby('Order_Date')['Total_Price'].sum()
            df_time_series = pd.DataFrame({'Order_Date': daily_sales.index, 'Total_Sales': daily_sales.values})
            df_time_series = df_time_series.set_index('Order_Date')
            train_size = int(len(df_time_series) * 0.8)
            train_data = df_time_series[:train_size]
            test_data = df_time_series[train_size:]

            # شريط التمرير لتعديل قيم p, d, q
            p = st.slider("قيمة p", 0, 10, 5)
            d = st.slider("قيمة d", 0, 5, 1)
            q = st.slider("قيمة q", 0, 5, 0)


            # زر لتشغيل النموذج
            if st.button("تشغيل نموذج التنبؤ"):
                # تدريب نموذج ARIMA
                model = ARIMA(train_data['Total_Sales'], order=(p, d, q))
                model_fit = model.fit()

                # التنبؤ بالمبيعات على بيانات الاختبار
                predictions = model_fit.predict(start=len(train_data), end=len(df_time_series)-1)

                # حساب RMSE
                rmse = sqrt(mean_squared_error(test_data['Total_Sales'], predictions))

                # عرض النتائج
                st.write("RMSE:", rmse)
                st.write("قيم التنبؤ:")
                st.write(predictions)

                # رسم بياني للمبيعات الفعلية والمتوقعة
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(test_data.index, test_data['Total_Sales'], label='المبيعات الفعلية', marker='o')
                ax.plot(test_data.index, predictions, label='المبيعات المتوقعة', marker='x')
                ax.set_title('المبيعات الفعلية مقابل المبيعات المتوقعة')
                ax.set_xlabel('تاريخ الطلب')
                ax.set_ylabel('إجمالي المبيعات')
                ax.legend()
                ax.grid(True)
                st.pyplot(fig)
                
        except Exception as e:
            st.error(f"حدث خطأ أثناء قراءة الملف أو معالجة البيانات: {e}")
