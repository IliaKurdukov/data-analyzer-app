import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.title("📊 Персональный анализатор данных")

uploaded_file = st.file_uploader("Загрузите CSV файл")
if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        st.success("Данные успешно загружены!")
        
        # Автоматическое определение столбцов
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) == 0:
            st.error("В файле нет числовых столбцов")
        else:
            selected_col = st.selectbox("Выберите столбец для анализа", numeric_cols)
            
            # Гистограмма
            fig, ax = plt.subplots()
            ax.hist(df[selected_col], bins=20)
            st.pyplot(fig)
            
    except Exception as e:
        st.error(f"Ошибка: {str(e)}")
