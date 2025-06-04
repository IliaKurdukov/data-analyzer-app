import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import pyreadstat
import seaborn as sns

st.title("📊 Персональный анализатор данных")

import tempfile
import pyreadstat
import streamlit as st

uploaded_file = st.file_uploader("Загрузите SAV файл (SPSS)", type="sav")

if uploaded_file:
    try:
        # Создаем временный файл
        with tempfile.NamedTemporaryFile(delete=False, suffix=".sav") as tmp:
            # Записываем содержимое загруженного файла во временный файл
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name
        
        # Читаем временный файл
        df, meta = pyreadstat.read_sav(tmp_path)
        st.success(f"Данные успешно загружены! Записей: {len(df)}")
        
        # Удаляем временный файл (опционально)
        import os
        os.unlink(tmp_path)

        # Автоматическое определение столбцов
        cols = df.columns
        if len(cols) == 0:
            st.error("В файле нет столбцов")
        else:
            col = st.selectbox("Выберите столбец для анализа", cols)

            n_resp = df.shape[0]

            def is_multi_response(col):
              '''
              Проверяет, вопрос с множественным ответом (True) или нет (False)
              '''
              if '_' in col:
                prefix = col.split('_')[0]
                matching_keys = [key for key in meta.column_names_to_labels if key.startswith(prefix)]
                matching_labels = set(meta.column_names_to_labels[key] for key in matching_keys)
                return len(matching_keys) > 1 and len(matching_labels) == 1
              return False
            
            def process_multi_response_1(col):
              '''
              Объединяет столбцы в вопросах с множественным выбором ответа (для 1 вопроса)
              '''
              if is_multi_response(col):
                data = pd.DataFrame()
                question = meta.column_names_to_labels[col]
                for key in meta.column_names_to_labels:
                  if meta.column_names_to_labels[key] == question:
                    data = pd.concat([data, df[key]],ignore_index=True)
              else:
                data = df[col]
              return(data)

            def get_barplot(col):
              data = process_multi_response_1(col)
              plot_df=data.value_counts().div(n_resp/100)
              plot_df = pd.DataFrame(plot_df)
              plot_df.reset_index(names=['Ответ'], inplace=True)
              plot_df['ans'] = plot_df['Ответ']
              plot_df['ans'] = plot_df['ans'].apply(lambda x: 0 if x < 90 else x)
              plot_df.sort_values(by=['ans', 'count'], ascending = [True, False], inplace=True)
              plot_df['Ответ'] = plot_df['Ответ'].map(lambda x: meta.variable_value_labels[col][x])
              plt.figure(figsize=(8, 4))
              ax = sns.barplot(x=plot_df['count'], y=plot_df['Ответ'], color = '#E62083')
              ax.spines['top'].set_visible(False)
              ax.spines['right'].set_visible(False)
              ax.spines['bottom'].set_visible(False)
              ax.bar_label(ax.containers[0], label_type = 'edge', padding=5, fontsize=12, fmt = '%.1f%%')
              plt.xticks([])
              plt.yticks(fontsize=12)
              question = meta.column_names_to_labels[col]
              plt.title(f'{question}', fontsize=12, pad=10)
              plt.xlabel('', fontsize=1)
              plt.ylabel('')

            # Диаграмма
            #fig, ax = plt.subplots()
            #ax.hist(df[selected_col], bins=20)
            #st.pyplot(fig)
            get_barplot(col)

    except Exception as e:
        st.error(f"Ошибка: {str(e)}")
