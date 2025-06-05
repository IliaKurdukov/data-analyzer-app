import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyreadstat
import seaborn as sns
import streamlit as st
import tempfile
import re

from math import log2
from scipy.stats import chi2_contingency
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportions_ztest
from scipy.stats import norm
from typing import Dict

st.title("📊 Анализ количественных данных")

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
        old_meta = meta
        n_resp = len(df)
        st.success(f"Данные успешно загружены! Записей: {n_resp}")

        # Удаляем временный файл (опционально)
        import os
        os.unlink(tmp_path)
       
       # Автоматическое определение столбцов
        list_of_questions = []
        for key in old_meta.column_names_to_labels:
          if old_meta.column_names_to_labels[key] in list_of_questions:
            continue
          else:
            list_of_questions.append(old_meta.column_names_to_labels[key])
        meta_inside_out = {}
        for question in list_of_questions:
          for key in old_meta.column_names_to_labels:
            if old_meta.column_names_to_labels[key] == question:
              meta_inside_out[question] = key
              break
        
        if len(list_of_questions) == 0:
            st.error("В файле нет столбцов")
        else:
            question = st.selectbox("Выберите столбец для вывода распределения", list_of_questions)
            col = 

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

            def unify_questions(labels_dict: Dict[str, str]) -> Dict[str, str]:
                """
                Унифицирует вопросы с одинаковыми числовыми префиксами.
                Например, вопросы с префиксами "13.А.", "13.Б." получат текст первого вопроса.
                """
                # Сначала извлекаем числовые префиксы (например, "13" из "13.А.")
                prefix_to_text = {}
                for col, label in labels_dict.items():
                    match = re.search(r'^(\d+)', label)  # Ищем цифры в начале
                    if match:
                        prefix = match.group(1)
                        if prefix not in prefix_to_text:
                            # Запоминаем первый встретившийся текст для этого префикса
                            prefix_to_text[prefix] = label

                # Заменяем метки вопросов с одинаковыми префиксами
                unified = {}
                for col, label in labels_dict.items():
                    match = re.search(r'^(\d+)', label)
                    if match:
                        prefix = match.group(1)
                        if prefix in prefix_to_text:
                            unified[col] = prefix_to_text[prefix]
                    else:
                        unified[col] = label
                return unified

            def remove_prefixes(labels_dict: Dict[str, str]) -> Dict[str, str]:
                """
                Удаляет префиксы вида "SEX.", "1.", "Q1." из меток вопросов.
                """
                cleaned = {}
                for col, label in labels_dict.items():
                    # Удаляем префиксы типа "SEX.", "1.", "Q1.", "13.А." и т.д.
                    cleaned_label = re.sub(r'^([a-zA-Zа-яА-Я0-9]+\.\s*)+', '', label)
                    cleaned[col] = cleaned_label.strip()
                return cleaned

            def clean_question_labels(meta, mode: str = 'both'):
                """
                ЗАМЕНЯЕТ исходные метаданные вопросов в DataFrame на очищенные.

                Параметры:
                    meta: Метаданные (будет изменен)
                    mode: 'prefix' - только удаление префиксов,
                          'unify' - только унификация,
                          'both' - оба преобразования
                """
                labels = meta.column_names_to_labels.copy()

                if mode in ('unify', 'both'):
                    labels = unify_questions(labels)
                if mode in ('prefix', 'both'):
                    labels = remove_prefixes(labels)

                meta.column_names_to_labels = labels

            def clean_value_labels(meta):
                """
                ЗАМЕНЯЕТ исходные метаданные ответов в DataFrame на очищенные для значений ≥90 (технических ответов).
                Правила замены:
                    - Удаляет префиксы вида '(ЗАЧИТАТЬ ПОСЛЕ ПАУЗЫ)', '(НЕ ЗАЧИТЫВАТЬ)' и т.п.
                    - Оставляет только слово 'Другое', удаляя всё после него
                """
                replacement_rules = {
                    r'\([^)]+\)\s*': '',  # Удаляет все скобочные префиксы
                    r'(Другое).*': r'\1' # Оставляет только первое вхождение "Другое"
                }

                for var_name, value_labels in meta.variable_value_labels.items():
                    updated_labels = {}
                    for value, label in value_labels.items():
                        if isinstance(value, (int, float)) and value >= 90: # Проверяем, что технический ответ
                            new_label = label
                            for pattern, repl in replacement_rules.items():
                                new_label = re.sub(pattern, repl, new_label)
                            updated_labels[value] = new_label.strip()
                        else:
                            updated_labels[value] = label
                    meta.variable_value_labels[var_name] = updated_labels

            clean_question_labels(meta)
            clean_value_labels(meta)
            
            #def get_barplot1(col):
            data = process_multi_response_1(col)
            plot_df=data.value_counts().div(n_resp/100)
            plot_df = pd.DataFrame(plot_df)
            plot_df.reset_index(names=['Ответ'], inplace=True)
            plot_df['ans'] = plot_df['Ответ']
            plot_df['ans'] = plot_df['ans'].apply(lambda x: 0 if x < 90 else x)
            plot_df.sort_values(by=['ans', 'count'], ascending = [True, False], inplace=True)
            if col in meta.variable_value_labels:
              plot_df['Ответ'] = plot_df['Ответ'].map(lambda x: meta.variable_value_labels[col][x])

            plt.figure(figsize=(5, 4))
            ax = sns.barplot(x=plot_df['count'], y=plot_df['Ответ'].astype(str), color = '#E62083') #E62083 #12AFFF
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            #ax.spines['left'].set_visible(False)

            ax.bar_label(ax.containers[0], label_type = 'edge', padding=5, fontsize=14, fmt = '%.1f%%')
            plt.xticks([])
            plt.yticks(fontsize=10)

            if len(question) > 30:
              question = question.replace('? ', '?\n')
              question = question.replace('. ', '.\n')
            plt.title(f'{question}', fontsize=14, pad=10)
            plt.xlabel('', fontsize=1)
            plt.ylabel('', fontsize=1);

            # Диаграмма
            st.pyplot(fig)

    except Exception as e:
        st.error(f"Ошибка: {str(e)}")
