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
            col = meta_inside_out[question]

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
            
            #def get_barplot(col):
            data = process_multi_response_1(col)
            plot_df=data.value_counts().div(n_resp/100)
            plot_df = pd.DataFrame(plot_df)
            plot_df.reset_index(names=['Ответ'], inplace=True)
            plot_df['ans'] = plot_df['Ответ']
            plot_df['ans'] = plot_df['ans'].apply(lambda x: 0 if x < 90 else x)
            plot_df.sort_values(by=['ans', 'count'], ascending = [True, False], inplace=True)
            if col in meta.variable_value_labels:
              plot_df['Ответ'] = plot_df['Ответ'].map(lambda x: meta.variable_value_labels[col][x])
            
            fig, ax = plt.subplots()
            #plt.figure(figsize=(5, 4))
            ax = sns.barplot(x=plot_df['count'], y=plot_df['Ответ'].astype(str), color = '#E62083') #E62083 #12AFFF
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.bar_label(ax.containers[0], label_type = 'edge', padding=5, fontsize=14, fmt = '%.1f%%')
            plt.xticks([])
            plt.yticks(fontsize=10)
            question = meta.column_names_to_labels[col] # текст вопроса
            if len(question) > 30:
              question = question.replace('? ', '?\n')
              question = question.replace('. ', '.\n')
            plt.title(f'{question}', fontsize=14, pad=10)
            plt.xlabel('', fontsize=1)
            plt.ylabel('', fontsize=1);

            # Диаграмма
            st.pyplot(fig)

            def convert_to_intervals(col, df = df):
                '''
               Преобразовывает непрерывные значения в интервальные
                '''
                if col not in meta.variable_value_labels and df[col].nunique() > 5:
                    if col == 'AGE':
                        bins = [0, 18, 25, 35, 45, 55, 65, float('inf')]
                        labels = ['до 18', '18 - 24', '25 - 34', '35 - 44', '45 - 54', '55 - 64', '65+']
                        return pd.cut(df[col], bins = bins, labels=labels, right=False).astype('str')
                    else:
                        binned_col = pd.cut(df[col], bins = 5, right = False, duplicates='drop')
                        return binned_col.apply(lambda x: f'{round(x.left)} - {round(x.right)}').astype('str')
                return df[col]
            
            def process_multi_response(col1, col2):
                '''
                Объединяет столбцы в вопросах с множественным выбором ответов
                '''
                data = pd.DataFrame()
                if is_multi_response(col1) and is_multi_response(col2):
                    raise KeyError('Не поддерживается пересечение 2 вопросов с множественными ответами')
                elif is_multi_response(col1) and not is_multi_response(col2) :
                    for i in [key for key in meta.column_names_to_labels.keys() if key.__contains__(col1.split('_')[0])]:
                         fict_df = df[[col2, i]]
                         fict_df.columns = [col2, col1]
                         data = pd.concat([data, fict_df], ignore_index=True)
                elif not is_multi_response(col1) and is_multi_response(col2):
                    for i in [key for key in meta.column_names_to_labels.keys() if key.__contains__(col2.split('_')[0])]:
                        fict_df = df[[col1, i]]
                        fict_df.columns = [col1, col2]
                        data = pd.concat([data, fict_df], ignore_index=True)
                else:
                    data = df[[col1, col2]]
                return data

            def apply_value_labels(series, is_index=True):
                '''
                Применяет метки значений к столбцам таблицы, предварительно удалив из метаданных технические слова
                '''
                words_to_remove = ['(НЕ ЗАЧИТЫВАТЬ) ', '(ЗАЧИТАТЬ ПОСЛЕ ПАУЗЫ) ', '(ЗАЧИТАТЬ) ', '(ПОСЛЕ ПАУЗЫ) ']
                pattern = "|".join(map(re.escape, words_to_remove))

                for question, response in meta.variable_value_labels.items():
                    for key, value in response.items():
                        response[key] = re.sub(pattern, '', value)

                if series.name in meta.variable_value_labels:
                    return series.map(lambda x: meta.variable_value_labels[series.name].get(x, x))
                return series

            def create_crosstab(col1, col2, adjustment_type = 'holm', threshhold=0.05):

                """
                Создает таблицу сопряженности между двумя переменными с нормировкой по столбцам и вычисляет
                статистические различия с поправкой на множественные сравнения.

                Args:
                    col1 (pd.Series): Первый столбец.
                    col2 (pd.Series): Второй столбец.
                    adjustment_type (str or None): Метод поправки на множественные сравнения ('holm', 'bonferroni', 'fdr_bh' и др.). Если None, поправка не применяется. По умолчанию 'holm'.
                    threshhold (float): Порог уровня значимости для вывода результатов тестирования. По умолчанию 0.05.

                Returns:
                    dict: Словарь с двумя ключами:
                        - 'table': стилизованная таблица сопряженности (pandas Styler object).
                        - 'notes': текстовый вывод с результатами проверки статистических гипотез
                                   и дополнительными комментариями.
                """
                data = process_multi_response(col1, col2)

                transformed_col1 = convert_to_intervals(col1, data)
                transformed_col2 = convert_to_intervals(col2, data)

                # Для вывода таблицы
                cross_df = pd.crosstab(transformed_col1,
                                       transformed_col2,
                                       margins=True,
                                       margins_name='Всего',
                                       normalize='columns')

                cross_df.index = apply_value_labels(cross_df.index)
                cross_df.columns = apply_value_labels(cross_df.columns)

                table = cross_df.style.format(lambda x: round(x*100, 1))\
                                      .background_gradient(cmap = 'Purples')\
                                      .set_caption(f'Таблица сопряженности между {col1} и {col2}')

                contingency_table = pd.crosstab(transformed_col1, transformed_col2)
                p_values = []
                comparisons = []

                contingency_table.index = apply_value_labels(contingency_table.index)
                contingency_table.columns = apply_value_labels(contingency_table.columns)

                for i in range(contingency_table.shape[1]):
                   for j in range(contingency_table.shape[0]):
                        for k in range(j+1, contingency_table.shape[0]):
                            success = [contingency_table.iloc[j, i], contingency_table.iloc[k, i]]
                            nobs = [contingency_table.iloc[j, :].sum(axis=0), contingency_table.iloc[k, :].sum(axis=0)]
                            if min(nobs) > 0:
                                _, p_value = proportions_ztest(success, nobs)
                                p_values.append(p_value)
                                comparisons.append(f'{contingency_table.index[j]} vs {contingency_table.index[k]} в группе {contingency_table.columns[i]}')

                if p_values:
                    if adjustment_type is not None:
                        reject, pvals_corrected, _, _ = multipletests(p_values, method=adjustment_type)
                    else:
                        pvals_corrected = p_values

                    significance_comparisons = [
                        f"{comp} (p = {p_value:.4f}{'*' if p_value < threshhold else ''})"
                        for comp, p_value in zip(comparisons, pvals_corrected)
                        if p_value < 0.1
                    ]

                    significance = "Значимые различия перечислены ниже \n" + "\n".join(significance_comparisons) if significance_comparisons else "Нет статистически значимых различий (p < 0.05)"
                else:
                    significance = "Не удалось вычислить z-тест (недостаточно данных)"

                notes = f'''
                Вывод о стат. значимости: \n {significance}
                Примечание к таблице: 'Значения нормированы по столбцам и представлены в %.'
                Расшифровка вопросов:
                {col1} - {meta.column_names_to_labels[col1]}
                {col2} - {meta.column_names_to_labels[col2]}
                '''

                return {'table': table,
                        'notes': notes}

            question2 = st.selectbox("Выберите уторой столбец для вывода таблицы сопряженности", list_of_questions)
            col2 = meta_inside_out[question2]

            result = create_crosstab(col, col2)
            display(result['table'])
            print(result['notes'])

    except Exception as e:
        st.error(f"Ошибка: {str(e)}")
