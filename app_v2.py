import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyreadstat
import seaborn as sns
import streamlit as st
import tempfile
import re

from matplotlib.font_manager import FontProperties # для шрифтов
from matplotlib.colors import LinearSegmentedColormap # для цветов
from matplotlib.ticker import PercentFormatter
from math import log2
from os.path import commonprefix
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
        df = df.drop_duplicates()
        n_resp = len(df)
        st.success(f"Данные успешно загружены! Записей: {n_resp}")

        # Удаляем временный файл (опционально)
        import os
        os.unlink(tmp_path)

        # Автоматическое определение столбцов
        numeric_cols = df.select_dtypes(include=['number']).columns
        list_of_questions = []
        for key in numeric_cols:
          if meta.column_names_to_labels[key] in list_of_questions:
            continue
          else:
            list_of_questions.append(meta.column_names_to_labels[key])
        meta_inside_out = {}
        for question in list_of_questions:
          for key in meta.column_names_to_labels:
            if meta.column_names_to_labels[key] == question:
              meta_inside_out[question] = key
              break

        if len(list_of_questions) == 0:
            st.error("В файле нет количественных данных")
        else:
            question = st.selectbox("Выберите вопрос для вывода распределения", list_of_questions)
            col = meta_inside_out[question]

            def classify_question_optimized(col, keywords=None):
                if keywords is None:
                  keywords = ["шкал", "насколько", "оцен", "степень", "балл", "уровень"]
                question_text = meta.column_names_to_labels.get(col, "").lower()
                answer_labels = meta.variable_value_labels.get(col, {})
                answer_count = len(answer_labels)
                keyword_match = any(keyword in question_text for keyword in keywords)
                if keyword_match:
                  return "Шкальный"
                else:
                  return "Категориальный"

            unique_answers = df[col].nunique()
            question_type = classify_question_optimized(col)
            if col not in meta.variable_value_labels and unique_answers > 15\
            and question_type != "Шкальный":
              vis_type = 'Гистограмма'
            elif unique_answers == 2:
              vis_type = "Круговая диаграмма";
            elif question_type == "Шкальный":
              vis_type = "Столбчатая диаграмма"
            elif question_type != "Шкальный":
              vis_type = "Столбчатая диаграмма с сортировкой"
            vis_list = ["Гистограмма", "Столбчатая диаграмма", "Круговая диаграмма", "Столбчатая диаграмма с сортировкой", "Диаграмма с группировкой"]
            vis_list.remove(vis_type)
            vis_list.insert(0, vis_type)
            vis_type = st.selectbox("Тип визуализации", vis_list)
            if vis_type == "Диаграмма с группировкой":
              if col not in meta.variable_value_labels:
                st.error("Для выбранного вопроса не построить диаграмму с группировкой")
                st.stop()
              else:
                list_of_table_columns = []
                for q in list_of_questions:
                  col_try = meta_inside_out[q]
                  if col_try in meta.variable_value_labels and \
                  meta.variable_value_labels[col] == meta.variable_value_labels[col_try]:
                    list_of_table_columns.append(q)
                if len(list_of_table_columns) < 2:
                  st.error("Для выбранного вопроса не построить диаграмму с группировкой")
                  st.stop()
              table_columns_text = st.multiselect(
                  'Выберите дополнительные вопросы для диаграммы с группировкой',
                  list_of_table_columns,
                  default=[question]
                  )
              table_columns = []
              for question in table_columns_text:
                add_col = meta_inside_out[question]
                table_columns.append(add_col)

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

            # обновление метаданных
            clean_question_labels(meta)
            clean_value_labels(meta)

            def format_title(title, max_length=70):
                """
                Форматирует заголовок с переносами строк:
                1. Сначала делит по предложениям (точка/вопрос)
                2. Если предложения длинные - делит по словам
                """
                def split_sentence(sentence, max_len):
                    """Делит слишком длинные предложения по словам"""
                    words = sentence.split()
                    lines = []
                    current_line = []
                    for word in words:
                        if len(' '.join(current_line + [word])) <= max_len:
                            current_line.append(word)
                        else:
                            if current_line:
                                lines.append(' '.join(current_line))
                            current_line = [word]
                    if current_line:
                        lines.append(' '.join(current_line))
                    return '\n'.join(lines)
                # Шаг 1: Делим на предложения
                sentences = re.split(r'(?<=[.!?])\s+', title)
                processed_lines = []
                for sentence in sentences:
                    if len(sentence) <= max_length:
                        processed_lines.append(sentence)
                    else:
                        # Шаг 2: Делим длинные предложения по словам
                        processed_lines.append(split_sentence(sentence, max_length))
                # Собираем результат
                result = '\n'.join(processed_lines)
                # Проверяем, не получились ли слишком длинные строки (на всякий случай)
                final_lines = []
                for line in result.split('\n'):
                    if len(line) > max_length:
                        final_lines.extend(split_sentence(line, max_length).split('\n'))
                    else:
                        final_lines.append(line)
                return '\n'.join(final_lines)

            def split_string_by_length(text, max_length=50, max_lines=None):
                """
                Форматирует заголовок с переносами строк:
                1. Сначала делит по предложениям (точка/вопрос)
                2. Если предложения длинные - делит по словам
                """
                # Обработка случая max_lines=0
                if max_lines is not None and max_lines == 0:
                    return '...'
                words = text.split(' ')
                lines = []
                current_line = []
                was_truncated = False  # Флаг обрезки текста
                for word in words:
                    if len(' '.join(current_line + [word])) <= max_length:
                        current_line.append(word)
                    else:
                        lines.append(' '.join(current_line))
                        current_line = [word]
                        if max_lines is not None and len(lines) >= max_lines:
                            was_truncated = True
                            break
                # Добавляем последнюю строку (если не было обрезки)
                if current_line and not was_truncated and (max_lines is None or len(lines) < max_lines):
                    lines.append(' '.join(current_line))
                # Добавляем многоточие при обрезке
                if max_lines is not None:
                    if was_truncated or len(lines) > max_lines:
                        lines = lines[:max_lines]
                        if lines:
                            if not lines[-1].endswith('...'):
                                last_line = lines[-1]
                                max_line_length = max_length - 3
                                if len(last_line) > max_line_length:
                                    last_line = last_line[:max_line_length].rstrip()
                                lines[-1] = last_line + '...'
                                was_truncated = True
                    elif len(lines) == max_lines and len(' '.join(words)) > len(' '.join(lines)):
                        if not lines[-1].endswith('...'):
                            lines[-1] = lines[-1] + '...'
                return '\n'.join(lines) if lines else '...'

            def get_max_line_length(formatted_text):
                """
                Возвращает максимальное количество символов в одной строке текста.
                Параметры:
                    formatted_text (str): Текст с переносами строк (результат split_string_by_length)
                Возвращает:
                    int: Максимальная длина строки (в символах)
                """
                if not formatted_text:
                    return 0
                lines = formatted_text.split('\n')
                max_length = max(len(line) for line in lines)
                return max_length

            def get_hist(col):
              fig, ax = plt.subplots(figsize=(9, 5))
              question = meta.column_names_to_labels[col]
              question = format_title(question)
              ax = sns.histplot(df[col], kde=False, stat="count", edgecolor="white", alpha=1, color='#E62083')
              plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y/n_resp*100:.1f}'))
              plt.title(f'{question}', fontsize=14, pad=10)
              plt.ylabel('Количество ответов (в %)', fontsize=10)
              plt.xticks(fontsize=10)
              plt.yticks(fontsize=10)
              plt.xlabel('')
              plt.grid(axis='y', alpha=0.3)
              plt.gca().yaxis.set_tick_params(length=0)
              ax.spines['top'].set_visible(False)
              ax.spines['right'].set_visible(False)
              #ax.spines['bottom'].set_visible(False)
              ax.spines['left'].set_visible(False)
              return fig

            def get_piechart(col):
              data = process_multi_response_1(col)
              plot_df=round(data.value_counts().div(df[col].count()/100))
              if not is_multi_response(col):
                teck_ans = 100 - plot_df.iloc[:-1].sum()
                if teck_ans >= 0:
                  plot_df.iloc[-1] = teck_ans
              plot_df = pd.DataFrame(plot_df)
              plot_df.reset_index(names=['Ответ'], inplace=True)
              plot_df['ans'] = plot_df['Ответ']
              plot_df['ans'] = plot_df['ans'].apply(lambda x: 0 if x < 90 else x)
              if classify_question_optimized(col) != "Шкальный":
                plot_df.sort_values(by=['ans', 'count'], ascending = [True, False], inplace=True)
              plot_df['color'] = plot_df['ans'].apply(lambda x: '#E62083' if x == 0 else '#bfbfbf') #E62083 #12AFFF #bfbfbf
              if col in meta.variable_value_labels:
                plot_df['Ответ'] = plot_df['Ответ'].map(lambda x: meta.variable_value_labels[col][x])
                max_lines = 20//len(meta.variable_value_labels[col])
                if max_lines == 0:
                  max_lines = 1
                plot_df['Ответ'] = plot_df['Ответ'].apply(lambda x: split_string_by_length(x, max_lines=max_lines))
              else:
                plot_df['Ответ'] = plot_df['Ответ'].astype('str')
              plot_df['lenth'] = plot_df['Ответ'].apply(lambda x: get_max_line_length(x))
              max_length_ticks = plot_df['lenth'].max()
              # Создаем цветовой градиент
              colors = ['#E62083', '#12AFFF']
              n_sectors = len(plot_df)
              cmap = LinearSegmentedColormap.from_list('custom_gradient', colors, N=n_sectors)
              sector_colors = [cmap(i) for i in np.linspace(0, 1, n_sectors)]
              # Строим диаграмму
              fig, ax = plt.subplots(figsize=(5, 5))
              wedges, texts, autotexts = ax.pie(
                  x=plot_df['count'],
                  colors=sector_colors,
                  autopct = lambda p: f'{p:.0f}' if p >= 5 else '',
                  wedgeprops={'linewidth': 1, 'edgecolor': 'white'},
                  startangle=90,
                  counterclock=False
              )
              #wedges[0].set_linewidth(5)
              for autotext in autotexts:
                autotext.set_size(12)            # Размер шрифта
                autotext.set_weight('bold')      # Жирный
                autotext.set_color('white')      # Белый цвет
                # autotext.set_bbox({'facecolor': 'black', 'alpha': 0.5})  # Фон (опционально)
              # Добавляем легенду с подписями из столбца 'Ответ'
              ax.legend(
                  wedges,                      # Сектора диаграммы
                  plot_df['Ответ'],            # Подписи для легенды
                  #title="Варианты ответов",    # Заголовок легенды
                  loc="center left",           # Позиция легенды
                  bbox_to_anchor=(1, 0.5),     # Выносим легенду за пределы диаграммы
                  fontsize=10,
                  frameon=False
              )
              # Настройка заголовка
              question = meta.column_names_to_labels[col]
              question = format_title(question)
              x = (max_length_ticks+50)/100
              ax.set_title(f'{question} (в %)', x=x, fontsize=14, pad=10)
              return fig

            def get_barplot(col, is_sorted=True):
              data = process_multi_response_1(col)
              plot_df=round(data.value_counts().div(df[col].count()/100))
              if not is_multi_response(col):
                teck_ans = 100 - plot_df.iloc[:-1].sum()
                if teck_ans >= 0:
                  plot_df.iloc[-1] = teck_ans
              plot_df = pd.DataFrame(plot_df)
              plot_df.reset_index(names=['Ответ'], inplace=True)
              plot_df['ans'] = plot_df['Ответ']
              plot_df['ans'] = plot_df['ans'].apply(lambda x: 0 if x < 90 else x)
              if is_sorted:
                plot_df.sort_values(by=['ans', 'count'], ascending = [True, False], inplace=True)
              else:
                plot_df.sort_values(by=['Ответ'], ascending = [True], inplace=True)
              plot_df['color'] = plot_df['ans'].apply(lambda x: '#E62083' if x == 0 else '#bfbfbf') #E62083 #12AFFF #bfbfbf
              if col in meta.variable_value_labels:
                plot_df['Ответ'] = plot_df['Ответ'].map(lambda x: meta.variable_value_labels[col][x])
                max_lines = 20//len(meta.variable_value_labels[col])
                if max_lines == 0:
                  max_lines = 1
                plot_df['Ответ'] = plot_df['Ответ'].apply(lambda x: split_string_by_length(x, max_lines=max_lines))
              else:
                plot_df['Ответ'] = plot_df['Ответ'].astype('str')
              plot_df['lenth'] = plot_df['Ответ'].apply(lambda x: get_max_line_length(x))
              max_length_ticks = plot_df['lenth'].max()
              heigh = 4.5
              if len(plot_df) > 20:
                heigh = len(plot_df) * 0.225
              fig, ax = plt.subplots(figsize=(5, heigh))
              sns.barplot(x=plot_df['count'],
                              y=plot_df.index.astype(str),
                              hue = plot_df.index.astype(str),
                              palette=plot_df['color'].tolist(),
                              legend=False)
              ax.set_yticks(range(len(plot_df)))
              ax.set_yticklabels(plot_df['Ответ'])
              ax.spines['top'].set_visible(False)
              ax.spines['right'].set_visible(False)
              ax.spines['bottom'].set_visible(False)
              for container in ax.containers:
                  ax.bar_label(container,
                              label_type='edge',
                              padding=5,
                              fontsize=10,
                              fmt='%.0f',
                              fontweight='bold',
                              fontfamily='sans-serif')
              plt.xticks([])
              plt.yticks(fontsize=10)
              question = meta.column_names_to_labels[col] # текст вопроса
              question = format_title(question)
              x = (50-max_length_ticks)/100
              plt.title(f'{question} (в %)', x=x, fontsize=14, pad=10)
              plt.xlabel('', fontsize=1)
              plt.ylabel('', fontsize=1)
              return fig

            def get_stacked(table_columns):
              questions = []
              for col in table_columns:
                questions.append(meta.column_names_to_labels[col])
              question = commonprefix(questions) # текст вопроса
              question = format_title(question)
              question += ' ...'
              long_df = df.melt(
                  value_vars=table_columns,
                  var_name='Вопрос',
                  value_name='Ответ'
              )
              plot_df = (long_df
                        .groupby(['Вопрос', 'Ответ'])
                        .size()
                        .unstack(fill_value=0)  # Преобразуем в широкий формат
                        .apply(lambda x: 100 * x / x.sum(), axis=1)  # Нормировка по строкам
                        .stack()
                        .reset_index(name='Доля (%)'))
              plot_df['Вопрос'] = plot_df['Вопрос'].map(lambda x: meta.column_names_to_labels[x])
              plot_df['Вопрос'] = plot_df['Вопрос'].map(lambda x: x[len(question)-4:])
              if table_columns[0] in meta.variable_value_labels:
                plot_df['Ответ'] = plot_df['Ответ'].map(lambda x: meta.variable_value_labels[table_columns[0]][x])
              plot_df['lenth'] = plot_df['Вопрос'].apply(lambda x: get_max_line_length(x))
              max_length_ticks = plot_df['lenth'].max()
              heigh = 4.5
              if len(plot_df) > 20:
                heigh = len(plot_df) * 0.225
              colors = ['#E62083', '#12AFFF']
              n_sectors = len(plot_df['Ответ'].unique())
              cmap = LinearSegmentedColormap.from_list('custom_gradient', colors, N=n_sectors)
              sector_colors = [cmap(i) for i in np.linspace(0, 1, n_sectors)]
              fig, ax = plt.subplots(figsize=(5, heigh))
              sns.barplot(x=plot_df['Доля (%)'],
                                y=plot_df['Вопрос'],
                                hue = plot_df['Ответ'],
                                palette=sector_colors,
                                legend=True)
              ax.set_yticks(range(len(table_columns)))
              #x.set_yticklabels(plot_df['Ответ'])
              ax.spines['top'].set_visible(False)
              ax.spines['right'].set_visible(False)
              ax.spines['bottom'].set_visible(False)
              for container in ax.containers:
                  ax.bar_label(container,
                              label_type='edge',
                              padding=5,
                              fontsize=10,
                              fmt='%.0f',
                              fontweight='bold',
                              fontfamily='sans-serif')
              x = (50-max_length_ticks)/100
              plt.legend(
                  title=False,
                  bbox_to_anchor=(x, 0),
                  loc='upper center',
                  ncol=len(plot_df['Ответ'].unique()),
                  frameon=False
              )
              plt.xticks([])
              plt.yticks(fontsize=10)
              plt.title(f'{question} (в %)', x=x, fontsize=14, pad=10)
              plt.xlabel('', fontsize=1)
              plt.ylabel('', fontsize=1)
              return fig

            # Диаграмма
            if vis_type == "Гистограмма":
              fig = get_hist(col)
            elif vis_type == "Круговая диаграмма":
              fig = get_piechart(col)
            elif vis_type == "Столбчатая диаграмма":
              fig = get_barplot(col, is_sorted=False)
            elif vis_type == "Столбчатая диаграмма с сортировкой":
              fig = get_barplot(col, is_sorted=True)
            elif vis_type == "Диаграмма с группировкой":
              fig = get_stacked(table_columns)
            st.pyplot(fig)

            # Таблица сопряженности
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

            def smart_format(x, precision=4):
                return f"{x:.{precision}f}" if abs(x) >= 0.0001 else f"{x:.2e}"

            def create_crosstab(col1, col2, adjustment_type = 'holm', chi2_threshhold = 0.05, z_threshhold = 0.05, min_n_obs = 10):

                """
                Создает таблицу сопряженности между двумя переменными и выполняет двухэтапный статистический анализ:
                1. Оценивает статистическую связь между признаками с помощью теста хи-квадрат.
                2. Если связь значима, осуществляет проверку значимости различий для каждой группы против остальных с помощью z-теста.
                Подсвечивает ячейки, статистически отличающиеся от остальных.

                Args:
                    col1 (pd.Series): Первый столбец.
                    col2 (pd.Series): Второй столбец.
                    adjustment_type (str or None): Метод поправки на множественные сравнения ('holm', 'bonferroni', 'fdr_bh' и др.). Если None, поправка не применяется. По умолчанию 'holm'.
                    chi2_threshhold (float): Порог уровня значимости для теста хи-квадрат. По умолчанию 0.1.
                    z_threshhold (float): Порог уровня значимости для z-теста пропорций. По умолчанию 0.05.
                    min_n_obs (int): Минимальное количество наблюдений в группе для z-теста.

                Returns:
                    dict: Словарь с двумя ключами:
                        - 'table': стилизованная таблица сопряженности (pandas Styler object).
                        - 'notes': текстовый вывод с результатами проверки статистических гипотез
                                и дополнительными комментариями.
                """

                # Шаг 1: Предобработка данных и создание таблицы сопряжённости

                # Обрабатываем множественные ответы
                data = process_multi_response(col1, col2)

                # Преобразовываем непрерывные значения в интервальные
                transformed_col1 = convert_to_intervals(col1, data)
                transformed_col2 = convert_to_intervals(col2, data)

                # Формируем таблицу сопряженности для проведения статистических тестов
                contingency_table = pd.crosstab(transformed_col1, transformed_col2)
                contingency_table.index = apply_value_labels(contingency_table.index)
                contingency_table.columns = apply_value_labels(contingency_table.columns)

                # Шаг 2: Проведение теста хи-квадрат на наличие связи между переменными
                _, chi2_pvalue, _, expected = chi2_contingency(contingency_table)

                # !!! Здесь должна быть проверка, что ожидаемые значение >= 5 более чем в 80 % ячеек (не включая технические вопросы)

                if chi2_pvalue >= chi2_threshhold:
                    chi2_notes = f'''
                Результаты применения теста Хи-квадрат свидетельствуют о том, что связь между переменными не является статистически значимой (p = {smart_format(chi2_pvalue)} ≥ {chi2_threshhold}).
                В следствие этого групповые сравнения не проводились. \n'''
                    chi_valid = False
                else:
                    chi2_notes = f'''
                Результаты применения теста Хи-квадрат свидетельствуют о том, что связь между переменными является статистически значимой (p = {smart_format(chi2_pvalue)}). \n'''
                    chi_valid = True

                # Шаг 3: Проведение z-теста для проверки значимости различий каждой группы против всех остальных

                significant_groups = {}
                detailed_results = []

                for answer in contingency_table.index:
                    z_pvalues = []
                    comparisons = []
                    group_labels = list(contingency_table.columns)

                    for group in group_labels:
                        group_success = contingency_table.loc[answer, group]
                        rest_success = contingency_table.loc[answer, :].sum() - group_success

                        group_nobs = contingency_table.loc[:, group].sum()
                        rest_nobs = contingency_table.sum().sum() - group_nobs

                        # Проверка достаточности наблюдений
                        if min([group_success, group_nobs - group_success, rest_success, rest_nobs - rest_success]) <= min_n_obs:
                            continue

                        success = [group_success, rest_success]
                        nobs = [group_nobs, rest_nobs]

                        zstat, pvalue = proportions_ztest(success, nobs)
                        z_pvalues.append(pvalue)
                        comparisons.append((group, answer, pvalue, zstat))

                    # Поправка на множественные сравнения внутри одного ответа
                    if z_pvalues:
                        if adjustment_type:
                            _, corr_pvalues, _, _ = multipletests(z_pvalues, method = adjustment_type)
                        else:
                            corr_pvalues = z_pvalues

                        for (group, answer, _, zstat), corr_pvalue in zip(comparisons, corr_pvalues):
                            if corr_pvalue < z_threshhold:
                                significant_groups[(group, answer)] = zstat
                                detailed_results.append(f'''
                {group} vs остальные в ответе '{answer}': p = {smart_format(corr_pvalue)}''')

                if chi_valid  & (len(detailed_results) > 0):
                    z_notes = f'''
                Согласно z-тесту пропорций, отдельные группы демонстрируют статистически значимое отличие по доле ответов на конкретные вопросы относительно всех остальных. В частности: {"".join(detailed_results)}'''
                elif chi_valid  & (len(detailed_results) == 0):
                    z_notes = f'''
                Согласно z-тесту пропорций, статистически значимых отличий на уровне отдельных групп нет.'''
                elif chi_valid == False:
                    z_notes = ''

                # Шаг 4: Создаем таблицу сопряженности для отображения в строке вывода
                crosstab_to_show = pd.crosstab(transformed_col1,
                                    transformed_col2,
                                    margins=True,
                                    margins_name='Всего',
                                    normalize = 'columns')

                crosstab_to_show.index = apply_value_labels(crosstab_to_show.index)
                crosstab_to_show.columns = apply_value_labels(crosstab_to_show.columns)

                def highlight_significant_groups(value, index, column):
                    if (column, index) in significant_groups:
                        return 'background-color: #7449D3' if significant_groups[(column, index)] > 0 else 'background-color: #F91E7F'
                    return ''

                # Применяем стили к таблице
                styled_table = crosstab_to_show.style.format(lambda x: round(x*100, 1))\
                                .apply(lambda df: pd.DataFrame([[highlight_significant_groups(df.iloc[i, j], df.index[i], df.columns[j])
                                                    for j in range(df.shape[1])]
                                                    for i in range(df.shape[0])],
                                                    index=df.index, columns=df.columns), axis=None)

                notes =  chi2_notes +  z_notes + \
                f'''
                 Значения нормированы по столбцам и представлены в %.
                 Цветом подсвечены ячейки, в которых наблюдаемые значения статистически отличаются от остальных групп (p < {z_threshhold}).
                 Если доля группы ниже всех остальных, ячейки выделены розовым цветом, если доля группы выше - фиолетовым.
                {f"Тесты учитывают поправку на множественные сравнения методом {adjustment_method}" if adjustment_type else 'Тесты не корректируются поправкой на множественные сравнения'}
                ''' + f'''
                {col} - {meta.column_names_to_labels[col]}
                {col2} - {meta.column_names_to_labels[col2]}
                '''

                return {'table': styled_table,
                        'notes': notes}

            question2 = st.selectbox("Выберите второй вопрос для вывода таблицы сопряженности", list_of_questions)
            col2 = meta_inside_out[question2]

            # Параметры тестов
            with st.expander("⚙️ Настройки статистического анализа", expanded=True):
                st.markdown("Задайте параметры проверки гипотез:")

                # Бегунок для выбора уровня значимости хи-квадрат
                alpha_chi2 = st.slider(
                    "Уровень значимости для теста Хи-квадрат, %",
                    min_value= 1,
                    max_value= 10,
                    value= 5,
                    step= 1
                )

                # Бегунок для выбора уровня значимости z-теста
                alpha_z = st.slider(
                    "Уровень значимости для z-теста пропорций, %",
                    min_value= 1,
                    max_value= 10,
                    value= 5,
                    step= 1
                )

                # Выбор поправки на множественные сравнения
                adjustment_method = st.selectbox(
                    "Метод поправки на множественные сравнения",
                    options=[
                        "Холма",
                        "Бонферрони",
                        "Беньямини-Хохберга",
                        "Без поправок"
                    ],
                    index=0
                )

            adjustment_dict = {"Холма": "holm",
                        "Бонферрони": "bonferroni",
                        "Беньямини-Хохберга": "fdr_bh",
                        "Без поправок": None}

            adjustment_type = adjustment_dict[adjustment_method]

            result = create_crosstab(col, col2, adjustment_type, alpha_chi2/100, alpha_z/100)

            st.subheader(f'🧾 Таблица сопряженности между {col} и {col2}')

            # Преобразуем Styler в HTML, чтобы избежать прокрутки
            st.markdown(result['table'].to_html(), unsafe_allow_html=True)

            st.markdown("---")
            st.subheader("📋 Интерпретация результатов")

            # Разбиваем текст на логические части по ключевым словам
            notes = result['notes'].strip().split("\n")
            note_blocks = {
            "Тест хи-квадрат": [],
            "Z-тест пропорций": [],
            "Примечание к таблице": [],
            "Расшифровка вопросов": []
            }

            current_block = None
            for line in notes:
                line = line.strip()
                if not line:
                    continue
                if "Хи-квадрат" in line:
                    current_block = "Тест хи-квадрат"
                elif "z-тест" in line:
                    current_block = "Z-тест пропорций"
                elif "Значения нормированы" in line:
                    current_block = "Примечание к таблице"
                elif col in line:
                    current_block = "Расшифровка вопросов"
                if current_block:
                    note_blocks[current_block].append(line)

            # Отображаем каждый блок
            for title, lines in note_blocks.items():
                if lines:
                    with st.expander(f"🔹 {title}", expanded=True):
                        for line in lines:
                            st.markdown(f"- {line}")

    except Exception as e:
        st.error(f"Ошибка: {str(e)}")
