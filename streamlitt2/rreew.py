import streamlit as st
import numpy as np
import pandas as pd
import lightfm as lf
import nmslib
import pickle
import scipy.sparse as sparse

def nearest_products_nms(product_id, index, n=10):
    """Функция для поиска ближайших соседей, возвращает построенный индекс"""
    nn = index.knnQuery(item_embeddings[product_id], k=n)
    return nn

def get_names(index):
    """
    input - idx of product
    return - list of names
    """
    names = []
    for idx in index:
        names.append(name_mapper[idx])
    return names

def load_embeddings():
    """
    Функция для загрузки векторных представлений
    """
    with open('./item_embeddings.pickle', 'rb') as f:
        item_embeddings = pickle.load(f)

    # Тут мы используем nmslib, чтобы создать наш быстрый knn
    nms_idx = nmslib.init(method='hnsw', space='cosinesimil')
    nms_idx.addDataPointBatch(item_embeddings)
    nms_idx.createIndex(print_progress=True)
    return item_embeddings,nms_idx

def make_mappers():
    """
    Функция для создания отображения id в title
    """
    name_mapper = dict(zip(products.itemid, products.title))

    return name_mapper

def read_files(folder_name='data'):
    """
    Функция для чтения файлов + преобразование к  нижнему регистру
    """
    products = pd.read_csv('./items.csv')
    products['title'] = products.title.str.lower()
    return  products 

#Загружаем данные
products  = read_files(folder_name='data') 
item_embeddings,nms_idx = load_embeddings()
name_mapper = make_mappers()

#Форма для ввода текста
title = st.text_input('item for search', '')
title = title.lower()

#Наш поиск
output = products[products.title.str.contains(title) > 0]

#Выбор из списка
option = st.selectbox('select', output['title'].values)

#Выводим 
'You selected: ', option

#Ищем рекомендации
val_index = output[output['title'].values == option].itemid
index = nearest_products_nms(val_index, nms_idx, 5)

#Выводим рекомендации 
'recommend for you: '
st.write('', get_names(index[0])[1:])
