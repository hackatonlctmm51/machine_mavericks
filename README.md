# machine_mavericks

Задача хакатона подразумевала под собой разработку ML-модели, способной определить идентичность товаров по названиям, атрибутам и изображениям. 

Проблема заключалась в том, что товары между собой могут различаться всего 1-2 атрибутами, иметь одинаковые изображения и практически полностью совпадающие тексты

Основные идеи заключались в:<>

1) Вычленении всего "мусора", то есть артикулов, партномеров, серийных номеров и тд. из названия. Дополнительно оставлены английские слова. Обработка проводилась с помощью регулярных выражений.

Далее товары были разделены на 46 категорий. 

Для каждой категории были извлечены TF-IDF фичи из оставшегося "огрызка" наименования.

Для каждой категории были извлечены TF-IDF фичи из атрибутов (конкатенированные и суммированные). Для атрибутов не проводилось доп. обработки

Векторы названий и атрибутов были сконкатенированы попарно для каждого товара соответсвующей категории
Для каждой пары были посчитаны косинусное и евклидовы расстояния

Расстояния были также посчитаны для изображений в одном из вариантов сабмита

Далее отрабатывал LightGBM, отдельно по суммированным TF-IDF фичам + расстояния, отдельно по конкатенированным TF-IDF фичам + расстояния

Для запуска модели достаточно запустить ноутбук submit_tf_idf_final.ipynb, предварительно проставив в загрузке датафреймов свои пути.

2) Получении фич от обработки атрибутов. 

-Если один и тот же атрибут совпадал у двух товаров, то в фиче проставлялось совпадение
-Если один и тот же атрибут не совпадал у двух товаров, то в фиче проставлялось несовпадение


3) Мультимодальная модель определяла схожесть товаров по полученным от организатора ембедингам изображений и текстов
4) Отдельно отрабатывалась сиамская модель получения фичей по полученным от организатора ембедингам изображений и текстов, но не нашла применения

Три модели выше объединялись с помощью мягкого голосования
