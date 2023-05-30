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



<br><br><br>
SUBMIT 3. (submit_vova_04.csv)

Модель включает два основных этапа: подготовка фичей и непосредственно процесс обучения модели на них и получение предиктов

1. Подготовка фичей (Скрипт: ft02_get_features.ipynb)

Входные данные: датасеты, предоставленные компанией Озон:<br>
* train_data.parquet<br>
* test_data.parquet<br>
* train_pairs.parquet<br>
* test_pairs_wo_target.parquet<br>

Выходные данные: файлы со спарс-матрицами фичей и вспомогательные датафреймы:<br>
  train_features_02.npz<br>
  test_features_02.npz<br>
  train_other_02.parquet<br>
  test_other_02.parquet<br>

При обработке столбца с атрибутами товаров из обучающей выборки всего было получено 1447 уникальных атрибута. Значения атрибутов представляют собой списки строковых данных, в подавляющем большинстве состоящие из одного элемента-атрибута, реже - из нескольких элементов в списке.

Рассматриваемая модель использует в качестве фичей совпадения, несовпадения и частичные совпадения атрибутов, которые обозначаются определенными числовыми значениями, например, 1, 0.1, 0.5. Частичное совпадение имеет место быть, когда хотя бы у одного из товаров список значений для конкретного рассматриваемого атрибута состоит из 2 или более элементов и они пересекаются со значениями(ем) в списке атрибута второго товара в паре. И наконец, в случаях, когда конкретного атрибута нет у сравниваемых товаров, или у одного товара есть, а у другого нет, в матрице фичей проставляются нули

2. Обучение и предикты модели (Скрипт: ft02_attr04.ipynb)

Входные данные: файлы со спарс-матрицами фичей и вспомогательные датафреймы:<br>
  train_features_02.npz<br>
  test_features_02.npz<br>
  train_other_02.parquet<br>
  test_other_02.parquet<br>

Выходные данные: датафрейм с предиктами модели:<br>
  submit_vova_04.csv

Первичный исследовательский анализ показал, что разные категории 3-го уровня довольно сильно различаются по множеству параметров - по количеству сэмплов в train и test, по распределению таргета в train, и конечно по атрибутам, характерным для данной категории.

Для того чтобы помочь модели разделять классы, для каждой фичи рассчитывается "коэффициент значимости", который "подсвечивает" для модели случаи, на которые нужно обратить внимание (например, в нулевом классе несовпадения атрибутов являются очень важным фактором, а совпадения атрибутов - наоборот, незначительным, на который не стоит обращать внимание (как впрочем в какой-то мере и совпадения атрибутов в первом классе)). Таким образом, для каждой пары товаров исходный вектор фичей (длиной 1447) умножается на factor - вектор коэффициентов для каждой фичи, подобранный в результате исследования и анализа влияния фичей на модель

Для того чтобы модель более прицельно могла классифицировать пары товаров, она обучалась отдельно на каждой категории 3го уровня. Далее предикты каждой категории объединялись в единый вектор предиктов. В качестве моделей для каждой категории была применена модель LightGBM
