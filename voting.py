class voting_model():
    import pandas as pd
    import tensorflow as tf
    import numpy as np

    def __init__(self, models, vecorizers, cols, cats, metric = 'auc'):
        # models словарь  из моделей, где ключь - название категории
        # vecorizers словарь обученных векторайзеров для tfidf моделей
        # cols список списков с названиями столбцов для каждого типа моделей
        # cats общий список категорий

        self.optimizer = self.tf.keras.optimizers.Adam(learning_rate=0.01)
        self.loss = self.tf.keras.metrics.BinaryCrossentropy()
        self.acc = self.tf.keras.metrics.AUC()

        self.cat_dict = {}
        self.cols_dict = {}
        self.models_dict = {}
        self.best_cat_model = {}
        self.vecorizers = vecorizers
        self.metric = metric
        self.model_types = {}

        self.num_models = len(models)
        for i in range(self.num_models):
            self.models_dict[i] = models[i]
            self.cols_dict[i] = cols[i]
            self.model_types[i] = self.models_dict[i]['type']
        self.categories = cats
        
    def features_to_numpy(self, val, cols):
        arrays = []
        for col in cols:
            arrays.append(self.np.array(val[col].tolist()))
        return arrays
    
    def sep_mapping(self, col_1, col_2, col_3, col_4, col_5, col_6, col_7, col_8 = None):
        if col_8 != None:
            ds_mapped = {'img_vec_1':col_1, 'img_vec_2': col_2, 'bert_1': col_3, 'bert_2': col_4, 'sim': col_5, 'opp': col_6, 'sngl': col_7,}, col_8
        else:
            ds_mapped = {'img_vec_1':col_1, 'img_vec_2': col_2, 'bert_1': col_3, 'bert_2': col_4, 'sim': col_5, 'opp': col_6, 'sngl': col_7,}
        return ds_mapped

    def sep_dataset(self, X, cols, y = None):
        nn_x = self.features_to_numpy(X, cols)
        if type(y) == self.pd.Series:
            train_labels = y.values
            nn_dataset = self.tf.data.Dataset.from_tensor_slices((nn_x[0],nn_x[1],nn_x[2],nn_x[3],nn_x[4],nn_x[5],nn_x[6], train_labels))
        else:
            nn_dataset = self.tf.data.Dataset.from_tensor_slices((nn_x[0],nn_x[1],nn_x[2],nn_x[3],nn_x[4],nn_x[5],nn_x[6]))
        nn_dataset = nn_dataset.map(self.sep_mapping).cache()
        nn_dataset = nn_dataset.batch(16, drop_remainder = False).prefetch(self.tf.data.AUTOTUNE)
        return nn_dataset

    def tfidf_ds(self, df, tfidf_1, tfidf_2, vectorizer):
        tfidf_1_vectorized = vectorizer.transform(df[tfidf_1])
        tfidf_2_vectorized = vectorizer.transform(df[tfidf_2])
        return tfidf_1_vectorized + tfidf_2_vectorized
    
    def indexes(self, df, cat):
        idx = self.np.sort(self.np.where(df['categories'] == cat)[0])
        return idx.astype(int)
    
    def predict(self, X_val, method = 'score'):
        pred_values = self.np.zeros(X_val.shape[0]) # предикт для всего датасета
        for cat in self.categories:  # цикл по категориям
            print('___{0}'.format(cat))
            if method == 'soft':
                idx_te = self.indexes(X_val, cat)
                for i in range(len(self.models_dict)):  # цикл по моделям для категории
                    #---логрег tfidf
                    if self.models_dict[i]['type'] == 'tfidf':
                        # print('tfidf__')
                        pred_values_cat = self.models_dict[i][cat].predict_proba(self.tfidf_ds(X_val.iloc[idx_te], self.cols_dict[i][0], self.cols_dict[i][1], self.vecorizers[cat]))
                        tfidf_class_0 = pred_values_cat[:,0]
                        tfidf_class_1 = pred_values_cat[:,1]
                    #---нейросеть
                    if self.models_dict[i]['type'] == 'keras':
                        # print('keras__')
                        valid_dataset = self.sep_dataset(X_val.iloc[idx_te], self.cols_dict[i]) #valid dataset
                        pred_values_cat = self.models_dict[i][cat].predict(valid_dataset, verbose=0)
                        pred_values_cat = pred_values_cat.reshape(pred_values_cat.shape[0])

                        keras_class_0 = self.np.ones(len(pred_values_cat)) - pred_values_cat
                        keras_class_1 = pred_values_cat
                    #---логрег атрибуты
                    if self.models_dict[i]['type'] == 'attr':
                        # print('attr__')
                        pred_values_cat = self.models_dict[i][cat].predict_proba(X_val.iloc[idx_te][self.cols_dict[i]].to_numpy())
                        attr_class_0 = pred_values_cat[:,0]
                        attr_class_1 = pred_values_cat[:,1]
                #----        
                # lowest_1 = self.np.amin([tfidf_class_1, keras_class_1, attr_class_1], axis = 0) # минимум из трех вероятностей для первого класса
                # mean_0 = self.np.mean([tfidf_class_0,keras_class_0,attr_class_0], axis = 0)
                mean_1 = self.np.mean([tfidf_class_1,keras_class_1,attr_class_1], axis = 0)
                # mean_0_idxs = self.np.where(self.np.argmax([mean_0, mean_1], axis = 0) < 1) # возвращает индексы, где максимум за нулевым классом
                # soft = self.np.amax([mean_0, mean_1], axis = 0) # возвращает наибольшее значение из двух векторов
                # soft[mean_0_idxs] = lowest_1[mean_0_idxs]
                pred_values_cat = mean_1   # результат - средние вероятности 1 класса
                # pred_values_cat = soft   # результат - средние вероятности 1 класса, но там где выигрывает 0 класс, вероятность первого класса минимальная из трех

            pred_values[idx_te] = pred_values_cat
        return pred_values