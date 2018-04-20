import pandas as pd
import sklearn

class ModelWrapper:

    def __init__(self, model, single):
        self.model_template = model
        self.model = None
        self.single = single

    def fit(self, train_dataframe):
        if self.single:
            self.model = self._fit_single_model(train_dataframe, self.model_template)
        else:
            self.model = self._fit_multi_model(train_dataframe, self.model_template)

    def predict(self, test_dataframe):
        if self.single:
            return self._predict_single_model(test_dataframe, self.model)
        else:
            return self._predict_multi_model(test_dataframe, self.model)

    @staticmethod
    def _frame_to_X_and_y(frame, remove_algorithm=False):
        y = frame['objective_function']
        del frame['objective_function']
        del frame['instance_id']
        if remove_algorithm:
            del frame['algorithm']

        return frame.as_matrix(), y.as_matrix()

    @staticmethod
    def _fit_multi_model(train_dataframe, pipeline):
        algorithms = train_dataframe.algorithm.unique()
        models = dict()

        expected_size = int(len(train_dataframe) / len(algorithms))

        for algorithm_id in algorithms:
            train_algorithm = train_dataframe.loc[train_dataframe['algorithm'] == algorithm_id]
            if len(train_algorithm) != expected_size:
                raise ValueError(
                    'Train frame wrong size. Excepted %d got %d' % (expected_size, len(train_algorithm)))
            train_X, train_y = ModelWrapper._frame_to_X_and_y(train_algorithm, True)
            pipeline_algorithm = sklearn.base.clone(pipeline)
            pipeline_algorithm.fit(train_X, train_y)
            models[algorithm_id] = pipeline_algorithm
        return models

    @staticmethod
    def _predict_multi_model(test_dataframe, models):
        algorithms = test_dataframe.algorithm.unique()
        test_task_ids = test_dataframe.instance_id.unique()

        task_algorithm_pred = {task: dict() for task in test_task_ids}

        for task_id in test_task_ids:
            test_frame = test_dataframe[test_dataframe['instance_id'] == task_id]

            for algorithm_id in algorithms:
                test_algorithm = test_frame.loc[test_frame['algorithm'] == algorithm_id]
                if len(test_algorithm) != 1:
                    raise ValueError()
                test_X, _ = ModelWrapper._frame_to_X_and_y(test_algorithm, True)
                y_hat = models[algorithm_id].predict(test_X)

                task_algorithm_pred[task_id][algorithm_id] = y_hat[0]
        return task_algorithm_pred

    @staticmethod
    def _fit_single_model(train_dataframe, pipeline):
        model = sklearn.base.clone(pipeline)
        train_X, train_y = ModelWrapper._frame_to_X_and_y(pd.get_dummies(train_dataframe))
        model.fit(train_X, train_y)
        return model

    @staticmethod
    def _predict_single_model(test_dataframe, model):
        algorithms = test_dataframe.algorithm.unique()
        test_task_ids = test_dataframe.instance_id.unique()

        task_algorithm_pred = {task: dict() for task in test_task_ids}

        for task_id in test_task_ids:
            test_frame = pd.get_dummies(test_dataframe[test_dataframe['instance_id'] == task_id])

            for algorithm_id in algorithms:
                test_algorithm = test_frame.loc[test_frame['algorithm_' + str(algorithm_id)] == 1]
                if len(test_algorithm) != 1:
                    raise ValueError()
                test_X, _ = ModelWrapper._frame_to_X_and_y(test_algorithm)
                y_hat = model.predict(test_X)

                task_algorithm_pred[task_id][algorithm_id] = y_hat[0]
        return task_algorithm_pred
