import os
import re
from collections import defaultdict

import numpy as np
from sklearn.model_selection import KFold


class Splitter:
    """
    Класс для корректного разбиения данных по пациентам и типам яичников.
    Разбиение происходит на train и validation по стратегии KFold.
    """

    def __init__(self, data_path):
        self.data_path = [
            os.path.join(data_path, name)
            for name in os.listdir(data_path)
            if os.path.isdir(os.path.join(data_path, name))
        ]
        self.__all_unique_patients_with_files = []
        self.__all_unique_patients = []
        self.train_sets = []
        self.test_sets = []

    @staticmethod
    def __extract_patient(text):
        pattern = r"(patient_\d+)"
        result = re.search(pattern, text)
        if result:
            return result.group(1)
        return None

    @staticmethod
    def __get_unique_patients_in_group(path):
        unique_patients = defaultdict(list)
        for file in os.listdir(path):
            patient = Splitter.__extract_patient(file)
            if patient is None:
                raise ValueError(f"Невозможно извлечь пациента из файла {file}")
            unique_patients[patient].append(os.path.join(path, file))

        return unique_patients

    def __get_all_patients(self):
        for path in self.data_path:
            unique_patients = self.__get_unique_patients_in_group(path)
            self.__all_unique_patients_with_files.append(unique_patients)
            self.__all_unique_patients.append(list(unique_patients.keys()))

    def __get_split(self, n_splits):
        kfs = []
        for unique_patients in self.__all_unique_patients:
            kf = KFold(n_splits=n_splits)
            kf.get_n_splits(unique_patients)
            kfs.append(kf)

        total_split = zip(
            kfs[0].split(self.__all_unique_patients[0]),
            kfs[1].split(self.__all_unique_patients[1]),
            kfs[2].split(self.__all_unique_patients[2]),
            kfs[3].split(self.__all_unique_patients[3]),
        )

        return total_split

    def get_train_test_sets(self, n_splits: int = 5):
        """Разбивает данные на k фолдов, содержащих train и test.

        Args:
            n_splits (int, optional): Количество фолдов. k-1 идет в train и 1 на validation. По умолчанию, 5.

        Returns:
            List[List], List[List]: Списки, содержащие train и validation части для соответствующего разбиения.
                                    Длина каждого равна k.
        """
        self.__get_all_patients()
        split = self.__get_split(n_splits=n_splits)
        for indexes in split:
            train = []
            test = []
            for i, index_group in enumerate(indexes):
                train_patients = np.array(self.__all_unique_patients[i])[index_group[0]]
                test_patients = np.array(self.__all_unique_patients[i])[index_group[1]]

                train_subset = []
                for patient in train_patients:
                    train_subset.extend(
                        self.__all_unique_patients_with_files[i][patient]
                    )

                test_subset = []
                for patient in test_patients:
                    test_subset.extend(
                        self.__all_unique_patients_with_files[i][patient]
                    )

                train.extend(train_subset)
                test.extend(test_subset)
            self.train_sets.append(train)
            self.test_sets.append(test)

        return self.train_sets, self.test_sets
