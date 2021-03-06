# -*- coding:utf-8 -*-

__author__ = 'ysekky'


import os
import yaml
import pandas
import time


CONFIG = yaml.load(open(os.path.join(os.path.dirname(__file__), '../config.yml')))


class LearnerBase(object):
    """
    学習と予測のベースモデル
    scikit-learn形式のインタフェースを持った学習器を使うことを想定して設計する

    - インタフェースの定義
    - データの取得，保存
    """

    learner = None

    @staticmethod
    def load_training_data():
        """
        訓練データを得る
        :return: pandas.DataFrame
        """
        return pandas.read_csv(CONFIG["training_data"])


    @staticmethod
    def load_test_data():
        """
        テストデータを得る
        :return: pandas.DataFrame
        """
        return pandas.read_csv(CONFIG["test_data"])


    def learning(self, training_data, feature_names, **kwargs):
        """
        学習を行うメソッド

        :param training_data: pandas.DataFrame: 訓練データ
        :param feature_names: list[str]: 訓練データで用いるカラム名をリストで定義
        :return:
        """
        x = self.__create_feature(training_data, feature_names)
        y = self.__encode_training_class(training_data)
        self.__train(x, y, **kwargs)

    def __train(self, x, y, **kwargs):
        """
        学習器を生成し, learnerに代入する
        :param x: 訓練入力データ
        :param y: 訓練正解データ
        :param kwargs: その他パラメータ等
        """
        #ここに固有の処理を書く
        pass

    def __predict(self, x):
        """
        モデルの予測結果とorderを返す．
        :param x:
        :return: list, list
        """
        #ここに固有の処理を書く
        return [], []

    def predict(self, test_data, feature_names):
        """
        モデルを使って提出用データを作成する
        :param test_data:
        :param feature_names:
        :return:
        """
        #固有の処理を書く

    def submit(self, test_data, y, rank_order):
        test_data['Class'] = self.decode_training_class(y)
        test_data['RankOrder'] = pandas.Series(rank_order)
        self.create_submit_data(test_data)

    @staticmethod
    def create_submit_data(test_data):
        """
        サブミット用のデータを出力.
        ファイル名はconfigでの指定にタイムスタンプをつける
        :param test_data: pandas.DataFrame
        :return:
        """
        #1からはじまるRankOrderを付与する必要がある．
        output_filename = "{}.{}".format(CONFIG["output_data"], int(time.time()))
        test_data.to_csv(output_filename, columns=["EventId", "RankOrder", "Class"], index=False)



    @staticmethod
    def create_feature(input_data, feature_names):
        """
        DataFrameから用いるカラムを選択し，学習器に与える形にして返す.

        :param training_data: pandas.DataFrame: 入力データ
        :param feature_names: list[str]: 訓練データで用いるカラム名をリストで定義
        :return: list[list]
        """
        return input_data[feature_names]

    @staticmethod
    def encode_training_class(training_data):
        """
        labelを0,1の形のリストにして返す．
        sを0, bを1にする

        :param training_data: pandas.DataFrame: 訓練データ
        :return: list[int]
        """
        return training_data["Label"].replace(['s', 'b'], [0, 1])

    @staticmethod
    def decode_training_class(predict_result):
        return pandas.Series(predict_result).replace([0, 1], ['s', 'b'])

