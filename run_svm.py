__author__ = 'ysekky'


from model.simple_svm import SimpleSVM

target_columns = [
    "DER_mass_MMC",
    "DER_mass_transverse_met_lep",
    "DER_mass_vis",
    'DER_deltaeta_jet_jet',
    'DER_mass_jet_jet',
    'DER_prodeta_jet_jet',
    'DER_sum_pt',
    'DER_pt_ratio_lep_tau',
    'DER_lep_eta_centrality',
    'DER_met_phi_centrality',
    'PRI_tau_pt',
    'DER_sum_pt',
    "PRI_jet_subleading_eta",
    'PRI_jet_leading_eta',
    'PRI_jet_leading_pt',
    'PRI_jet_num',
    'PRI_met_sumet',
    'PRI_met']


def missing_data_by_median(training_data, test_data):
    """

    :param training_data: pandas.DataFrame
    :param test_data: pandas.DataFrame
    :return:
    """
    training_data = training_data.replace([-999,], [None,])
    training_data = training_data.fillna(training_data.median())
    test_data = test_data.replace([-999,], [None,])
    test_data = test_data.fillna(training_data.median())
    return training_data, test_data

def run():
    svm = SimpleSVM()
    training_data, test_data = missing_data_by_median(svm.load_training_data(), svm.load_test_data())
    svm.learning(training_data, target_columns)
    svm.predict(test_data, target_columns)