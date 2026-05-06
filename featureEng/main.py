# This is a sample Python script.
import yaml
from src.train.utils import set_seeds

from src.train.hyper_trainer import run_feat_UCI, run_feat_save


# data has over ten features, index
# [4, 5, 11, 12, 15, 16, 17, 19, 20, 21, 27, 31, 32, 33, 34, 35, 36, 38, 42, 43, 44, 46, 50, 51, 52, 53, 54, 61, 63, 65, 66, 67, 68, 69, 70, 71, 72, 73, 77, 80, 81, 83]
def main(hei = 'NAME'):
    params_file = "D:\\XGBoost-by-ensemble-feature-engineering-main\\XGBoost-by-ensemble-feature-engineering-main\\featureEng\\src\\config\\params.UCIdata.yml"
    with open(params_file, 'r') as f:
        params = yaml.safe_load(f)
    params['method'] = 'bffeatxgbAv51'

    # Different seed for each of the 5 network embeddings
    params["seed"] = params["seed"] + params["class_label"]
    set_seeds(params["seed"])

    #run_train_UCI(params, params_file)
    run_feat_save(params, params_file)
    #run_feat_UCI(params, params_file)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main('PyCharm')