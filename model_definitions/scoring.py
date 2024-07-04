from sklearn import metrics
from teradataml import (
    get_context,
    DataFrame,
    PMMLPredict,
    configure
)
from aoa import (
    record_evaluation_stats,
    aoa_create_context,
    store_byom_tmp,
    ModelContext
)

import os
import json
import glob
    
configure.byom_install_location = "mldb"


def score(context: ModelContext, **kwargs):
    aoa_create_context()

    print( context.dataset_info)
    # this evaluation.py can hanlde pmml.
    
    with open(f"{context.artifact_input_path}/model.{model_type}", "rb") as f:
        model_bytes = f.read()
        
    model_id = context.model_version
    
    print("Storing pmml into byom_models with model id:",context.model_version)
    save_byom(model_id = context.model_version, model_file = model_bytes, table_name = 'byom_models') 
    
    target_name = context.dataset_info.target_names[0]

    print("Retriving pmml with model id:",context.model_version)
    byom_target_sql = " CAST(CAST(json_report AS JSON).JSONExtractValue('$.cluster') AS INT)"
    model_tdf = retrieve_byom("iris_df_pmml", table_name = 'byom_models')
    
    print("Running PMML Predict)
    pmml = PMMLPredict(
            modeldata=model
            ,newdata=DataFrame.from_query(context.dataset_info.sql)
            accumulate=[context.dataset_info.entity_key, target_name])
        
    predictions_df = pmml.result
    print("prediction")
    print(predictions_df)

    # calculate stats if training stats exist
    if os.path.exists(f"{context.artifact_input_path}/data_stats.json"):
        record_evaluation_stats(features_df=DataFrame.from_query(context.dataset_info.sql),
                                predicted_df=DataFrame.from_query("SELECT * FROM predictions_tmp"),
                                context=context)
