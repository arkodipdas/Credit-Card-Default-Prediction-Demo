from src.CreditcardDefaultPrediction.pipelines.prediction_pipeline import CustomDataset



custdataobj = CustomDataset(20000,2,2,1,24,2,2,-1,-1,-2,-2,3913,3102,689,0,0,0,0,689,0,0,0,0)

data = custdataobj.get_data_as_dataframe()

print(data)