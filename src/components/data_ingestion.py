import os,sys
import pandas as pd
from src.logging import logging
from src.exception import CustomException
from sklearn.model_selection import train_test_split
from src.components.data_transformation import DataTransformation,DataTransformationConfig
from pymongo import MongoClient
from dataclasses import dataclass


@dataclass
class DataIngestionConfig:
    raw_path:str=os.path.join('artifact','raw.csv')
    train_path:str=os.path.join('artifact','train.csv')
    test_path:str=os.path.join('artifact','test.csv')

class dataingestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()
    
    def initiate_data_ingestion(self):
        try:
            logging.info('data ingestion from db started')
        
            client = MongoClient("mongodb+srv://ayan:ayan@cluster0.njinath.mongodb.net/?retryWrites=true&w=majority")
            db = client.credit 
            collection=db.credit_card
            cursor=collection.find({})
            df=list(cursor)
            df=pd.DataFrame(df)
            data=df.drop(columns=['_id'],axis=1)
            data.columns=data.iloc[0]
            data2=data.drop(df.index[0])
            logging.info('data ingestion has ended')

            os.makedirs(os.path.dirname(self.ingestion_config.raw_path),exist_ok=True)
            data2.to_csv(self.ingestion_config.raw_path,index=False)

            logging.info('initializing train test split to split training and testing data')
            train_set,test_set=train_test_split(data2,test_size=0.20,random_state=42)
            train_set.to_csv(self.ingestion_config.train_path,index=False)
            test_set.to_csv(self.ingestion_config.test_path,index=False)

            logging.info('train and test data had been uploaded to artifacts')

            return(self.ingestion_config.train_path,self.ingestion_config.test_path)

        except Exception as e:
            raise CustomException(e,sys)
        


if __name__=="__main__":
    obj=dataingestion()
    train_data,test_data=obj.initiate_data_ingestion()

    obj_trans=DataTransformation()
    train_arr,test_arr,_=obj_trans.initiate_data_trasformation(train_data,test_data)

        

