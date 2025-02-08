import numpy as np 
from sentence_transformers import SentenceTransformer
import pandas as pd


train=pd.read_csv('./task-specific-datasets/banking_data/train.csv')
test=pd.read_csv('./task-specific-datasets/banking_data/test.csv')
data=pd.concat([train,test]).reset_index()
data=data.drop(columns=['index'])

non_anom=['card_linking','activate_my_card','visa_or_mastercard','card_linking',
'visa_or_mastercard','country_support','supported_cards_and_currencies'
,'fiat_currency_support','card_acceptance','edit_personal_details'
,'getting_spare_card','card_about_to_expire','apple_pay_or_google_pay',
'order_physical_card','exchange_rate','card_delivery_estimate','age_limit']

data['class']=data['category'].apply(lambda x: 0 if x in non_anom else 1 )

print('attempting full')
# performing SBERT vectorization
sb_model = SentenceTransformer("all-MiniLM-L6-v2") # 1 of the 3 mentioned in article 
embeddings = sb_model.encode(data['text'])
print(embeddings.shape)
print('completedt embedding')
np.save('./embedding.npy', embeddings)

print('Saved file as embedding.npy')

      


