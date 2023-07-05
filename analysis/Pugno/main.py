import pandas as pd  
import numpy as np
from sklearn import preprocessing
from collections import Counter
from sklearn.preprocessing import LabelEncoder
df = pd.read_csv('data\SpotifySongPolularityAPIExtract.csv')
#df.head(100)
#print(df.head(100))
#df.info()
#df.describe()   
#df.isnull() 
#df.dropna()
#df.index
#df.to_numpy()
#print(df.to_numpy())
#merged_text = ' '.join(df['track_name'].astype(str))

# Suddividi la stringa in parole
#words = merged_text.split()

# Conta le occorrenze di ogni parola
#word_counts = Counter(words)

# Ottieni i primi 20 elementi più ripetuti
#top_20 = word_counts.most_common(20)

# Visualizza i risultati
#for word, count in top_20:
    #print(f'{word}: {count}')

#duplicated_texts = df[df.duplicated(subset='track_name', keep=False)]
#duplicated_texts = duplicated_texts.sort_values('track_name')
#print(data['artist_name'].unique())
#print(duplicated_texts['track_name'])
from sklearn.preprocessing import LabelEncoder

data = df.copy()

#label_encoder = LabelEncoder()

# Seleziona le colonne non numeriche
#non_numeric_columns = data.select_dtypes(exclude=['int', 'float']).columns

# Codifica le colonne non numeriche
#for column in non_numeric_columns:
 #   data[column] = label_encoder.fit_transform(data[column])
#selected_columns = ['artist_name', 'track_id', 'track_name']
#data = df[selected_columns]
#data = data.fillna(0) 
#scaler = preprocessing.StandardScaler()
#scaled_data = scaler.fit_transform(data)

artista = 'artist_name'
filtro_artista = data['artist_name'] == artista
dati_artista = data[filtro_artista]
media = dati_artista['popularity'].mean()

colonna_categorica = 'artist_name'
data['target_encoding'] = data.groupby(colonna_categorica)['popularity'].transform('mean')