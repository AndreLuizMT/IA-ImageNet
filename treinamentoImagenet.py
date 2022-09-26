"""
Created on Fri Oct  8 08:45:44 2021

@author: André


https://www.analyticsvidhya.com/blog/2020/02/learn-image-classification-cnn-convolutional-neural-networks-3-datasets/


"""
#Determinação das classes de imagens contidas nas pastas
imagenette_map = { 
    "n01440764" : "tench",
    "n02102040" : "springer",
    "n02979186" : "casette_player",
    "n03000684" : "chain_saw",
    "n03028079" : "church",
    "n03394916" : "French_horn",
    "n03417042" : "garbage_truck",
    "n03425413" : "gas_pump",
    "n03445777" : "golf_ball",
    "n03888257" : "parachute"
}


from keras.preprocessing.image import ImageDataGenerator


#criação de uma nova geração para treinamento e teste do modelo
imagemgeracao = ImageDataGenerator()

#determinação do treinamento
treino = imagemgeracao.flow_from_directory("imagenette2/train/", class_mode="categorical", shuffle=False, batch_size=128, target_size=(224, 224))

validacao = imagemgeracao.flow_from_directory("imagenette2/val/", class_mode="categorical", shuffle=False, batch_size=128, target_size=(224, 224))

from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, InputLayer, BatchNormalization, Dropout

#Determino o modelo de CNN como sequencial
modelo = Sequential()

#configuro o modelo de acordo com as imagens de entrada para 224 x 224 e colorida
modelo.add(InputLayer(input_shape=(224, 224, 3)))

# primeiro bloco convolucional
modelo.add(Conv2D(25, (5, 5), activation='relu', strides=(1, 1), padding='same'))
modelo.add(MaxPool2D(pool_size=(2, 2), padding='same'))

# segundo bloco convolucional
modelo.add(Conv2D(50, (5, 5), activation='relu', strides=(2, 2), padding='same'))
modelo.add(MaxPool2D(pool_size=(2, 2), padding='same'))
modelo.add(BatchNormalization())

# terceiro bloco convolucional
modelo.add(Conv2D(70, (3, 3), activation='relu', strides=(2, 2), padding='same'))
modelo.add(MaxPool2D(pool_size=(2, 2), padding='valid'))
modelo.add(BatchNormalization())

# Rede neural configuração
modelo.add(Flatten())
modelo.add(Dense(units=100, activation='relu'))
modelo.add(Dense(units=100, activation='relu'))
modelo.add(Dropout(0.25))

# Camada de saída da CNN
modelo.add(Dense(units=10, activation='softmax'))

# compilar o modelo
modelo.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])

# Treinamento dos dados para 30 épocas
modelo.fit_generator(treino, epochs=30, validation_data=validacao)
    