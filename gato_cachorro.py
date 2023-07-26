from tensorflow.keras.models import Sequential # atualizado: tensorflow==2.0.0-beta1
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout # atualizado: tensorflow==2.0.0-beta1
from tensorflow.python.keras.layers.normalization import BatchNormalization # atualizado: tensorflow==2.0.0-beta1
from tensorflow.keras.preprocessing.image import ImageDataGenerator # atualizado: tensorflow==2.0.0-beta1
import numpy as np
from tensorflow.keras.preprocessing import image # atualizado: tensorflow==2.0.0-beta1

# rede neural convolucional
# 3 canais por estar trabalhando com rgb
classificador = Sequential()
classificador.add(Conv2D(32, (3,3), input_shape = (64, 64, 3), activation = 'relu'))
classificador.add(BatchNormalization())
classificador.add(MaxPooling2D(pool_size = (2,2)))

classificador.add(Conv2D(32, (3,3), input_shape = (64, 64, 3), activation = 'relu'))
classificador.add(BatchNormalization())
classificador.add(MaxPooling2D(pool_size = (2,2)))

classificador.add(Flatten())

# rede neural densa
classificador.add(Dense(units = 128, activation = 'relu'))
classificador.add(Dropout(0.2))
classificador.add(Dense(units = 128, activation = 'relu'))
classificador.add(Dropout(0.2))
classificador.add(Dense(units = 1, activation = 'sigmoid'))

classificador.compile(optimizer = 'adam', loss = 'binary_crossentropy',
                      metrics = ['accuracy'])

gerador_treinamento = ImageDataGenerator(rescale = 1./255,
                                         rotation_range = 7,
                                         horizontal_flip = True,
                                         shear_range = 0.2,
                                         height_shift_range = 0.07,
                                         zoom_range = 0.2)
gerador_teste = ImageDataGenerator(rescale = 1./255)

base_treinamento = gerador_treinamento.flow_from_directory('dataset/training_set',
                                                           target_size = (64, 64),
                                                           batch_size = 32,
                                                           class_mode = 'binary')
base_teste = gerador_teste.flow_from_directory('dataset/test_set',
                                               target_size = (64, 64),
                                               batch_size = 32,
                                               class_mode = 'binary')

# treinamento
classificador.fit_generator(base_treinamento, steps_per_epoch = 4000 / 32,
                            epochs = 5, validation_data = base_teste,
                            validation_steps = 1000 / 32)

# criando a imagem para teste
imagem_teste = image.load_img('dataset/test_set/cachorro/dog.3500.jpg',
                              target_size = (64,64))
# conversão da imagem
imagem_teste = image.img_to_array(imagem_teste)

# normalização (para os valores ficarem na escala entre 0 e 1)
imagem_teste /= 255

# expandir as dimensoes (adiciona mais uma coluna na imagem_teste)
imagem_teste = np.expand_dims(imagem_teste, axis = 0)

previsao = classificador.predict(imagem_teste)
previsao = (previsao > 0.5)

# cachorro = 0
# gato = 1
base_treinamento.class_indices