from libraries import *

fashion_mnist=keras.datasets.fashion_mnist
(train_images, train_labels),(test_images, test_labels)=fashion_mnist.load_data()

class_names=['T-shirt/top','Trouser','Pullover','Dress','Coad','Sandal','Shirt','Sneaker','Bag','Ankle boot']


train_images=train_images/255.0
test_images=test_images/255.0

model=keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128,activation='relu'),
    keras.layers.Dense(10,activation='softmax')
])
model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

model.fit(train_images, train_labels,epochs=2)

test_loss,test_accuracy=model.evaluate(test_images,test_labels,verbose=2)
print("Test loss = %s"%test_loss)
print("Test accuracy = %s"%test_accuracy)

predictions=model.predict(test_images)
print(predictions[0])
print("The model predicts the image is %s"%class_names[np.argmax(predictions[0])])
print("The image is actually %s"%class_names[test_labels[0]])