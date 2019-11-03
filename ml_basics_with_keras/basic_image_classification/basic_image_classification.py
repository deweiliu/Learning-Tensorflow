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

def plot_image(i, predictions_array, true_label, img):
  predictions_array, true_label, img = predictions_array, true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  predictions_array, true_label = predictions_array, true_label[i]
  plt.grid(False)
  plt.xticks(range(10))
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')

i = 12
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions[i],  test_labels)
plt.show()

# Add the image to a batch where it's the only member.
img = (np.expand_dims(test_images[1],0))
predictions=model.predict(img)
plt.figure()
plot_value_array(1, predictions[0], test_labels)
_ = plt.xticks(range(10), class_names, rotation=45)
print(class_names[np.argmax(predictions[0])])
plt.show()
