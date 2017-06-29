from keras.models import Sequential
from keras.layers import Dense
import numpy

##numpy.random.seed(56)

df= numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")

x=df[:,0:8]
y=df[:,8]

model = Sequential()
model.add(Dense(12, input_dim=8, init='uniform', activation='relu'))
model.add(Dense(8, init='uniform', activation='relu'))
model.add(Dense(1, init='uniform', activation='sigmoid'))

##model.compile(loss='binary-crossentropy',optimizer = 'adam', metrics=['accuracy'])
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(x,y, epochs = 150, batch_size = 10)

scores = model.evaluate(x,y)

model.save("keras_test.h5")

print("\n%s: %.2f%%" %(model.metrics_names[1],scores[1]*100))

"""
to make predictions we would use function
model.predict(<input array>) 

LOading saving model in KERAS 

model.save(<nm.h5>)
del model
model = load_model(<nm.h5>)

model architecture------>
json_string= model.to_json()
model = model_from_json()

"""







