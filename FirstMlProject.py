# Load libraries
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Load dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv(url, names=names)#The names=names argument specifies that the columns of the dataset should be labeled with the names provided in the names list


'''
# shape
print(dataset.head(20))#<- this print first 20 row #print(dataset.iloc[52])<- this will print 52th row
# descriptions
print(dataset.describe())

#class distribution , count of class type
print (dataset.groupby('class').size())

'''


# Univariate Plots  -> plots of individual variable
#box and whisker plots
'''dataset.plot(kind='box' , subplots = True , layout = (2,2), sharex= False , sharey = False)
plt.show()'''
'''kind='box': Specifies that you want to create box plots.
subplots=True: This will create a separate subplot for each column in the DataFrame.
layout=(2,2): This arranges the subplots in a 2x2 grid. If you have more than four columns, the layout will adjust accordingly.
sharex=False: Each subplot will have its own x-axis.
sharey=False: Each subplot will have its own y-axis.'''


'''
#we can alse create an histogram
dataset.hist()
plt.show()'''



#Multivariate Plots -> this showes interaction between variables , scatter plot matrix
'''
scatter_matrix(dataset)
plt.show()'''

'''Set-up the test harness to use 10-fold cross validation.We will split the loaded dataset into two, 80% of which we will use to train, evaluate and
select among our models, and 20% that we will hold back as a validation dataset.'''

# Split-out validation dataset
array = dataset.values
X = array[:,0:4]
y = array[:,4]
X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1)
'''X = array[:,0:4]:

This line creates a variable X that contains all rows of the array and the columns from index 0 to 3 (the first four columns). In the context of the Iris dataset, these columns represent the sepal length, sepal width, petal length, and petal width.
y = array[:,4]:

This line creates a variable y that contains all rows of the array and only the column at index 4 (the fifth column). '''


...
# Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))
# evaluate each model in turn
results = []
names = []
for name, model in models:
	kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
	cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
	results.append(cv_results)
	names.append(name)
	print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))
	'''In this case, we can see that it looks like Support Vector Machines (SVM) has the largest estimated accuracy score at about 0.98 or 98%.'''

#A useful way to compare the samples of results for each algorithm is to create a box and whisker plot for each distribution and compare the distributions.
#Box and Whisker Plot Comparing Machine Learning Algorithms on the Iris Flowers Dataset.
# Compare Algorithms
plt.boxplot(results, labels=names)
plt.title('Algorithm Comparison')
plt.show()	



''' -> Prediction making-> We can fit the model on the entire training dataset and make predictions on the validation dataset.
'''
...
# Make predictions on validation dataset
model = SVC(gamma='auto')
model.fit(X_train, Y_train)
predictions = model.predict(X_validation)



# Evaluate predictions
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))






'''
# Make predictions on validation dataset
validation_predictions = model.predict(X_validation)

# Print the validation predictions
print("Validation Predictions:", validation_predictions)
'''


# Function to get flower measurements from user input
def get_flower_input():
  while True:
    try:

      sepal_length = float(input("🌸 Enter sepal length : "))
      sepal_width = float(input("🌼 Enter sepal width : "))
      petal_length = float(input("🌺 Enter petal length : "))
      petal_width = float(input("🌻 Enter petal width : "))
      return [sepal_length, sepal_width, petal_length, petal_width]
    except ValueError:
      print("😱 Oops! That doesn't look right. Please enter numeric values.")

# Get input for multiple flowers
new_data = []
num_samples = int(input("Enter the number of flower samples to predict: "))

for _ in range(num_samples):
    print(f"Input measurements for flower sample {_ + 1}:")
    new_data.append(get_flower_input())

# Make predictions
predictions = model.predict(new_data)

# Print the predictions
print("👀 Validation Set Predictions:", predictions)
