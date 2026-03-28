data=pd.read_csv("creditcard.csv")

x=data.drop("Class",axis=1)
y=data["Class"]

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

model=RandomForestClassifier(n_estimators=100)
model.fit(x_train,y_train)

joblib.dump(model,"model.pkl")

print("Model saved")
