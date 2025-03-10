#Bayesian network
import pandas as pd
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination

# "Play Tennis" sample dataset
data = pd.DataFrame([
    ['Sunny', 'Hot', 'High', 'Weak', 'No'],
    ['Sunny', 'Hot', 'High', 'Strong', 'No'],
    ['Overcast', 'Hot', 'High', 'Weak', 'Yes'],
    ['Rain', 'Mild', 'High', 'Weak', 'Yes'],
    ['Rain', 'Cool', 'Normal', 'Weak', 'Yes'],
    ['Rain', 'Cool', 'Normal', 'Strong', 'No'],
    ['Overcast', 'Cool', 'Normal', 'Strong', 'Yes'],
    ['Sunny', 'Mild', 'High', 'Weak', 'No'],
    ['Sunny', 'Cool', 'Normal', 'Weak', 'Yes'],
    ['Rain', 'Mild', 'Normal', 'Weak', 'Yes'],
    ['Sunny', 'Mild', 'Normal', 'Strong', 'Yes'],
    ['Overcast', 'Mild', 'High', 'Strong', 'Yes'],
    ['Overcast', 'Hot', 'Normal', 'Weak', 'Yes'],
    ['Rain', 'Mild', 'High', 'Strong', 'No']
], columns=['Outlook', 'Temperature', 'Humidity', 'Wind', 'PlayTennis'])

# Bayesian Network structure
model = BayesianNetwork([
    ('Outlook', 'PlayTennis'),
    ('Temperature', 'PlayTennis'),
    ('Humidity', 'PlayTennis'),
    ('Wind', 'PlayTennis')
])

# Maximum Likelihood Estimation
model.fit(data, estimator=MaximumLikelihoodEstimator)

# Inference
infer = VariableElimination(model)

# Query: What is the probability of playing tennis given a sunny outlook?
query_result = infer.query(variables=['PlayTennis'], evidence={'Outlook': 'Sunny'})
print(query_result)
