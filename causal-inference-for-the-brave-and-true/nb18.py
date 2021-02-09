import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from toolz import merge
from sklearn.preprocessing import LabelEncoder

def ltv_with_coupons(coupons=None):
    
    n = 10000
    t = 30
    
    np.random.seed(12)

    age = 18 + np.random.poisson(10, n)
    income = 500+np.random.exponential(2000, size=n).astype(int)
    region = np.random.choice(np.random.lognormal(4, size=50), size=n)

    if coupons is None:
        coupons = np.clip(np.random.normal((age-18), 0.01, size=n) // 5 * 5, 0, 15)
    
    assert len(coupons) == n

    np.random.seed(12)
    
    # treatment effect on freq
    freq_mu = 0.5*coupons * age + age
    freq_mu = (freq_mu - 150) / 30
    freq_mu += 2
    

    freq = np.random.lognormal(freq_mu.astype(int))

    churn = np.random.poisson((income-500)/2000 + 22, n)

    ones = np.ones((n, t))
    alive = (np.cumsum(ones, axis=1) <= churn.reshape(n, 1)).astype(int)
    buy = np.random.binomial(1, ((1/(freq+1)).reshape(n, 1) * ones))

    cacq = -1*abs(np.random.normal(region, 2, size=n).astype(int))

    # treatment effect on transactions
    np.random.seed(12)
    transactions = np.random.lognormal(2, size=(n, t)).astype(int) * buy * alive

    transaction_mu = 0.1 + (((income - 500) / 900) * (coupons/8)) + coupons/9
    transaction_mu = np.clip(transaction_mu, 0, 5)
    transaction_mu = np.tile(transaction_mu.reshape(-1,1), t)
    
    np.random.seed(12)
    transactions = np.random.lognormal(transaction_mu, size=(n, t)).astype(int) * buy * alive

    data = pd.DataFrame(merge({"customer_id": range(n), "cacq":cacq},
                              {f"day_{day}": trans 
                               for day, trans in enumerate(transactions.T)}))

    encoded = {value:index for index, value in
           enumerate(np.random.permutation(np.unique(region)))}

    customer_features = pd.DataFrame(dict(customer_id=range(n), 
                                          region=region,
                                          income=income,
                                          coupons=coupons,
                                          age=age)).replace({"region":encoded}).astype(int)
    
    return data, customer_features
