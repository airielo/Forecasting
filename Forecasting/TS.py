import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

from statsmodels.stats.diagnostic import lilliefors
from scipy.special import inv_boxcox

from fbprophet import Prophet

%matplotlib inline


class TS:
    def __init__(self, df):
        
        #df = df.reset_index()
        #df.columns = ['ds', 'y']
        
        self.raw_df = df
        
    def preprocess(self, test_size=0.3):
        
        # cleaning the data
        df = self.raw_df.dropna().drop_duplicates()
        
        if len(df) > 365:
            self.step = "W"
            
        else:
            self.step = "D"
        
        # aggregating data by daily sales
        df = df.resample(self.step).apply(sum)
                
        self.df = df.reset_index()
        self.df.columns = ["ds", "y"]
        
        self.index = int(len(self.df)*test_size)
        
        # ----------------------------------------
        
        #---------- Test of normality ------------
        
        # ----------------------------------------
        
        # Lilliefors Test
        self.lilliefors_D, p = lilliefors(self.df.y)
        
        #Kolmogorov-Smirnov Goodness of Fit Test statistic at 0.05% significance
        self.KS_stat_05 = 1.36 / len(self.df)**0.5        
        
        if self.lilliefors_D > self.KS_stat_05:
            print("[ The H0 normality hypothesis at alpha = 0.05 is rejected ]")
            print("[ Lilliefors test statistic: {:.5f}, Kolmogorov-Smirnov ".format(self.lilliefors_D) +
                  "critical value: {:.5f} ]".format(self.KS_stat_05))
            self.normalize = True
                                                                                                        
        
        # Box-Cox transformation
        
        if self.normalize:
            
            self.df = self.df[self.df.y > 0]
            
            x, self.optimal_lambda = stats.boxcox(self.df.y[:-self.index])
            print("[ Applying Box-Cox Transformation. Optimal lambda: {:.5f} ]".format(self.optimal_lambda))
            self.df.y = stats.boxcox(self.df.y, self.optimal_lambda)
                
    def predict(self):
            
        
        if self.step == "W":
            self.index_prophet = self.index*7
        else:
            self.index_prophet = self.index
        
        train = self.df[:-self.index]
        
        self.model = Prophet()
        self.model.fit(train)
        
        future = self.model.make_future_dataframe(periods=self.index_prophet)
        
        self.forecast = self.model.predict(future)
        
        if self.normalize:
            for x in ["trend","yhat", 'yhat_lower', 'yhat_upper']:
                self.forecast[x] = inv_boxcox(self.forecast[x], self.optimal_lambda)
            self.df.y = inv_boxcox(self.df.y, self.optimal_lambda)        
        
        
    def assess(self):
        
        # combining prediction with test values
        forecast = self.forecast.set_index("ds")
        forecast = forecast[["yhat", 'yhat_lower', 'yhat_upper']]
        test = self.df.set_index("ds")
        
        self.combined_df = forecast.join(test)
                
        df = self.combined_df.copy()
        
        df['e'] = df['y'] - df['yhat']
        df['p'] = 100 * df['e'] / df['y']
        
        predicted = df[-self.index:]
        
        self.MAPE = np.mean(np.abs(predicted["p"]))
        
    
    def plot(self):
        
        plt.figure(figsize=(15, 7))
        plt.plot(self.combined_df["yhat"], "C9", label="prediction", linewidth=2.0)
        plt.plot(self.combined_df["y"], "black",label="actual", linewidth=1.0, alpha=0.8)
        plt.fill_between(self.combined_df.index,self.combined_df["yhat_lower"], 
                         self.combined_df["yhat_upper"],color="C9", alpha=0.5)
        
        plt.legend(loc="best")
        plt.grid(True)
        plt.show()
        

    
