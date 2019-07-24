from __future__ import division
from collections import OrderedDict
import time

from quantopian.algorithm import order_optimal_portfolio
from quantopian.algorithm import attach_pipeline, pipeline_output, order_optimal_portfolio
from quantopian.pipeline import Pipeline, CustomFactor
from quantopian.pipeline.data import Fundamentals
from quantopian.pipeline.data.builtin import USEquityPricing
from quantopian.pipeline.factors import SimpleMovingAverage
from quantopian.pipeline.filters import QTradableStocksUS
import quantopian.optimize as opt
from quantopian.optimize import TargetWeights
import quantopian.algorithm as algo
from quantopian.pipeline.experimental import risk_loading_pipeline

import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import Imputer, StandardScaler

holding_days = 5 
days_for_analysis = 30

## Define the machine learnign model.
class Predictor(CustomFactor):
    
 
    # Factors that use to predict 
    factor_dict = OrderedDict([
              ('Open Price' , USEquityPricing.open),
              ('Volume' , USEquityPricing.volume),
              ('cf_yield' , Fundamentals.cf_yield),
              ('earning_yield' , Fundamentals.earning_yield),
              ('pb_ratio' , Fundamentals.pb_ratio),
              ('pe_ratio' , Fundamentals.pe_ratio), 
              ('roa' , Fundamentals.roa)
    ])
 
    columns = factor_dict.keys()
    inputs = factor_dict.values()
 
    
    def compute(self, today, assets, out, *inputs):
        
        
        ## Import Data and define y.
        
        inputs = OrderedDict([(self.columns[i] , pd.DataFrame(inputs[i]).fillna(method='ffill',axis=1).fillna(method='bfill',axis=1)) for i in range(len(inputs))]) # bring in data with some null handling.
        num_secs = len(inputs['Open Price'].columns)
        y = (np.log(inputs['Open Price']) - np.log(inputs['Open Price'].shift(holding_days))).shift(-holding_days-1).dropna(axis=0,how='all').stack(dropna=False)
        
        ## Get rid of our y value as an input into our machine learning algorithm.
        del inputs['Open Price']
 
        ## Munge X and y
        x = pd.concat([df.stack(dropna=False) for df in inputs.values()], axis=1)
        x = Imputer(strategy='median',axis=1).fit_transform(x) # fill nulls.
        y = np.ravel(Imputer(strategy='median',axis=1).fit_transform(y)) # fill nulls.
        scaler = StandardScaler()
        x = scaler.fit_transform(x) # demean and normalize
 
        ## Run Model
        model = LinearRegression()
        model_x = x[:-num_secs*(holding_days+1),:]
        model.fit(model_x, y)
 
        out[:] = model.predict(x[-num_secs:,:])

def initialize(context):
    
    # Rebalance every day, 1 hour after market open.
    algo.schedule_function(
        rebalance,
        algo.date_rules.every_day(),
        algo.time_rules.market_open(hours=1),
    )

    # contrains to the contest
    context.max_posTam = 0.045
    context.max_lever = 0.97

    # Record tracking variables at the end of each day.
    algo.schedule_function(
        record_vars,
        algo.date_rules.every_day(),
        algo.time_rules.market_close(),
    )

    # Create our dynamic stock selector.
    algo.attach_pipeline(risk_loading_pipeline(), 'risk_loading_pipeline')
    algo.attach_pipeline(make_pipeline(), 'pipeline')




def make_pipeline():
   
    # Base universe set to the QTradableStocksUS
    base_universe = QTradableStocksUS()
    
    # Count 10 day min
    mean_10 = SimpleMovingAverage(
        inputs=[USEquityPricing.close],
        window_length=10,
        mask=base_universe
    )
    
    # Count 30 day max
    mean_30 = SimpleMovingAverage(
        inputs=[USEquityPricing.close],
        window_length=30,
        mask=base_universe
    )

    percent_difference = (mean_10 - mean_30) / mean_30
    shorts = percent_difference.top(75)
    longs = percent_difference.bottom(75)

    # use the model
    pipe = Pipeline(
        columns={
            'longs': longs,
            'shorts': shorts,
            'Model': Predictor(
                window_length=days_for_analysis
                ,mask=base_universe
            )
        }
        ,screen = base_universe
    )
    
    return pipe


def before_trading_start(context, data):
    pipe_results = algo.pipeline_output('pipeline')
    
    context.longs = []
    for sec in pipe_results[pipe_results['longs']].index.tolist():
        if data.can_trade(sec):
            context.longs.append(sec)
    
    context.shorts = []
    for sec in pipe_results[pipe_results['shorts']].index.tolist():
        if data.can_trade(sec):
            context.shorts.append(sec)        
    context.risk_loading_pipeline = pipeline_output('risk_loading_pipeline')
    
def compute_target_weights(context, data):
    
    weights = {}
    if context.longs and context.shorts:
        long_weight = 10.5 / len(context.longs)
        short_weight = 9.1 / len(context.shorts)
    else:
        return weights
    
    for security in context.portfolio.positions:
        if security not in context.longs and security not in context.shorts and data.can_trade(security):
            weights[security] = 0

    for security in context.longs:
        weights[security] = long_weight+2.5

    for security in context.shorts:
        weights[security] = short_weight

    return weights

def rebalance(context, data):
    
    pipeline_output_df = pipeline_output('pipeline').dropna(how='any')
    
    # part of constrains
    max_lever = opt.MaxGrossExposure(context.max_lever)
    dollar_net = opt.DollarNeutral()
    constrain_sector_style_risk = opt.experimental.RiskModelExposure(  
        risk_model_loadings=context.risk_loading_pipeline,  
        version=0,
    )

    todays_predictions = pipeline_output_df.Model
    target_weight_series = todays_predictions.sub(todays_predictions.mean())
    target_weight_series = target_weight_series/target_weight_series.abs().sum()
    order_optimal_portfolio(
        objective=TargetWeights(target_weight_series),
        constraints=[
            #constrain_posTam,
            max_lever,
            constrain_sector_style_risk,
            dollar_net
        ]
    )

    pass


def record_vars(context, data):
    longs = shorts = 0
    for position in context.portfolio.positions.itervalues():
        if position.amount > 0:
            longs += 1
        elif position.amount < 0:
            shorts += 1

    record(
        leverage=context.account.leverage,
        long_count=longs,
        short_count=shorts
    )
    pass