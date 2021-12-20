import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from tensorflow.keras.utils import to_categorical
from ELM import ELM


#------- DATA PREP --------

def fill_OHLC_gap(df, n_markets=3):
    data = df.copy()
    correlations = data.corr(method='pearson')
    
    # Create np.array where each row will correspond to O/H/L/C column name for
    # each of markets
    ohlc_cols = np.zeros((3, 4), dtype=object)
    for i in range(3):
        ohlc_cols[i] = [f'Market{i+1}: Close',
                        f'Market{i+1}: High',
                        f'Market{i+1}: Low',
                        f'Market{i+1}: Open']
    ohlc_cols = ohlc_cols.T

    for pricepoint in ohlc_cols:             # For Highs, Lows, Opens, Closes
        for market in pricepoint:            # For each market's O/H/L/C
            if data[market].isnull().any():  # If has NULLs
                
                # Find the best correlated market to currently iterated
                idx = pd.Index(pricepoint)
                idx_excluded = idx.difference(pd.Index([market]))
                market_match = correlations.loc[idx_excluded, market].idxmax()
                
                # Determine the ratio of the columns, replace NULLs
                def replace_missing(col1, col2):
                    c1, c2 = (col1.copy(), col2.copy())
                    ratio = (c1 / c2).mean()
                    idx_nulls = c1.isnull()
                    c1.loc[idx_nulls] = c2.loc[idx_nulls] * ratio
                    return c1
                
                data[market] = replace_missing(data[market],
                                               data[market_match])
                
                # If still missing, try to use thrid market's data
                if data[market].isnull().any():
                    market2 = pricepoint[(pricepoint!=market) &
                                         (pricepoint!=market_match)]
                    data[market] = replace_missing(data[market],
                                                  data[market2[0]])
                    
    return data


def make_model(series, pdq, pdqs, plot=False):
    '''Returns ARIMA model'''
    model = ARIMA(series.to_numpy(),
                  order=pdq,
                  seasonal_order=pdqs,
                  missing='none')

    model_fit = model.fit()
    if plot:
#       plt_ts_corr(pd.Series(model_fit.resid), pd.Series(model_fit.resid))
        plt.plot(model_fit.predict()[200:300])
        plt.plot(series.to_numpy()[200:300])
        plt.show()
    
    return model_fit

def fill_from_model(series, model):
    '''Fills missing points by ARIMA model prediction'''
    prediction = pd.Series(model.predict(), index=series.index)
    series.fillna(prediction, inplace=True)

def fill_vol_oi(df):
    '''Fills missing points of volume/oi columns by ARIMA models'''
    m2_oi_model = make_model(df['Market2: Open_Interest'],
                             (1,1,0), (0,0,0,0))
    m3_oi_model = make_model(df['Market3: Open_Interest'],
                             (1,1,0), (0,0,0,0))
    m2_vol_model = make_model(df['Market2: Volume'],
                              (2,1,0), (2,0,0,3))
    m3_vol_model = make_model(df['Market3: Volume'],
                              (2,1,0), (2,0,0,3))
    
    cols = ['Market2: Open_Interest', 'Market3: Open_Interest',
            'Market2: Volume', 'Market3: Volume']
    models = [m2_oi_model, m3_oi_model,
              m2_vol_model, m3_vol_model]
    for i, col in enumerate(cols):
        fill_from_model(df[col], models[i])

    return df    

def prepare_data(df):
    '''Prepares data by calling all above functions'''
    df = fill_OHLC_gap(df, n_markets=3)
    df = fill_vol_oi(df)
    # Drop any nan rows
    df.dropna(inplace=True)

    df['Market1: Direction'] = \
        (df['Market1: Close']  \
        -df['Market1: Open']).map(lambda x: 0 if x<0 else 1)
            
    df['Market2: Direction'] = \
        (df['Market2: Close']  \
        -df['Market2: Open']).map(lambda x: 0 if x<0 else 1)
            
    df['Market3: Direction'] = \
        (df['Market3: Close']  \
        -df['Market3: Open']).map(lambda x: 0 if x<0 else 1)
    
    return df


#------- ELM FUNCTIONS --------

def z_score(array, u, std):
    return (array-u) / std

def split(series, k, holdout, normalize=True):
    # To ensure objectivity, reserve last 50 days for model ranking
    holdout  = 50 + k                             # Comment out when (gen algo)
    test     = series[-holdout:]
    series   = series[:-holdout]

    # Split rest 80:20 (adjust for k extra points needed in each split)
    valid_size = int(np.floor(0.2 * (len(series)-2*k)//50) * 50) + k
    #valid_size = 300+k                           # Use when using gen algo
    train = series[valid_size:]
    valid = series[:valid_size]

    return train, valid, test

# x_np is single feature only currently!
def ELM_data_prep(X_np, k, holdout, y_np=None,
                  normalize_x=True,normalize_y=True,
                  verbose=False):
   
    # Get input splits
    X_train, X_valid, X_test = split(X_np, k, holdout)

    # If labels (y_np) are not provided we are predicting time series itself
    # thus y labels are generated from X
    if type(y_np) is not np.ndarray:
        # lables as array (Nx1)
        y_train = X_train.reshape((-1, 1))[k:]   # We slice out first k points
        y_valid = X_valid.reshape((-1, 1))[k:]   # reserved for first predict.
        y_test  = X_test.reshape((-1, 1))[k:] 
    else:
        y_train, y_valid, y_test = split(y_np, k, holdout)
        y_train, y_valid, y_test = (y_train[k:], y_valid[k:], y_test[k:])
     
    # Normalize all sets by X_train statistics.
    u_norm     = X_train.mean()
    std_norm   = X_train.std()
    if normalize_x:
        X_train  = z_score(X_train, u_norm, std_norm)
        X_valid  = z_score(X_valid, u_norm, std_norm)
        X_test   = z_score(X_test, u_norm, std_norm)   
    if normalize_y:                                      # Do not normalize
        y_train  = z_score(y_train, u_norm, std_norm)    # labels in 
        y_valid  = z_score(y_valid, u_norm, std_norm)    # classification
        y_test   = z_score(y_test, u_norm, std_norm)          
        
    # TODO: rework such that X can be actual design matrix, not just series
    # X matrix as 2D(Nxk) array where each row will be k preceeding points
    def stack(array, k):
        stacked = np.zeros((len(array)-k, k))
        for i, _ in enumerate(stacked):
            stacked[i] = array[i:i+k]
        return stacked    
    
    X_train = stack(X_train, k)
    X_valid = stack(X_valid, k)
    X_test  = stack(X_test, k)   
    
    if verbose:
        print(f'Train samples: {len(X_train)}',
              f'\nValidation samples: {len(X_valid)}',
              f'\nTest samples: {len(X_test)}')           
        print(X_train.shape, y_train.shape)
        print(X_valid.shape, y_valid.shape)
        print(X_test.shape, y_test.shape)    
    
    return X_train, y_train, X_valid, y_valid, X_test, y_test, u_norm, std_norm



# Model is fed input vector [1xk] and outputs next point prediction
def get_ELM(k, hidden, X_train, y_train, num_classes=1):
    input_length = k
    num_hidden_layers = hidden

    model = ELM(input_length,
                num_hidden_layers,
                num_classes)
    
    model.fit(X_train, y_train, display_time=False)
    return model

# Evaluate time-series denormalized rmse
def evaluate(X_valid, y_valid, model, u_norm, std_norm, output='rmse'):
    prediction = model(X_valid)
    prediction = prediction * std_norm + u_norm    #De-normalized
    y_valid    = y_valid * std_norm + u_norm       #De-normalized
    residuals  = y_valid - prediction
    
    if output=='prediction':
        return prediction
    
    return np.sqrt(np.mean((residuals)**2))

def polish_ELM(hyperparams, X_train, y_train, X_valid, y_valid,
               directional=False, n=1000):
    '''retrains ELM model and keep the best one'''
    k, hidden  = hyperparams
    k          = round(k)
    hidden     = round(hidden)    
    lowest_error = np.inf
    best_model   = None
    
    classes = 2 if directional else 1
    
    for i in range(n):
        # Optimal model training
        model = get_ELM(k, hidden, X_train, y_train, num_classes=classes)

        if directional:
            _, acc = model.evaluate(X_valid, y_valid)  
            err = 1-acc
        else:
            err = evaluate(X_valid, y_valid, model,
                          0, 1, output='rmse')

        if err < lowest_error:
            lowest_error = err
            best_model = model

    return best_model

def forecast(data, model, forecast_len, forecast_start=None,
             oos_only=False):
    # Single point prediction
    insample = data[:forecast_start]
    prediction = model(insample)
    k = model._num_input_nodes
    
    # Forecast takes first input vector after forcast start
    if forecast_start == None:
        forecast_start = -1       
    inp_forecast = (data[forecast_start])

    forecast = []
    for i in range(forecast_len):
        new_point = model(inp_forecast)
        forecast.append(new_point[0])
        inp_forecast = np.append(inp_forecast, new_point)[1:]
    
    if oos_only:
        return forecast
    
    entire = np.append(prediction, forecast)
    return entire

def next_day(dataset):
    holdout = 0
    elm_hyperparams_dir = (16, 74)
    elm_hyperparams_ts  = (12, 86)
    
    direction = dataset['Market2: Direction'].to_numpy()
    direction = to_categorical(direction, 2).astype(np.float32)  
    t_series = dataset['Market2: Open'].to_numpy()
   
    # DIRECTION PREDICTION
    k = elm_hyperparams_dir[0]
    X_train, y_train, X_valid, y_valid,\
    X_test, y_test, u_norm, std_norm = ELM_data_prep(t_series, k, holdout,
                                                     y_np=direction,
                                                     normalize_y=False)
    directional = polish_ELM(elm_hyperparams_dir, X_train, y_train,
                             X_valid, y_valid, directional=True, n=2000)
    # Last k points for first out of sample prediction
#    latest = np.array(t_series[-k:]).reshape((1,-1))
    next_dir = directional(X_test[-1])
    next_dir = 'Bearish' if next_dir.argmax()==0 else 'Bullish'

    # PRICE PREDICTION
    k = elm_hyperparams_ts[0]
    X_train, y_train, X_valid, y_valid,\
    X_test, y_test, u_norm, std_norm = ELM_data_prep(t_series, k, holdout)
    
    pricemove = polish_ELM(elm_hyperparams_ts, X_train, y_train,
                             X_valid, y_valid, n=2000)
    # Last k points for first out of sample prediction
#    latest = np.array(t_series[-k:]).reshape((1,-1))    
    next_move = pricemove(X_test[-1])
    next_move = next_move * std_norm + u_norm
    
    return next_dir, next_move