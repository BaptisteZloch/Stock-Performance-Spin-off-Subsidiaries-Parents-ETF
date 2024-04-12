
from datetime import datetime

import pandas as pd
from data.bloomberg_access import BloombergAPI
import pandas_market_calendars as mcal
class DataManager:
    
    @staticmethod
    def create_rebalancing_calendar(start_date: datetime, end_date: datetime):
        """
        Create a rebalancing calendar with business days between the start and end date.
    
        Parameters:
        - start_date (datetime): The start date of the calendar.
        - end_date (datetime): The end date of the calendar.
      
        Returns:
        list[datetime]: A list of rebalancing dates from start_date to end_date
        Raises:
        ValueError: If start_date is after end_date.
        """
      
        # Création d'un calendrier de trading pour la bourse US
        nyse = mcal.get_calendar('NYSE')

        # Générer les dates de rebalancement à la fin de chaque mois
        rebalance_dates = nyse.valid_days(start_date=start_date, end_date=end_date)
        rebalance_dates = [date.to_pydatetime().date() for i, date in enumerate(rebalance_dates[:-1]) if rebalance_dates[i + 1].month != date.month]

        return rebalance_dates


    

    @staticmethod
    def get_rebalancing_date(date, step):
        """
        Calculate the rebalancing date by adding a specified number of months to the given date.
    
        Parameters:
        - date (datetime): The starting date for calculating the rebalancing date.
        - step (int): The number of months to add to the given date.
    
        Returns:
        datetime.date: The calculated rebalancing date.
        """
        # Ajoute ou soustrait le nombre de mois à la date de départ
        rebalancing_date = date + pd.DateOffset(months=step)
    
        # Récupère le dernier jour du mois pour la date calculée
        rebalancing_date = rebalancing_date + pd.offsets.MonthEnd(0)
        
        # # Convertir la date en datetime avec un fuseau horaire
        # rebalancing_datetime = datetime.combine(rebalancing_date, datetime.min.time())
        rebalancing_date=rebalancing_date.date()
        # Création d'un calendrier de trading pour la bourse US
        nyse = mcal.get_calendar('XNYS')
        valid_days_index = nyse.valid_days(start_date=(rebalancing_date-pd.Timedelta(days=3)), end_date=rebalancing_date)
        valid_days_list = [date.to_pydatetime().date() for date in valid_days_index]
        if not valid_days_list:
            return rebalancing_date
        
        while rebalancing_date not in valid_days_list:
            # Si la date calculée n'est pas un jour ouvré, ajustez-la jusqu'à obtenir un jour ouvré
            rebalancing_date -= pd.Timedelta(days=1)
        
        return rebalancing_date

    
    @staticmethod
    def check_universe(universe, market_data, date, next_date):
        """
        Check if the market data for each ticker in the universe is available between the given dates.
    
        Parameters:
        - universe (List[str]): List of ticker symbols representing the universe of assets.
        - market_data (Dict[str, pd.DataFrame]): Dictionary containing market data for each ticker.
        - date (datetime): The start date for checking market data availability.
        - next_date (datetime): The end date for checking market data availability.
    
        Returns:
        List[str]: List of ticker symbols for which market data is available between the given dates.
        """        
        return [ticker for ticker in universe if DataManager.check_data_between_dates(market_data[ticker], date, next_date)]
    
    @staticmethod
    def check_data_between_dates(df, start_date, end_date):
        """
        Check if there is non-NaN data between two dates in a DataFrame.
        
        Args:
        df (pd.DataFrame): DataFrame with dates as index.
        start_date (str): Start date (inclusive).
        end_date (str): End date (inclusive).
        
        Returns:
        bool: True if there is non-NaN data between the specified dates, False otherwise.
        """
        # Vérification si les dates existent dans l'index
        if start_date not in df.index or end_date not in df.index:
            return False
        
        # Vérification si des données existent entre les deux dates
        data_subset = df.loc[(df.index > start_date) & (df.index < end_date)]
        if not data_subset.empty:
            return True
        return False
            
    
    @staticmethod
    def get_historical_compositions(start_date : datetime, end_date : datetime, ticker : str):
        strFields = ["INDX_MWEIGHT_HIST"]  
        blp = BloombergAPI()
        
        rebalancing_dates = DataManager.create_rebalancing_calendar(start_date, end_date)
        composition_par_date = {}


        for date in rebalancing_dates:
            str_date = date.strftime('%Y%m%d')
            compo = blp.bds(strSecurity=[ticker], strFields=strFields, strOverrideField="END_DATE_OVERRIDE", strOverrideValue=str_date)
            list_tickers = compo[strFields[0]].index.tolist()
            composition_par_date[date] = [ticker.split(' ')[0] + ' US Equity' for ticker in list_tickers]
            
        return composition_par_date
        
    @staticmethod
    def get_historical_prices(start_date : datetime, end_date : datetime, tickers : list[str], curr : str):
        
        blp = BloombergAPI()
        global_market_data = {}
        tickers_a_supp = []
        
        for ticker in tickers:
            try:
                historical_prices = blp.bdh(strSecurity=[ticker], strFields=["PX_LAST"], startdate=start_date, enddate=end_date, curr=curr, fill="NIL_VALUE")
                historical_prices["PX_LAST"]=historical_prices["PX_LAST"].sort_index(ascending=True)
                
                if not historical_prices["PX_LAST"].empty:
                    global_market_data[ticker] = historical_prices["PX_LAST"]
                else:
                    tickers_a_supp.append[ticker]
                    
            except Exception as e:
                print(f"Erreur lors du traitement du ticker {ticker}: {str(e)}")
            continue
        
        return global_market_data,tickers_a_supp
    
    @staticmethod
    def fetch_backtest_data(start_date : datetime, end_date : datetime, ticker : str, curr : str):
        
    
        composition_par_date = DataManager.get_historical_compositions(start_date, end_date, ticker)
    
        tickers_uniques = list({ticker for composition in composition_par_date.values() for ticker in composition})
        start_date = DataManager.get_rebalancing_date(start_date, step=-6)   
       
        global_market_data, tickers_a_supp = DataManager.get_historical_prices(start_date, end_date, tickers_uniques, curr)
        
        # suppression des tickers sans données 
        composition_par_date = {date: [ticker for ticker in tickers if ticker not in tickers_a_supp] for date, tickers in composition_par_date.items()}    
        
        return composition_par_date, global_market_data     


    @staticmethod
    def fetch_other_US_data(start_date : datetime, end_date : datetime, ticker : str):
                  
         tickers = ["USRINDEX Index","US0003M Index"]
         tickers.append(ticker)
         other_US_data ={}
         
         other_US_data, tickers_a_supp = DataManager.get_historical_prices(start_date, end_date, tickers, curr)
             
         return other_US_data       
                
                             