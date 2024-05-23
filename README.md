# Replication of the research paper : *The Stock Price Performance of Spin-Off Subsidiaries, Their Parents, and the Spin-Off ETF, 2001-2013* by JOHN J. MCCONNELL, STEVEN E. SIBLEY, AND WEI XU.
# Setting up the project
Run the file `install_for_windows.bat`, it will install dependencies and create a virtual environment for the project.

All the code is in the `src` folder.
The research paper and the project guidelines are in the `static` folder.

# Execute the script
`python .\src\execute_strategy.py --index "SXXP Index" -p "long" -s "2020-01-01" -e "2020-12-31"`


`python .\src\execute_strategy.py --index "SX5E Index" -p "short" -s "2015-01-01" -e "2020-12-31"`


`python .\src\execute_strategy.py --index "RTY Index" -p "short" -s "2015-01-01" -e "2020-12-31" --backtest_type "parents"`


`python .\src\execute_strategy.py --index "SPX Index" -p "long" -s "2015-01-01" -e "2020-12-31" --backtest_type "subsidiaries"`


`python .\src\execute_strategy.py --index "SX5E Index" -p "short" -s "2015-01-01" -e "2020-12-31" --transaction_cost 0.1` 

