@echo off
set /p "port=Enter port number: "
set /p "address=Enter address: "
set /p "conda=Enter full path to conda.bat: "
call %conda% activate finestra
panel serve deflection_chart.ipynb --port=%port% --allow-websocket-origin=%address%:%port% --address=%address% --prefix=dgu_deflection
