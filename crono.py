import os 
import subprocess
from crontab import CronTab
from util import is_windows
from util import setJsonValue
import traceback

def startSimulation():#chatGpt
        morning_path = os.path.abspath('./morningRun.py')
        evening_path = os.path.abspath('./eveningRun.py')
        if is_windows():
            # Windows Task Scheduler
            subprocess.run(f'schtasks /Create /SC WEEKLY /D MON,TUE,WED,THU,FRI /TN "stockPredictMorningRun" /TR "python {morning_path}" /ST 09:35 /F', shell=True)
            subprocess.run(f'schtasks /Create /SC WEEKLY /D MON,TUE,WED,THU,FRI /TN "stockPredictEveningRun" /TR "python {evening_path}" /ST 15:55 /F', shell=True)
        else:
            # Linux/macOS with cron
            cron = CronTab(user=True)
            job = cron.new(command=f'python3 {morning_path}', comment='stockPredictMorningRun')
            job2 = cron.new(command=f'python3 {evening_path}', comment='stockPredictEveningRun')
            job.setall(35, 9, '*', '*', '1-5')
            job2.setall(55, 15, '*', '*', '1-5')
            cron.write()
        setJsonValue("./Data/systemMetaData.json", "status", "online")



def stopSimulation():#ChatGpt
    if is_windows():
        subprocess.run('schtasks /Delete /TN "stockPredictMorningRun" /F', shell=True)
        subprocess.run('schtasks /Delete /TN "stockPredictEveningRun" /F', shell=True)
    else:
        cron = CronTab(user=True)
        for job in cron:
            if job.comment in ['stockPredictMorningRun', 'stockPredictEveningRun']:
                cron.remove(job)
        cron.write()
    setJsonValue("./Data/systemMetaData.json", "status", "offline")