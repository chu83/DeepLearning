
import datetime

now = datetime.datetime.now()
print(now)

dstr = f'{now : %Y%m%d%H%M%S}'
print(dstr)