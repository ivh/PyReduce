# -*- coding: utf-8 -*-
import datetime as dt
from os.path import dirname, join

import numpy as np
from astroquery.eso import Eso

instrument = "HARPS"
dates = [
    "2003-12-13",
    "2005-01-04",
    "2005-05-26",
    "2005-05-27",
    "2006-03-21",
    "2008-04-12",
    "2009-11-18",
    "2009-11-23",
    "2009-11-24",
    "2009-11-25",
    "2009-11-26",
    "2009-11-27",
    "2009-11-29",
    "2009-11-30",
    "2009-11-30",
    "2009-12-02",
    "2010-02-08",
    "2010-02-09",
    "2010-03-21",
    "2010-03-22",
    "2010-03-23",
    "2010-03-24",
    "2010-03-25",
    "2010-03-26",
    "2010-03-27",
    "2010-03-28",
    "2010-03-29",
    "2010-04-18",
    "2010-04-20",
    "2010-04-21",
    "2010-04-22",
    "2010-04-23",
    "2011-01-13",
    "2011-01-31",
    "2011-02-19",
    "2011-02-23",
    "2011-02-27",
    "2011-03-11",
    "2011-03-20",
    "2011-04-03",
    "2011-04-21",
    "2011-04-30",
    "2013-01-17",
    "2013-01-18",
    "2013-01-19",
    "2013-01-22",
    "2013-01-23",
    "2013-01-26",
    "2013-01-30",
    "2013-02-01",
    "2013-02-06",
    "2013-02-07",
    "2013-02-08",
    "2013-02-11",
    "2013-02-13",
]

destination = join(dirname(__file__), "raw")
print(destination)

dates = [dt.datetime.strptime(d, "%Y-%m-%d") for d in dates]
day = dt.timedelta(days=1)


eso = Eso()
eso.login("awehrhahn")

for d in dates:
    filters = {
        "instrument": instrument,
        "dp_cat": "CALIB",
        "stime": str(d - day)[:10],
        "etime": str(d + day)[:10],
    }
    table = eso.query_main(column_filters=filters)
    files = table["Dataset ID"]
    eso.retrieve_data(files, destination=destination, continuation=True)
