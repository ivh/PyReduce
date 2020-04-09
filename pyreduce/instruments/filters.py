import numpy as np
from fnmatch import fnmatch
import re
from datetime import datetime, date
from dateutil import parser
import logging
from astropy.time import Time
from astropy import units as u

logger = logging.getLogger(__name__)


class Filter:
    def __init__(self, keyword, dtype="U20", wildcards=False, regex=False, flags=0):
        self.keyword = keyword
        self.dtype = dtype
        self.wildcards = wildcards
        self.regex = regex
        self.flags = flags
        self.data = []

    def collect(self, header):
        value = header.get(self.keyword)
        self.data.append(value)
        return value

    def match(self, value):
        # result = np.full(len(self.data), False)
        if self.regex:
            regex = re.compile(f"^(?:{value})$", flags=self.flags)
            match = [regex.match(f) is not None for f in self.data]
            result = np.asarray(match, dtype=bool)
        elif self.wildcards:
            match = [fnmatch(f, value) for f in self.data]
            result = np.asarray(match, dtype=bool)
        else:
            match = np.asarray(self.data) == value
            result = match
        return result

    def classify(self, value):
        if value is not None:
            match = self.match(value)
            data = np.asarray(self.data)
            data = np.unique(data[match])
        else:
            data = np.unique(self.data)
        data = [(d, self.match(d)) for d in data]
        return data

    def clear(self):
        self.data = []


class InstrumentFilter(Filter):
    def __init__(self, keyword="INSTRUME", **kwargs):
        super().__init__(keyword, dtype="U20", **kwargs)


class ObjectFilter(Filter):
    def __init__(self, keyword="OBJECT", **kwargs):
        super().__init__(keyword, dtype="U20", **kwargs)


class NightFilter(Filter):
    def __init__(self, keyword="DATE-OBS", timeformat="fits", **kwargs):
        super().__init__(keyword, dtype=datetime, **kwargs)
        self.timeformat = timeformat

    @staticmethod
    def observation_date_to_night(observation_date):
        """Convert an observation timestamp into the date of the observation night
        Nights start at 12am and end at 12 am the next day
        """
        if observation_date.to_datetime().hour < 12:
            observation_date -= 1 * u.day
        return observation_date.to_datetime().date()

    def collect(self, header):
        value = header.get(self.keyword)
        value = Time(value, format=self.timeformat)
        value = NightFilter.observation_date_to_night(value)
        self.data.append(value)
        return value

    def match(self, value):
        try:
            value = parser.parse(value).date()
        except Exception:
            pass
        match = super().match(value)
        return match
