# -*- coding: utf-8 -*-
import logging
import re
from datetime import datetime
from fnmatch import fnmatch

import numpy as np
from astropy import units as u
from astropy.time import Time
from dateutil import parser

logger = logging.getLogger(__name__)


class Filter:
    def __init__(
        self,
        keyword,
        dtype="U20",
        wildcards=False,
        regex=False,
        flags=0,
        unique=True,
        ignorecase=True,
    ):
        self.keyword = keyword
        self.dtype = dtype
        self.wildcards = wildcards
        self.regex = regex
        self.flags = flags
        self.data = []
        self.unique = unique
        self.ignorecase = ignorecase

        if self.ignorecase and not self.flags & re.IGNORECASE:
            self.flags += re.IGNORECASE

    def _collect_value(self, header):
        if self.keyword is None:
            value = ""
        elif "{" in self.keyword:
            kws = re.findall(r"{([^{}]+)}", self.keyword)
            values = {kw: header.get(kw, "") for kw in kws}
            value = self.keyword.format(**values)
        else:
            value = header.get(self.keyword)
        if value.__class__ == header.__class__:
            if len(value) > 0:
                value = value[0]
            else:
                value = ""
        return value

    def collect(self, header):
        value = self._collect_value(header)
        self.data.append(value)
        return value

    def match(self, value):
        if self.keyword is None:
            result = np.full(len(self.data), False)
        else:
            try:
                if self.regex:
                    regex = re.compile(f"^(?:{value})$", flags=self.flags)
                elif self.wildcards:
                    regex = re.compile(fnmatch.translate(value), flags=self.flags)
                else:
                    regex = re.compile(value, flags=self.flags)

                result = [
                    regex.match(f) is not None if f is not None else False
                    for f in self.data
                ]
            except TypeError as ex:
                result = [f == value for f in self.data]
        result = np.asarray(result, dtype=bool)
        return result

    def classify(self, value):
        if self.unique:
            if value is not None and value != "":
                match = self.match(value)
                data = np.asarray(self.data)
                data = np.unique(data[match])
            else:
                data = set(self.data)
            data = [(d, self.match(d)) for d in data]
        else:
            if value is not None and value != "":
                match = self.match(value)
            else:
                match = np.full(len(self.data), True)
            data = [(value, match)]
        return data

    def clear(self):
        self.data = []


class InstrumentFilter(Filter):
    def __init__(self, keyword="INSTRUME", **kwargs):
        kwargs["dtype"] = "U20"
        kwargs["unique"] = False
        super().__init__(keyword, **kwargs)


class ObjectFilter(Filter):
    def __init__(self, keyword="OBJECT", **kwargs):
        kwargs["dtype"] = "U20"
        kwargs["unique"] = False
        super().__init__(keyword, **kwargs)


class NightFilter(Filter):
    def __init__(
        self,
        keyword="DATE-OBS",
        timeformat="fits",
        timezone="utc",
        timezone_local=None,
        **kwargs,
    ):
        super().__init__(keyword, dtype=datetime, **kwargs)
        self.timeformat = timeformat
        self.timezone = timezone
        self.timezone_local = timezone_local

    @staticmethod
    def observation_date_to_night(observation_date):
        """Convert an observation timestamp into the date of the observation night
        Nights start at 12am and end at 12 am the next day
        """
        if observation_date.to_datetime().hour < 12:
            observation_date -= 1 * u.day
        return observation_date.to_datetime().date()

    def collect(self, header):
        value = super()._collect_value(header)
        if value is not None:
            try:
                value = Time(value, format=self.timeformat, scale=self.timezone)
                value = self.observation_date_to_night(value)
            except ValueError:
                logger.warning(
                    "Could not determine the observation date of %s, skipping it",
                    header,
                )
        else:
            logger.warning(
                "Could not determine the observation date of %s, skipping it", header
            )
        self.data.append(value)
        return value

    def match(self, value):
        try:
            value = parser.parse(value).date()
        except Exception:
            pass
        match = super().match(value)
        return match


class ModeFilter(Filter):
    def __init__(
        self,
        keyword,
        dtype="U20",
        wildcards=False,
        regex=False,
        flags=0,
        unique=True,
        ignorecase=True,
        replacement={},
    ):
        super().__init__(
            keyword,
            dtype=dtype,
            wildcards=wildcards,
            regex=regex,
            flags=flags,
            unique=unique,
            ignorecase=ignorecase,
        )
        self.replacement = replacement

    def classify(self, value):
        data = super().classify(value)
        data = [
            (self.replacement[d] if d in self.replacement.keys() else d, m)
            for d, m in data
        ]
        return data
