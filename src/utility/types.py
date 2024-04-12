from dataclasses import dataclass
from datetime import datetime
from strenum import StrEnum


@dataclass
class SpinOff:
    parent_company: str
    subsidiary_company: str
    spin_off_ex_date: datetime


class RebalanceFrequencyEnum(StrEnum):
    DAILY = "1B"
    WEEKLY = "1W-FRI"
    MONTH_END = "BME"
    MONTH_START = "BMS"
    QUARTER_END = "BQE"
    QUARTER_START = "BQS"

    @classmethod
    def list_values(cls):
        return list(map(lambda c: c.value, cls))

    @classmethod
    def list_names(cls):
        return list(map(lambda c: c.name, cls))


class AllocationMethodsEnum(StrEnum):
    EQUALLY_WEIGHTED = "EQUALLY_WEIGHTED"
    MAX_SHARPE = "MAX_SHARPE"
    RISK_PARITY = "RISK_PARITY"

    @classmethod
    def list_values(cls):
        return list(map(lambda c: c.value, cls))

    @classmethod
    def list_names(cls):
        return list(map(lambda c: c.name, cls))
