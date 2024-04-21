import blpapi

TRADING_DAYS = 252
BBG_HOST = "localhost"
BBG_PORT = 8194
MONTH_BBG_MAP = {v: k for k, v in zip("FGHJKMNQUVXZ", range(1, 13))}
EVENT_DICT = {
    blpapi.Event.ADMIN: "blpapi.Event.ADMIN",
    blpapi.Event.AUTHORIZATION_STATUS: "blpapi.Event.AUTHORIZATION_STATUS",
    blpapi.Event.PARTIAL_RESPONSE: "blpapi.Event.PARTIAL_RESPONSE",
    blpapi.Event.REQUEST: "blpapi.Event.REQUEST",
    blpapi.Event.REQUEST_STATUS: "blpapi.Event.REQUEST_STATUS",
    blpapi.Event.RESOLUTION_STATUS: "blpapi.Event.RESOLUTION_STATUS",
    blpapi.Event.RESPONSE: "blpapi.Event.RESPONSE",
    blpapi.Event.SERVICE_STATUS: "blpapi.Event.SERVICE_STATUS",
    blpapi.Event.SESSION_STATUS: "blpapi.Event.SESSION_STATUS",
    blpapi.Event.SUBSCRIPTION_DATA: "blpapi.Event.SUBSCRIPTION_DATA",
    blpapi.Event.SUBSCRIPTION_STATUS: "blpapi.Event.SUBSCRIPTION_STATUS",
    blpapi.Event.TIMEOUT: "blpapi.Event.TIMEOUT",
    blpapi.Event.TOKEN_STATUS: "blpapi.Event.TOKEN_STATUS",
    blpapi.Event.TOPIC_STATUS: "blpapi.Event.TOPIC_STATUS",
    blpapi.Event.UNKNOWN: "blpapi.Event.UNKNOWN",
}

AUTHORIZATION_SUCCESS = blpapi.Name("AuthorizationSuccess")
AUTHORIZATION_FAILURE = blpapi.Name("AuthorizationFailure")
TOKEN_SUCCESS = blpapi.Name("TokenGenerationSuccess")
TOKEN_FAILURE = blpapi.Name("TokenGenerationFailure")

SERVICES = {
    "HistoricalDataRequest": "//blp/refdata",
    "ReferenceDataRequest": "//blp/refdata",
    "IntradayTickRequest": "//blp/refdata",
    "IntradayBarRequest": "//blp/refdata",
    "BeqsRequest": "//blp/refdata",
    "FieldInfoRequest": "//blp/apiflds",
    "FieldListRequest": "//blp/apiflds",
    "instrumentListRequest": "//blp/instruments",
    "GetFills": "//blp/emsx.history",
    "sendQuery": "//blp/bqlsvc",
}
