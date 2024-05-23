# inspired version of the code from : https://github.com/matthewgilbert/blp/tree/master
from typing import (
    Callable,
    Dict,
    Generator,
    Iterable,
    List,
    Optional,
    Sequence,
    Union,
)
import pandas as pd
import itertools
import json
import logging
import blpapi


from utility.constants import (
    BBG_HOST,
    BBG_PORT,
    EVENT_DICT,
    AUTHORIZATION_SUCCESS,
    AUTHORIZATION_FAILURE,
    TOKEN_SUCCESS,
    TOKEN_FAILURE,
    SERVICES,
)
from utility.utils import (
    message_to_dict,
    dict_to_req,
)

logger = logging.getLogger(__name__)


class BlpSession:
    def __init__(self, host: str, port: int, **kwargs):
        """Manage a Bloomberg session.

        A BlpSession is used for managing the lifecycle of a connection to a blpapi.Session. This includes managing
        session options, event handlers and authentication.

        Args:
            host: Host to connect session on
            port: Port to connect session to
            **kwargs: Keyword arguments used in blpapi.SessionOptions

        """
        self.session_options = self.create_session_options(host, port, **kwargs)
        self.session = blpapi.Session(options=self.session_options)
        self.identity = None

    def __repr__(self):
        host, port = (
            self.session_options.serverHost(),
            self.session_options.serverPort(),
        )
        return "{} with <address={}:{}><identity={!r}>".format(
            type(self), host, port, self.identity
        )

    @staticmethod
    def create_session_options(host: str, port: int, **kwargs) -> blpapi.SessionOptions:
        """Create blpapi.SessionOptions class used in blpapi.Session.

        Args:
            host: Host to connect session on
            port: Port to connection session to
            **kwargs: Keyword args passed to the blpapi.SessionOpts, if authentication is needed use
                setAuthenticationOptions

        Returns: A blpapi.SessionOptions

        """
        session_options = blpapi.SessionOptions()
        kwargs["setServerHost"] = host
        kwargs["setServerPort"] = port

        # logging and subscription logic does not currently support multiple correlationIds
        kwargs["setAllowMultipleCorrelatorsPerMsg"] = False
        kwargs.setdefault("setAutoRestartOnDisconnection", True)
        kwargs.setdefault("setNumStartAttempts", 1)
        kwargs.setdefault("setRecordSubscriptionDataReceiveTimes", True)
        for key in kwargs:
            getattr(session_options, key)(kwargs[key])
        return session_options

    def authenticate(self, timeout: int = 0) -> None:
        """Authenticate the blpapi.Session.

        Args:
            timeout: Milliseconds to wait for service before the blpapi.EventQueue returns a blpapi.Event.TIMEOUT

        """
        token_event_queue = blpapi.EventQueue()
        self.session.generateToken(eventQueue=token_event_queue)

        event = token_event_queue.nextEvent(timeout)
        for n, msg in enumerate(event):
            if msg.messageType() == TOKEN_SUCCESS:
                logger.info(f"TOKEN_STATUS - Message {n} - {msg}")
                auth_service = self.session.getService("//blp/apiauth")
                auth_request = auth_service.createAuthorizationRequest()
                auth_request.set("token", msg.getElementAsString("token"))
                identity = self.session.createIdentity()
                logger.info(f"Send authorization request\n{auth_request}")
                self.session.sendAuthorizationRequest(auth_request, identity)
            elif msg.messageType() == TOKEN_FAILURE:
                raise ConnectionError(f"TOKEN_STATUS - Message {n} - {msg}")

        event = token_event_queue.nextEvent(timeout)
        for n, msg in enumerate(event):
            if msg.messageType() == AUTHORIZATION_FAILURE:
                raise ConnectionError(f"RESPONSE - Message {n} - {msg}")
            elif msg.messageType() == AUTHORIZATION_SUCCESS:
                logger.info(f"RESPONSE - Message {n} - {msg}")
                self.identity = identity


class BlpQuery(BlpSession):
    def __init__(
        self,
        **kwargs,
    ):
        """A class to manage a synchronous Bloomberg request/response session.

        Args:
            **kwargs: Keyword arguments used in blpapi.SessionOptions

        Examples:
            >>> BlpQuery()
            <class 'blp.blp.BlpQuery'> with <address=localhost:8194><identity=None><eventHandler=None>

        """
        self.parser = BlpParser()
        self._field_column_map: Dict = {}
        self._started = False
        self._services: Dict[str, blpapi.Service] = {}
        self.timeout = 100000
        super().__init__(BBG_HOST, BBG_PORT, **kwargs)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def start(self):
        """Start the blpapi.Session and open relevant services."""
        if not self._started:
            self._started = self.session.start()
            if not self._started:
                msg = next(iter(self.session.tryNextEvent()))
                logger.info(f"Failed to connect to Bloomberg:\n{msg}")
                raise ConnectionError(f"Failed to start {self!r}")
            logger.debug(f"Started {self!r}")
            logger.debug(
                f"{next(iter(self.session.tryNextEvent()))}{next(iter(self.session.tryNextEvent()))}"
            )
        for service in set(SERVICES.values()):
            if not self.session.openService(service):
                raise ConnectionError(f"Unknown service {service!r}")
            logger.debug(f"Service {service!r} opened")
            logger.debug(f"{next(iter(self.session.tryNextEvent()))}")
            self._services[service] = self.session.getService(service)
        return self

    def stop(self):
        self.session.stop()

    def __del__(self):
        self.stop()

    @staticmethod
    def _pass_through(x, _):
        yield x

    def query(
        self,
        request_data: Dict,
        parse: Optional[Callable] = None,
        collector: Optional[Callable] = None,
    ):
        """Request and parse Bloomberg data.

        Args:
            request_data: A dictionary representing a blpapi.Request, specifying both the service and the data
            parse: Callable which takes a dictionary response and request and yields 0 or more values. If None, use
              default parser. If False, do not parse the response
            collector: Callable which takes an iterable

        Returns: A result from collector, if collector=None default is a itertools.chain

        Examples:
            >>> bq = BlpQuery().start() # doctest: +SKIP

            A historical data request collected into a list

            >>> rd = {
            ...  'HistoricalDataRequest': {
            ...    'securities': ['CL1 Comdty'],
            ...    'fields': ['PX_LAST'],
            ...    'startDate': '20190102',
            ...    'endDate': '20190102'
            ...   }
            ... }
            >>> bq.query(rd, collector=list) # doctest: +SKIP
            [
             {
              'security':'CL1 Comdty',
              'fields':['PX_LAST'],
              'data':[{'date': Timestamp('2019-01-02 00:00:00'), 'PX_LAST':46.54}]
             }
            ]

            A historical data request with no parsing collected into a list

            >>> bq.query(rd, collector=list, parse=False) # doctest: +SKIP
            [
               {
                  'eventType':5,
                  'eventTypeName':'blpapi.Event.RESPONSE',
                  'messageNumber':0,
                  'message':{
                     'fragmentType':0,
                     'correlationIds':[8],
                     'messageType':'HistoricalDataResponse',
                     'topicName':'',
                     'timeReceived':None,
                     'element':{
                        'HistoricalDataResponse':{
                           'securityData':{
                              'security':'CL1 Comdty',
                              'eidData':[],
                              'sequenceNumber':0,
                              'fieldExceptions':[],
                              'fieldData':[{'fieldData':{'date': Timestamp('2019-01-02 00:00:00'),'PX_LAST':46.54}}]
                           }
                        }
                     }
                  }
               }
            ]

        """

        if parse is False:
            parse = self._pass_through
        elif parse is None:
            parse = self.parser
        data_queue = blpapi.EventQueue()
        request = self.create_request(request_data)
        self.send_request(request, data_queue)
        res = (
            parse(data, request_data)
            for data in self.get_response(data_queue, self.timeout)
        )
        res = itertools.chain.from_iterable(res)  # type: ignore
        if collector:
            res = collector(res)
        return res

    def create_request(self, request_data: Dict) -> blpapi.Request:
        """Create a blpapi.Request.

        Args:
            request_data: A dictionary representing a blpapi.Request, specifying both the service and the data

        Returns: blpapi.Request

        Examples:
            >>> bq = BlpQuery().start() # doctest: +SKIP
            >>> bq.create_request({
            ...  'HistoricalDataRequest': {
            ...    'securities': ['CL1 Comdty'],
            ...    'fields': ['PX_LAST'],
            ...    'startDate': '20190102',
            ...    'endDate': '20190102'
            ...   }
            ... }) # doctest: +SKIP

        """
        operation = list(request_data.keys())[0]
        service = self._services[SERVICES[operation]]
        request = service.createRequest(operation)
        rdata = request_data[operation]
        request = dict_to_req(request, rdata)
        return request

    def send_request(
        self,
        request: blpapi.Request,
        data_queue: blpapi.EventQueue,
        correlation_id: Optional[blpapi.CorrelationId] = None,
    ) -> blpapi.CorrelationId:
        """Send a request who's data will be populated into data_queue.

        Args:
            request: Request to send
            data_queue: Queue which response populates
            correlation_id: Id associated with request/response

        Returns: blpapi.CorrelationId associated with the request

        """
        logger.debug(
            f"Sent {request} with identity={self.identity!r}, correlationId={correlation_id!r}, event_queue={data_queue!r}",  # noqa: E501
        )
        cid = self.session.sendRequest(
            request, self.identity, correlation_id, data_queue
        )
        return cid

    def get_response(
        self, data_queue: blpapi.EventQueue, timeout: Optional[int] = None
    ) -> Generator:
        """Yield dictionary representation of blpapi.Messages from a blpapi.EventQueue.

        Args:
            data_queue: Queue which contains response
            timeout: Milliseconds to wait for service before the blpapi.EventQueue returns a blpapi.Event.TIMEOUT

        Returns: A generator of messages translated into a dictionary representation

        """
        if timeout is None:
            timeout = self.timeout
        while True:
            event = data_queue.nextEvent(timeout=timeout)
            event_type = event.eventType()
            event_type_name = EVENT_DICT[event_type]
            if event_type == blpapi.Event.TIMEOUT:
                raise ConnectionError(
                    f"Unexpected blpapi.Event.TIMEOUT received by {self!r}"
                )
            for n, msg in enumerate(event):
                logger.debug(f"Message {n} in {event_type_name}:{msg}")
                response = {
                    "eventType": event_type,
                    "eventTypeName": event_type_name,
                    "messageNumber": n,
                    "message": message_to_dict(msg),
                }
                yield response
            if event_type == blpapi.Event.RESPONSE:
                return

    def cast_columns(self, df: pd.DataFrame, fields: Iterable) -> pd.DataFrame:
        res = {}
        for field in fields:
            col_data = df.get(field)
            if field in self._field_column_map:
                col = self._field_column_map[field]
                col_data = col(col_data)
            res[field] = col_data
        # handle the case where all values are None
        try:
            return pd.DataFrame(res)
        except ValueError as e:
            if df.empty:
                return pd.DataFrame(columns=res.keys())
            else:
                raise e

    def bdh(
        self,
        securities: Sequence[str],
        fields: List[str],
        start_date: str,
        end_date: str,
        overrides: Optional[Sequence] = None,
        options: Optional[Dict] = None,
    ) -> pd.DataFrame:
        """Bloomberg historical data request.

        Args:
            securities: list of strings of securities
            fields: list of strings of fields
            start_date: start date as '%Y%m%d'
            end_date: end date as '%Y%m%d'
            overrides: List of tuples containing the field to override and its value
            options: key value pairs to to set in request

        Returns: A pd.DataFrame with columns ['date', 'security', fields[0], ...]

        """
        query = create_historical_query(
            securities, fields, start_date, end_date, overrides, options
        )
        res = self.query(query, self.parser, self.collect_to_bdh)
        dfs = []
        for sec in res:
            dfs.append(res[sec].assign(security=sec))
        df = (
            pd.concat(dfs)
            .sort_values(by="date", axis=0)
            .loc[:, ["date", "security"] + fields]
            .reset_index(drop=True)
        )
        return df

    def collect_to_bdh(self, responses: Iterable) -> Dict[str, pd.DataFrame]:
        """Collector for bdh()."""
        dfs: Dict = {}
        for response in responses:
            security = response["security"]
            fields = response["fields"] + ["date"]
            # have not seen example where a HistoricalDataResponse for a single security is broken across
            # multiple PARTIAL_RESONSE/RESPONSE but API docs are vague about whether technically possible
            sec_dfs = dfs.get(security, [])
            df = pd.DataFrame(response["data"])
            df = self.cast_columns(df, fields)
            sec_dfs.append(df)
            dfs[security] = sec_dfs
        for sec in dfs:
            df_list = dfs[sec]
            if len(df_list) > 1:
                dfs[sec] = pd.concat(df_list).sort_values(
                    by="date", axis=0, ignore_index=True
                )
            else:
                dfs[sec] = df_list[0]

        return dfs

    def bql(
        self,
        expression: str,
        overrides: Optional[Sequence] = None,
        options: Optional[Dict] = None,
    ) -> pd.DataFrame:
        """Bloomberg query language request.

        Args:
            expression: BQL expression
            overrides: List of tuples containing the field to override and its value
            options: key value pairs to to set in request

        Returns: A pd.DataFrame with columns ["security", "field", "secondary_name", "secondary_value", "value"]

        Examples:
            >>> bquery = blp.BlpQuery().start() # doctest: +SKIP
            >>> bquery.bql(expression="get(px_last()) for(['AAPL US Equity', 'IBM US Equity'])") # doctest: +SKIP

            The resulting DataFrame will look like this:
                     security      field  secondary_name       secondary_value        value
            0  AAPL US Equity  px_last()        CURRENCY                   USD   192.755005
            1  IBM US Equity   px_last()        CURRENCY                   USD   139.289993
            2  AAPL US Equity  px_last()            DATE  2023-07-24T00:00:00Z   192.755005
            3  IBM US Equity   px_last()            DATE  2023-07-24T00:00:00Z   139.289993
        """
        query = create_bql_query(expression, overrides, options)

        bql_parser = BlpParser(
            processor_steps=[
                BlpParser._clean_bql_response,
                BlpParser._validate_event,
                BlpParser._validate_response_error,
            ]
        )
        df = self.query(query, bql_parser, self.collect_to_bql)
        df = df[["id", "field", "secondary_name", "secondary_value", "value"]]
        df = df.rename(columns={"id": "security"})
        return df

    def collect_to_bql(self, responses: Iterable) -> pd.DataFrame:
        """Collector for bql()."""
        data = []
        fields = {"secondary_name", "secondary_value", "field", "id", "value"}
        for field in responses:
            field_df = pd.DataFrame(field)

            id_vars = ["field", "id", "value"]
            secondary_columns = field_df.columns.difference(id_vars)

            if len(secondary_columns) == 0:
                # If we dont have any secondary columns, we just add empty columns
                field_df["secondary_name"] = None
                field_df["secondary_value"] = None
            else:
                # If we have multiple secondary columns, we need to melt the dataframe
                field_df = field_df.melt(
                    id_vars=id_vars,
                    value_vars=field_df.columns.difference(id_vars),
                    var_name="secondary_name",
                    value_name="secondary_value",
                )

            column_order = ["secondary_name", "secondary_value", "field", "id", "value"]
            field_df = field_df[column_order]

            data.append(field_df)

        df = pd.concat(data)
        return self.cast_columns(df, fields)

    def bdp(
        self,
        securities: Sequence[str],
        fields: List[str],
        overrides: Optional[Sequence] = None,
        options: Optional[Dict] = None,
    ) -> pd.DataFrame:
        """Bloomberg reference data point request.

        Args:
            securities: list of strings of securities
            fields: list of strings of fields
            overrides: list of tuples containing the field to override and its value
            options: key value pairs to to set in request

        Returns: A pd.DataFrame where columns are ['security', field[0], ...]

        """
        query = create_reference_query(securities, fields, overrides, options)
        df = self.query(query, self.parser, self.collect_to_bdp)
        df = df.loc[:, ["security"] + fields]
        return df

    def collect_to_bdp(self, responses: Iterable) -> pd.DataFrame:
        """Collector for bdp()."""
        rows = []
        fields = {"security"}
        for response in responses:
            data = response["data"]
            # possible some fields are missing for different securities
            fields = fields.union(set(response["fields"]))
            for _, value in data.items():
                if isinstance(value, list):
                    raise TypeError(
                        f"Bulk reference data not supported, expected singleton values but received {data}"
                    )
            data["security"] = response["security"]
            rows.append(data)
        df = pd.DataFrame(rows)
        return self.cast_columns(df, fields)

    def bds(
        self,
        security: str,
        field: str,
        overrides: Optional[Sequence] = None,
        options: Optional[Dict] = None,
    ) -> pd.DataFrame:
        """Bloomberg reference data set request.

        Args:
            security: String representing security
            field: String representing field
            overrides: List of tuples containing the field to override and its value
            options: key value pairs to to set in request

        Returns: A pd.DataFrame where columns are data element names

        """
        query = create_reference_query(security, field, overrides, options)
        return self.query(query, self.parser, self.collect_to_bds)

    def collect_to_bds(self, responses: Iterable) -> pd.DataFrame:
        """Collector for bds()."""
        rows = []
        field = None
        for response in responses:
            keys = list(response["data"].keys())
            if len(keys) > 1:
                raise ValueError(f"responses must have only one field, received {keys}")
            if field is not None and field != keys[0]:
                raise ValueError(
                    f"responses contain different fields, {field} and {keys[0]}"
                )
            field = keys[0]
            data = response["data"][field]
            try:
                rows.extend(data)
            except TypeError:
                raise TypeError(
                    f"response data must be bulk reference data, received {response['data']}"
                )
        df = pd.DataFrame(rows)
        return self.cast_columns(df, df.columns)

    def collect_many_to_bds(self, responses) -> Dict:
        """Collector to nested dictionary of DataFrames.

        Top level keys are securities, next level keys are fields and values are DataFrame in bds() form

        """
        res: Dict = {}
        for response in responses:
            security = response["security"]
            sec_dict = res.get(security, {})
            for field in response["data"]:
                data = response["data"][field]
                if data:
                    rows = sec_dict.get(field, [])
                    rows.extend(data)
                    sec_dict[field] = rows
            res[security] = sec_dict
        for s in res:
            for f in res[s]:
                # what does res[s][f] look like? can it be passed to to_series directly?
                df = pd.DataFrame(res[s][f])
                res[s][f] = self.cast_columns(df, df.columns)
        return res


class BlpParser:
    """A callable class with a default response parsing implementation.

    The parse method parses the responses from BlpQuery.get_response into a simplified representation the can easily
    be collected using collectors in BlpQuery.

    Args:
        processor_steps: A list of processors which take in a response and request_data and returns a
          validated and possibly modified response. Processors are called sequentially at the start of parse()
        raise_security_errors: If True, raise errors when response contains an INVALID_SECURITY error, otherwise
          log as a warning. This is ignored if ``processor_steps`` is not None.

    """

    def __init__(
        self,
        processor_steps: Optional[Sequence] = None,
        raise_security_errors: bool = True,
    ):
        if processor_steps is None and raise_security_errors:
            processor_steps = [
                self._validate_event,
                self._validate_response_type,
                self._validate_response_error,
                self._validate_security_error,
                self._process_field_exception,
            ]
        elif processor_steps is None:
            processor_steps = [
                self._validate_event,
                self._validate_response_type,
                self._validate_response_error,
                self._warn_security_error,
                self._process_field_exception,
            ]
        self._processor_steps = processor_steps

    @staticmethod
    def _clean_bql_response(response, _):
        """
        The purpose of this method is to standardize a BQL (Bloomberg Query Language) response.
        BQL responses differ from standard responses, hence the need for cleanup to make them more consistent.
        """
        aux = json.loads(response["message"]["element"])

        if aux["responseExceptions"]:
            aux["responseError"] = aux["responseExceptions"][0]["message"]
            del aux["responseExceptions"]

        response["message"]["element"] = {"BQLResponse": aux}

        return response

    @staticmethod
    def _validate_event(response, _):
        if response["eventType"] not in (
            blpapi.Event.PARTIAL_RESPONSE,
            blpapi.Event.RESPONSE,
        ):
            raise TypeError(f"Unknown eventType: {response}")
        return response

    @staticmethod
    def _validate_response_type(response, _):
        rtype = list(response["message"]["element"].keys())[0]
        known_responses = (
            "ReferenceDataResponse",
            "HistoricalDataResponse",
            "IntradayBarResponse",
            "BeqsResponse",
            "IntradayTickResponse",
            "fieldResponse",
            "InstrumentListResponse",
            "GetFillsResponse",
        )
        if rtype not in known_responses:
            raise TypeError(f"Unknown {rtype!r}, must be in {known_responses}")
        return response

    @staticmethod
    def _validate_response_error(response, request):
        rtype = list(response["message"]["element"].keys())[0]
        if "responseError" in response["message"]["element"][rtype]:
            raise TypeError(
                f"Response contains responseError\nresponse: {response}\nrequest: {request}"
            )
        return response

    @staticmethod
    def _process_field_exception(response, _):
        rtype = list(response["message"]["element"].keys())[0]
        response_data = response["message"]["element"][rtype]
        if rtype in (
            "IntradayBarResponse",
            "IntradayTickResponse",
            "BeqsResponse",
            "fieldResponse",
            "InstrumentListResponse",
            "GetFillsResponse",
        ):
            return response
        if rtype == "HistoricalDataResponse":
            response_data = [response_data]
        for sec_data in response_data:
            field_exceptions = sec_data["securityData"]["fieldExceptions"]
            for fe in field_exceptions:
                fe = fe["fieldExceptions"]
                einfo = fe["errorInfo"]["errorInfo"]
                if (
                    einfo["category"] == "BAD_FLD"
                    and einfo["subcategory"] == "NOT_APPLICABLE_TO_REF_DATA"
                ):
                    field = fe["fieldId"]
                    sec_data["securityData"]["fieldData"]["fieldData"][field] = None
                else:
                    raise TypeError(
                        f"Response for {sec_data['securityData']['security']} contains fieldException {fe}"
                    )
        return response

    @staticmethod
    def _validate_fields_exist(response, request_data):
        rtype = list(response["message"]["element"].keys())[0]
        if rtype != "HistoricalDataResponse":
            return response

        fields = set(request_data["HistoricalDataRequest"]["fields"])
        sec_data = response["message"]["element"]["HistoricalDataResponse"][
            "securityData"
        ]
        if not sec_data["fieldData"] and fields:
            raise TypeError(
                f"fieldData for {sec_data['security']!r} is missing fields {fields!r}"
            )
        for fd in sec_data["fieldData"]:
            fd = fd["fieldData"]
            diff = fields.difference(fd.keys())
            if diff:
                raise TypeError(
                    f"fieldData for {sec_data['security']!r} is missing fields {diff!r} in {fd!r}"
                )

    @staticmethod
    def _validate_security_error(response, _):
        rtype = list(response["message"]["element"].keys())[0]
        response_data = response["message"]["element"][rtype]
        if rtype in (
            "IntradayBarResponse",
            "IntradayTickResponse",
            "BeqsResponse",
            "fieldResponse",
            "InstrumentListResponse",
            "GetFillsResponse",
        ):
            return response
        if rtype == "HistoricalDataResponse":
            response_data = [response_data]
        for sec_data in response_data:
            data = sec_data["securityData"]
            if "securityError" in data:
                raise TypeError(
                    f"Response for {data['security']!r} contains securityError {data['securityError']}"
                )
        return response

    @staticmethod
    def _warn_security_error(response, _):
        rtype = list(response["message"]["element"].keys())[0]
        response_data = response["message"]["element"][rtype]
        if rtype in (
            "IntradayBarResponse",
            "IntradayTickResponse",
            "BeqsResponse",
            "fieldResponse",
            "InstrumentListResponse",
            "GetFillsResponse",
        ):
            return response
        if rtype == "HistoricalDataResponse":
            response_data = [response_data]
        for sec_data in response_data:
            data = sec_data["securityData"]
            if "securityError" in data:
                logger.warning(
                    f"Response for {data['security']!r} contains securityError {data['securityError']}"
                )
        return response

    def __call__(self, response, request_data):
        """A default parser to parse dictionary representation of response.

        Parses data response to a generator of dictionaries or raises a TypeError if the response type is unknown.
        There is support for ReferenceDataResponse, HistoricalDataResponse, IntradayBarResponse, IntradayTickResponse,
        fieldResponse, InstrumentListResponse and GetFillsResponse. Parsed dictionaries have the following forms:

        .. code-block:: text

            1. ReferenceDataResponse
                Schema: {'security': <str>, 'fields': <list of str>, 'data': <dict of field:value>}
                Examples:
                    {'security': 'SPY US Equity', 'fields': ['NAME'], 'data': {'NAME': 'SPDR S&P 500 ETF TRUST'}}
                    {
                        'security': 'C 1 Comdty',
                        'fields': ['FUT_CHAIN'],
                        'data': {'FUT_CHAIN': [
                            {'Security Description': 'C H10 Comdty'},
                            {'Security Description': 'C K10 Comdty'}
                        ]}
                    }

            2. HistoricalDataResponse
              Schema: {'security': <str>, 'fields': <list of str>, 'data': <list of dict of field:value>}
              Examples:
                  {
                    'security': 'SPY US Equity',
                    'fields': ['PX_LAST'],
                    'data': [
                      {'date': pd.Timestamp(2018, 1, 2), 'PX_LAST': 268.77},
                      {'date': pd.Timestamp(2018, 1, 3), 'PX_LAST': 270.47}
                    ]
                  }

            3. IntradayBarResponse
              Schema: {'security': <str>, 'events': [<str>],
                       'data': <list of {'time': <pd.Timestamp>, 'open': <float>, 'high': <float>, 'low': <float>,
                                         'close': <float>, 'volume': <int>, 'numEvents': <int>, 'value': <float>}}
                      }
              Examples:
                  {
                    'security': 'CL1 Comdty',
                    'data': [{'time': pd.Timestamp('2019-04-24 08:00:00'), 'open': 65.85, 'high': 65.89,
                              'low': 65.85, 'close': 65.86, 'volume': 565, 'numEvents': 209, 'value': 37215.16}],
                    'events': ['TRADE']
                  }

            4. IntradayTickResponse
              Schema: {'security': <str>, 'events': <list of str>,
                       'data': <list of  {'time': <pd.Timestamp>, 'type': <str>, 'value': <float>, 'size': <int>}>}
              Examples:
                  {
                     'security': 'CL1 Comdty',
                     'data': [
                       {'time': pd.Timestamp('2019-04-24 08:00:00'), 'type': 'BID', 'value': 65.85, 'size': 4},
                       {'time': pd.Timestamp('2019-04-24 08:00:00'), 'type': 'BID', 'value': 65.85, 'size': 41},
                       {'time': pd.Timestamp('2019-04-24 08:00:00'), 'type': 'ASK', 'value': 65.86, 'size': 50},
                     ],
                     'events': ['BID', 'ASK']
                  }

            5. fieldResponse
              Schema: {'id': <list of str>, data: {<str>: {field: value}}}
              Examples:
                  {
                    'id': ['PX_LAST', 'NAME'],
                    'data': {
                      'DS002': {
                        'mnemonic': 'NAME',
                        'description': 'Name',
                        'datatype': 'String',
                        'categoryName': [],
                        'property': [],
                        'overrides': [],
                        'ftype': 'Character'
                      },
                     'PR005': {
                       'mnemonic': 'PX_LAST',
                       'description': 'Last Price',
                       'datatype': 'Double',
                       'categoryName': [],
                       'property': [],
                       'overrides': ['PX628', 'DY628',...]
                       'ftype': 'Price'
                      }
                    }
                  }

            6. InstrumentListResponse
            Schema: {'security': <str>, 'description': <str>}
            Examples:
                {
                   'security': 'T<govt>,
                   'description': 'United States Treasury Note/Bond (Multiple Matches)'
                }

            7. GetFillsResponse
            Schema: {'Fills': <dict>}
            Examples:
                {
                    'fills': [
                        {'Ticker': 'GCZ9', 'Exchange': 'CMX', 'Type': 'MKT', ...},
                        {'Ticker': 'SIZ9', 'Exchange': 'CMX', 'Type': 'LMT', ...}
                    ]
                }

        Args:
            response (dict): Representation of a blpapi.Message
            request_data (dict): A dictionary representing a blpapi.Request

        Returns: A generator of responses parsed to dictionaries

        """
        for processor in self._processor_steps:
            response = processor(response, request_data)

        rtype = list(response["message"]["element"].keys())[0]
        if rtype == "ReferenceDataResponse":
            sec_data_parser = self._parse_reference_security_data
        elif rtype == "HistoricalDataResponse":
            sec_data_parser = self._parse_historical_security_data
        elif rtype == "IntradayBarResponse":
            sec_data_parser = self._parse_bar_security_data
        elif rtype == "IntradayTickResponse":
            sec_data_parser = self._parse_tick_security_data
        elif rtype == "BeqsResponse":
            sec_data_parser = self._parse_equity_screening_data
        elif rtype == "BQLResponse":
            sec_data_parser = self._parse_bql_data
        elif rtype == "fieldResponse":
            sec_data_parser = self._parse_field_info_data
        elif rtype == "InstrumentListResponse":
            sec_data_parser = self._parse_instrument_info_data
        elif rtype == "GetFillsResponse":
            sec_data_parser = self._parse_fills_data
        else:
            known_responses = (
                "ReferenceDataResponse",
                "HistoricalDataResponse",
                "IntradayBarResponse",
                "IntradayTickResponse",
                "BeqsResponse",
                "BQLResponse",
                "fieldResponse",
                "InstrumentListResponse",
                "GetFillsResponse",
            )
            raise TypeError(f"Unknown {rtype!r}, must be in {known_responses}")

        return sec_data_parser(response, request_data)

    @staticmethod
    def _parse_reference_security_data(response, request_data):
        rtype = list(response["message"]["element"].keys())[0]
        response_data = response["message"]["element"][rtype]
        req_type = list(request_data.keys())[0]
        for sec_data in response_data:
            result = {
                "security": sec_data["securityData"]["security"],
                "fields": request_data[req_type]["fields"],
            }
            field_data = sec_data["securityData"]["fieldData"]["fieldData"]
            data = {}
            for field in field_data.keys():
                # bulk reference data
                if isinstance(field_data[field], list):
                    rows = []
                    for fd in field_data[field]:
                        datum = {}
                        for name, value in fd[field].items():
                            datum[name] = value
                        rows.append(datum)
                    data[field] = rows
                # reference data
                else:
                    data[field] = field_data[field]
            result["data"] = data
            yield result

    @staticmethod
    def _parse_historical_security_data(response, request_data):
        rtype = list(response["message"]["element"].keys())[0]
        response_data = [response["message"]["element"][rtype]]
        req_type = list(request_data.keys())[0]
        for sec_data in response_data:
            result = {
                "security": sec_data["securityData"]["security"],
                "fields": request_data[req_type]["fields"],
            }
            field_data = sec_data["securityData"]["fieldData"]
            data = []
            for fd in field_data:
                data.append(fd["fieldData"])
            result["data"] = data
            yield result

    @staticmethod
    def _parse_bar_security_data(response, request_data):
        rtype = list(response["message"]["element"].keys())[0]
        bar_data = response["message"]["element"][rtype]["barData"]["barTickData"]
        data = []
        for bd in bar_data:
            data.append(bd["barTickData"])
        req_type = list(request_data.keys())[0]
        result = {
            "security": request_data[req_type]["security"],
            "data": data,
            "events": [request_data[req_type]["eventType"]],
        }
        yield result

    @staticmethod
    def _parse_tick_security_data(response, request_data):
        rtype = list(response["message"]["element"].keys())[0]
        bar_data = response["message"]["element"][rtype]["tickData"]["tickData"]
        data = []
        for bd in bar_data:
            data.append(bd["tickData"])
        req_type = list(request_data.keys())[0]
        result = {
            "security": request_data[req_type]["security"],
            "data": data,
            "events": request_data[req_type]["eventTypes"],
        }
        yield result

    @staticmethod
    def _parse_equity_screening_data(response, _):
        rtype = list(response["message"]["element"].keys())[0]
        response_data = response["message"]["element"][rtype]["data"]
        fields = list(response_data["fieldDisplayUnits"]["fieldDisplayUnits"].keys())

        for sec_data in response_data["securityData"]:
            result = {
                "security": sec_data["securityData"]["security"],
                "fields": fields,
                "data": sec_data["securityData"]["fieldData"]["fieldData"],
            }
            yield result

    @staticmethod
    def _parse_bql_data(response, _):
        rtype = list(response["message"]["element"].keys())[0]
        response_data = response["message"]["element"][rtype]["results"]

        for field in response_data.values():
            # ID column may be a security ticker
            field_data = {
                "field": field["name"],
                "id": field["idColumn"]["values"],
                "value": field["valuesColumn"]["values"],
            }

            # Secondary columns may be DATE or CURRENCY, for example
            for secondary_column in field["secondaryColumns"]:
                field_data[secondary_column["name"]] = secondary_column["values"]

            yield field_data

    @staticmethod
    def _parse_field_info_data(response, request_data):
        rtype = "fieldResponse"
        field_data = response["message"]["element"][rtype]
        data = {}
        for fd in field_data:
            datum = fd["fieldData"]["fieldInfo"]["fieldInfo"]
            data[fd["fieldData"]["id"]] = datum
        if "FieldInfoRequest" in request_data:
            ids = request_data["FieldInfoRequest"]["id"]
        else:
            ids = list(data.keys())
        result = {"id": ids, "data": data}
        yield result

    @staticmethod
    def _parse_instrument_info_data(response, _):
        data = response["message"]["element"]["InstrumentListResponse"]["results"]
        for datum in data:
            result = datum["results"]
            yield result

    @staticmethod
    def _parse_fills_data(response, _):
        data = response["message"]["element"]["GetFillsResponse"]["Fills"]
        result = []
        for datum in data:
            result.append(datum["Fills"])
        yield {"Fills": result}


def create_query(
    request_type: str, values: Dict, overrides: Optional[Sequence] = None
) -> Dict:
    """Create a request dictionary used to construct a blpapi.Request.

    Args:
        request_type: Type of request
        values: key value pairs to set in the request
        overrides: List of tuples containing the field to override and its value

    Returns: A dictionary representation of a blpapi.Request

    Examples:

        Reference data request

        >>> create_query(
        ...   'ReferenceDataRequest',
        ...   {'securities': ['CL1 Comdty', 'CO1 Comdty'], 'fields': ['PX_LAST']}
        ... )
        {'ReferenceDataRequest': {'securities': ['CL1 Comdty', 'CO1 Comdty'], 'fields': ['PX_LAST']}}

        Reference data request with overrides

        >>> create_query(
        ...   'ReferenceDataRequest',
        ...   {'securities': ['AUD Curncy'], 'fields': ['SETTLE_DT']},
        ...   [('REFERENCE_DATE', '20180101')]
        ... )  # noqa: E501
        {'ReferenceDataRequest': {'securities': ['AUD Curncy'], 'fields': ['SETTLE_DT'], 'overrides': [{'overrides': {'fieldId': 'REFERENCE_DATE', 'value': '20180101'}}]}}

        Historical data request

        >>> create_query(
        ...   'HistoricalDataRequest',
        ...   {
        ...    'securities': ['CL1 Comdty'],
        ...    'fields': ['PX_LAST', 'VOLUME'],
        ...    'startDate': '20190101',
        ...    'endDate': '20190110'
        ...   }
        ... )  # noqa: E501
        {'HistoricalDataRequest': {'securities': ['CL1 Comdty'], 'fields': ['PX_LAST', 'VOLUME'], 'startDate': '20190101', 'endDate': '20190110'}}

    """
    request_dict: Dict = {request_type: {}}
    for key in values:
        request_dict[request_type][key] = values[key]
    ovrds = []
    if overrides:
        for field, value in overrides:
            ovrds.append({"overrides": {"fieldId": field, "value": value}})
        request_dict[request_type]["overrides"] = ovrds
    return request_dict


def create_bql_query(
    expression: str,
    overrides: Optional[Sequence] = None,
    options: Optional[Dict] = None,
) -> Dict:
    """Create a sendQuery dictionary request.

    Args:
        expression: BQL query string

    Returns: A dictionary representation of a blpapi.Request
    """
    values = {"expression": expression}
    if options:
        values.update(options)
    return create_query("sendQuery", values, overrides)


def create_historical_query(
    securities: Union[str, Sequence[str]],
    fields: Union[str, Sequence[str]],
    start_date: str,
    end_date: str,
    overrides: Optional[Sequence] = None,
    options: Optional[Dict] = None,
) -> Dict:
    """Create a HistoricalDataRequest dictionary request.

    Args:
        securities: list of strings of securities
        fields: list of strings of fields
        start_date: start date as '%Y%m%d'
        end_date: end date as '%Y%m%d'
        overrides: List of tuples containing the field to override and its value
        options: key value pairs to to set in request

    Returns: A dictionary representation of a blpapi.Request

    """
    if isinstance(securities, str):
        securities = [securities]
    if isinstance(fields, str):
        fields = [fields]
    values = {
        "securities": securities,
        "fields": fields,
        "startDate": start_date,
        "endDate": end_date,
    }
    if options:
        values.update(options)
    return create_query("HistoricalDataRequest", values, overrides)


def create_reference_query(
    securities: Union[str, Sequence[str]],
    fields: Union[str, Sequence[str]],
    overrides: Optional[Sequence] = None,
    options: Optional[Dict] = None,
) -> Dict:
    """Create a ReferenceDataRequest dictionary request.

    Args:
        securities: list of strings of securities
        fields: list of strings of fields
        overrides: List of tuples containing the field to override and its value
        options: key value pairs to to set in request

    Returns: A dictionary representation of a blpapi.Request

    """
    if isinstance(securities, str):
        securities = [securities]
    if isinstance(fields, str):
        fields = [fields]
    values = {"securities": securities, "fields": fields}
    if options:
        values.update(options)
    return create_query("ReferenceDataRequest", values, overrides)
