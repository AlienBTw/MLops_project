import logging
from elasticsearch import Elasticsearch

class ElasticsearchHandler(logging.Handler):
    def __init__(self, hosts=None, index='mlflow-metrics', level=logging.NOTSET):
        super().__init__(level)
        self.index = index
        # Use the service name defined in docker-compose.yml
        self.es = Elasticsearch(hosts or ['http://elasticsearch:9200'])

    def emit(self, record):
        try:
            log_entry = self.format(record)
            self.es.index(index=self.index, body={"message": log_entry})
        except Exception:
            self.handleError(record)

# Setup logger for MLflow events or other monitoring signals
logger = logging.getLogger("MLflowLogger")
logger.setLevel(logging.INFO)
es_handler = ElasticsearchHandler()
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
es_handler.setFormatter(formatter)
logger.addHandler(es_handler)