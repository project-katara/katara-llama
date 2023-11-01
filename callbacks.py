from langchain.callbacks.base import BaseCallbackHandler

class MyCustomSyncHandler(BaseCallbackHandler):
    def __init__(self, redisClient):
        self.message = ''
        self.redisClient = redisClient

    def on_llm_new_token(self, token: str, **kwargs) -> Any:
        self.message += token
        self.redisClient.publish(f'{kwargs["tags"][0]}',  self.message)

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> Any:
        self.redisClient.publish(f'{kwargs["tags"][0]}',  'end')

    def on_llm_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> Any:
        self.redisClient.publish(f'{kwargs["tags"][0]}',  'end')

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> Any:
        self.redisClient.publish(f'{kwargs["tags"][0]}',  'end')
