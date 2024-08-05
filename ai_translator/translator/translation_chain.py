from langchain.chains import LLMChain
from langchain_community.llms.chatglm3 import ChatGLM3
from utils import LOG
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain import PromptTemplate, LLMChain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from functools import partial
import importlib.util
import logging
from typing import Any, List, Mapping, Optional, Tuple

from pydantic import Extra

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM

DEFAULT_MODEL_ID = "THUDM/chatglm2-6b"

logger = logging.getLogger(__name__)


class ChatGLMPipeline(LLM):
    """Wrapper around ChatGLM Pipeline API.

    To use, you should have the ``transformers`` python package installed.

    Example using from_model_id:
        .. code-block:: python

            from langchain.llms import ChatGLM
            hf = ChatGLM.from_model_id(
                model_id="THUDM/chatglm2-6b",
                model_kwargs={"trust_remote_code": True, device='cuda'}
            )
    """

    model: Any  #: :meta private:
    tokenizer: Any  # : :meta private:
    histoty: List[Tuple[str, str]] = []
    model_id: str = DEFAULT_MODEL_ID
    """Model name to use."""
    model_kwargs: Optional[dict] = None
    """Key word arguments passed to the model."""
    streaming: bool = True
    """Whether to stream the results, token by token."""

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    @classmethod
    def from_model_id(
        cls,
        model_id: str,
        device: int = -1,
        model_kwargs: Optional[dict] = None,
        **kwargs: Any,
    ) -> LLM:
        """Construct the pipeline object from model_id and task."""
        try:
            from transformers import (
                AutoModel,
                AutoTokenizer,
            )
        except ImportError:
            raise ValueError(
                "Could not import transformers python package. "
                "Please install it with `pip install transformers`."
            )

        _model_kwargs = model_kwargs or {}
        tokenizer = AutoTokenizer.from_pretrained(model_id, **_model_kwargs)

        def get_model(_model_kwargs):
            mdl = None  # Initialize mdl as None instead of 0
            quantize = _model_kwargs.pop("quantize", -1)
            cuda = _model_kwargs.pop("device", "") == "cuda"
            flt = _model_kwargs.pop("float", False)

            cuda_device_count = 0
            if cuda and importlib.util.find_spec("torch") is not None:
                import torch
                cuda_device_count = torch.cuda.device_count()
                if device < -1 or (device >= cuda_device_count):
                    raise ValueError(
                        f"Got device=={device}, "
                        f"device is required to be within [-1, {cuda_device_count})"
                    )
                if device < 0 and cuda_device_count > 0:
                    logger.warning(
                        "Device has %d GPUs available. "
                        "Provide device={deviceId} to `from_model_id` to use available"
                        "GPUs for execution. deviceId is -1 (default) for CPU and "
                        "can be a positive integer associated with CUDA device id.",
                        cuda_device_count,
                    )

            if quantize > -1 and flt and cuda:
                mdl = AutoModel.from_pretrained(model_id, **_model_kwargs).float().quantize(quantize).cuda()
            elif quantize > -1 and cuda:
                mdl = AutoModel.from_pretrained(model_id, **_model_kwargs).quantize(quantize).cuda()
            elif cuda:
                mdl = AutoModel.from_pretrained(model_id, **_model_kwargs).cuda()
            else:
                mdl = AutoModel.from_pretrained(model_id, **_model_kwargs)

            if mdl is None:
                raise ValueError("Failed to load the model. Please check the provided arguments.")

            return mdl.eval()

        try:
            model = get_model(_model_kwargs)
        except ImportError as e:
            raise ValueError(
                f"Could not load the {model_id} model due to missing dependencies."
            ) from e
        if "trust_remote_code" in _model_kwargs:
            _model_kwargs = {
                k: v for k, v in _model_kwargs.items() if k != "trust_remote_code"
            }
        return cls(
            model=model,
            model_id=model_id,
            tokenizer=tokenizer,
            model_kwargs=_model_kwargs,
            **kwargs,
        )

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {
            "model_id": self.model_id,
            "model_kwargs": self.model_kwargs
        }

    @property
    def _llm_type(self) -> str:
        return "chatglm_pipeline"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        if self.streaming:
            return self.stream(prompt=prompt, stop=stop, run_manager=run_manager)
        else:
            response, history = self.model.chat(
                self.tokenizer, prompt, history=self.histoty, return_past_key_values=True)
            return response

    def stream(
        self,
        prompt,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> str:
        current_length = 0
        text_callback = None
        if run_manager:
            text_callback = partial(
                run_manager.on_llm_new_token, verbose=self.verbose)
        text = ""
        for response, history, past_key_values in self.model.stream_chat(self.tokenizer, prompt, history=self.histoty, return_past_key_values=True):
            if text_callback:
                text_callback(response[current_length:])
            text += response[current_length:]
            current_length = len(response)
        return text


class TranslationChain:
    def __init__(self, endpoint_url: str = "http://127.0.0.1:8000", verbose: bool = True):
        
        # 翻译任务指令始终由 System 角色承担
        template = (
            """You are a translation expert, proficient in various languages. \n
            Translates {source_language} to {target_language}."""
        )
        system_message_prompt = SystemMessagePromptTemplate.from_template(template)

        # 待翻译文本由 Human 角色输入
        human_template = "{text}"
        human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

        # 使用 System 和 Human 角色的提示模板构造 ChatPromptTemplate
        chat_prompt_template = ChatPromptTemplate.from_messages(
            [system_message_prompt, human_message_prompt]
        )

        callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])


        # 使用 chatglm-6b 进行翻译
        chat = ChatGLMPipeline.from_model_id(
            model_id="THUDM/chatglm2-6b",
            device=-1, # if use GPU set to 0
            model_kwargs={"temperature": 0, "max_length": 5000, "trust_remote_code": True},
            callback_manager=callback_manager, 
            verbose=True,
        )
        self.chain = LLMChain(llm=chat, prompt=chat_prompt_template, verbose=verbose)

    def run(self, text: str, source_language: str, target_language: str) -> (str, bool):
        result = ""
        try:
            result = self.chain.run({
                "text": text,
                "source_language": source_language,
                "target_language": target_language,
            })
        except Exception as e:
            LOG.error(f"An error occurred during translation: {e}")
            return result, False

        return result, True