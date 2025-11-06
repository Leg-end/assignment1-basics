import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
from typing import Any, Optional, Dict
from typing_extensions import override
from cs336_basics.Tokenizer import BPETokenizer
from cs336_basics.Transformer import BasicsTransformerLM
from cs336_basics.qwen2_5 import Qwen2_5
from tests.adapters import run_load_checkpoint
from langchain_core.language_models.llms import CallbackManagerForLLMRun, LLM
from langchain_core.outputs import GenerationChunk
from pydantic import Field, BaseModel
from fastapi import FastAPI, HTTPException, Request, APIRouter
from fastapi.responses import StreamingResponse
from omegaconf import DictConfig
from rich.pretty import pprint as pprint
from rich.traceback import install
from contextlib import asynccontextmanager
import hydra
import uvicorn
import time
import logging

logger = logging.getLogger(__name__)

if torch.cuda.is_available():
    torch.set_float32_matmul_precision("high")

install(show_locals=True)


class GenerationRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: Optional[int] = None
    stop: Optional[list[str]] = None
    stream: bool = True
    
class GenerationResponse(BaseModel):
    text: str
    tokens_generated: int


class ChatBot(LLM):
    
    model: torch.nn.Module = Field(default=None)
    tokenizer: BPETokenizer = Field(default=None)
    device: str = Field(default="cuda" if torch.cuda.is_available() else "cpu")
    
    @override
    def _call(self,
              prompt: str,
              stop: Optional[list[str]] = None,
              run_manager: Optional[CallbackManagerForLLMRun] = None,
              **kwargs) -> str:
        input_ids = self.tokenizer.encode(prompt)
        input_tensor = torch.tensor([input_ids], device=self.device).to(torch.int64)
        if stop is not None:
            stop_token_ids = [self.tokenizer.encode(s) for s in stop]
        else:
            stop_token_ids = None
        output_tokens = self.model.generate(
            input_tensor,
            stop_token_ids=stop_token_ids,
            eos_token_id=self.tokenizer.eos_token_id,
            **kwargs
        )
        output_ids = output_tokens[0].cpu().numpy().tolist()
        text = self.tokenizer.decode(output_ids)
        return text
    
    @override
    def _stream(self,
                prompt: str,
                max_new_tokens: int,
                temperature: float = 1.0,
                top_k: int | None = 50,
                top_p: float | None = 0.9,
                repetition_penalty: float = 1.0,
                stop: Optional[list[str]] = None,
                run_manager: Optional[CallbackManagerForLLMRun] = None):
        input_ids = self.tokenizer.encode(prompt)
        input_ids = torch.tensor([input_ids], device=self.device).to(torch.int64)
        if stop is not None:
            stop_token_ids = [self.tokenizer.encode(s) for s in stop]
        else:
            stop_token_ids = None
        eos_token_id = self.tokenizer.eos_token_id
        
        idx = input_ids
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.model.context_length else idx[:, -self.model.context_length:]
            idx_next = self.model.sample(input_ids=idx_cond,
                                   full_ids=idx,
                                   temperature=temperature,
                                   top_k=top_k,
                                   top_p=top_p,
                                   repetition_penalty=repetition_penalty)
            idx = torch.cat((idx, idx_next), dim=1)
            
            if eos_token_id is not None:
                if idx_next.item() == eos_token_id:
                    break
            if stop_token_ids is not None:
                if idx_next.item() in stop_token_ids:
                    break
            
            chunk = GenerationChunk(text=self.tokenizer.decode([idx_next.item()]))
            if run_manager:
                run_manager.on_llm_new_token(chunk.text, chunk=chunk)
            yield chunk
            
    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Return a dictionary of identifying parameters."""
        return {
            # The model name allows users to specify custom token counting
            # rules in LLM monitoring applications (e.g., in LangSmith users
            # can provide per token pricing for their model and monitor
            # costs for the given LLM.)
            "model_name": self.model.__class__.__name__,
        }
        
    @property
    def _llm_type(self) -> str:
        """Get the type of language model used by this chat model. Used for logging purposes only."""
        return self.model.__class__.__name__
    
def create_router():
    router = APIRouter()
    @router.post("/generate", response_model=GenerationResponse)
    async def generate(request: Request, data: dict[Any, Any]):
        print(data)
        model_server: ChatBot = request.app.state.model_server
        prompt = data.pop("prompt")
        stream = data.pop("stream", False)
        if stream:
            return StreamingResponse(
                model_server.astream(
                    prompt,
                    **data
                ),
                media_type="text/plain"
            )
        else:
            text = await model_server.ainvoke(
                prompt,
                **data
            )
            return GenerationResponse(text=text, tokens_generated=len(text.split()))
        

    @router.post("/v1/chat/completions")
    async def chat_completions(request: Request, data: dict[Any, Any]):
        """兼容OpenAI格式的接口"""
        print(data)
        model_server: ChatBot = request.app.state.model_server
        stream = data.pop("stream", False)
        messages = data.pop("messages", [])
        prompt = messages[0]["content"]
        
        if stream:
            # 流式响应
            return StreamingResponse(
                model_server.astream(
                    prompt,
                    **data
                ),
                media_type="text/event-stream"
            )
        else:
            text = await model_server.ainvoke(
                prompt,
                **data
            )
            return GenerationResponse(text=text, tokens_generated=len(text.split()))


    @router.get("/health")
    async def health_check(request: Request):
        model_server = request.app.state.model_server
        return {"status": "healthy", "model_loaded": model_server is not None}
    return router


def create_app(model, tokenizer, device):
    """创建 FastAPI 应用工厂函数"""
    
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # 启动时初始化模型
        app.state.model_server = ChatBot(model=model, tokenizer=tokenizer, device=device)
        yield
        # 关闭时清理
        if hasattr(app.state, 'model_server'):
            del app.state.model_server
    
    app = FastAPI(lifespan=lifespan)
    
    # 注册路由
    router = create_router()
    app.include_router(router)
    return app


@hydra.main(config_path="configs/", config_name="evaluate_cs336_lm", version_base=None)
def main(cfg: DictConfig):
    model_config, eval_config, tokenizer_config = cfg.model, cfg.eval, cfg.tokenizer
    tokenizer = BPETokenizer.from_files(**tokenizer_config)
    logger.info(f"vocab size: {tokenizer.vocab_size}")
    if cfg.model_type == "qwen2_5":
        model = Qwen2_5(**model_config)
    else:
        model = BasicsTransformerLM(**model_config)
    pprint(model)
    torch.manual_seed(eval_config.seed)
    if torch.cuda.is_available():
        gpu_id = eval_config.get("gpu_id", 0)
        device = f"cuda:{gpu_id}"
    else:
        device = "cpu"
    model = model.to(device)
    model.eval()
    ckpt_path = os.path.join(eval_config.save_path, f"ckpt_iter{eval_config.iteration}.pt")
    iteration = run_load_checkpoint(ckpt_path, model)
    logger.info(f"Loading from checkpoint {iteration} from path {ckpt_path}")
    
    app = create_app(model, tokenizer, device)
    
    uvicorn.run(app,
                host="0.0.0.0",
                port=8000)


if __name__ == "__main__":
    main()
    
