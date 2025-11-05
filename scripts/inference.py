import torch
from typing import Any, Optional, Dict
from typing_extensions import override
from cs336_basics.Tokenizer import BPETokenizer
from langchain_core.language_models.llms import CallbackManagerForLLMRun, LLM
from langchain_core.outputs import GenerationChunk
from pydantic import Field


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