import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import requests
import json
from scripts.printer import StreamPrinter
from typing import Optional, Union


def stream_generate(url: str,
                    prompt: Union[str, list[dict[str, str]]],
                    stream: bool = True):
    """调用普通生成流式接口"""
    data = {
        "max_new_tokens": 512,
        "temperature": 0.8,
        "top_p": 0.9,
        "repetition_penalty": 1.0,
        "stream": stream
    }
    if isinstance(prompt, str):
        data["prompt"] = prompt
    else:
        data["messages"] = prompt
        prompt = prompt[0]['content']
    try:
        response = requests.post(url, json=data, stream=stream, timeout=60)
        response.raise_for_status()
        if stream:
            printer = StreamPrinter()
            printer.update(prompt)
            for chunk in response.iter_content(decode_unicode=True, chunk_size=None):
                # print(chunk.decode('utf-8'))
                printer.update(chunk)
            printer.complete()
        else:
            result = response.json()
            print(result['text'])
            print(result["tokens_generated"])
    except requests.exceptions.HTTPError as e:
        print(f"HTTP Error: {e}")
        if hasattr(e, 'response') and e.response:
            print(f"Response content: {e.response.text}")
    except requests.exceptions.Timeout:
        print("Request timeout")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    # 测试普通流式生成
    print("=== 普通流式生成 ===")
    stream_generate("http://localhost:8000/generate", "Once upon a time")
    stream_generate("http://localhost:8000/generate", "Tom and Lily are friends")
    stream_generate("http://localhost:8000/generate", "Long long ago, there was a monk living in a temple")
    
    # print("\n\n=== 聊天流式生成 ===")
    # messages = [{"role": "user", "content": "Once upon a time"}]
    # stream_generate("http://localhost:8000/v1/chat/completions", messages)